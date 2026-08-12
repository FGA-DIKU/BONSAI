import importlib.util

import torch.nn as nn

from bonsai.modules.networks.components.embeddings import EhrEmbeddings
from bonsai.modules.networks.components.layer import TransformerLayer

_FLASH_ATTENTION_AVAILABLE = importlib.util.find_spec("flash_attn") is not None

if _FLASH_ATTENTION_AVAILABLE:
    from flash_attn.bert_padding import pad_input, unpad_input


class BonsaiBase(nn.Module):
    def __init__(
        self,
        # Embedding / vocab
        vocab_size,
        max_seqlen,
        # Model dimensions
        hidden_size,
        num_layers,
        num_attention_heads,
        # Attention / behavior
        bias,
        dropout,
        attention_dropout,
        causal,
        attn_type,
    ):
        if attn_type == "flash" and not _FLASH_ATTENTION_AVAILABLE:
            raise ImportError(
                "flash_attn is not available. Please install flash-attn or use `attn_type='sdpa'` instead."
            )
        if attn_type not in ["flash", "sdpa"]:
            raise ValueError(
                f"Invalid attn_type '{attn_type}'. Must be one of ['flash', 'sdpa']."
            )
        super().__init__()
        self.embeddings = EhrEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seqlen=max_seqlen,
        )
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    bias=bias,
                    max_seqlen=max_seqlen,
                    causal=causal,
                    attn_type=attn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(hidden_size, bias=bias)

        self.hparams = {
            "vocab_size": vocab_size,
            "max_seqlen": max_seqlen,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "bias": bias,
            "dropout": dropout,
            "causal": causal,
            "attn_type": attn_type,
        }

    def forward(self, batch):
        x = self.embeddings(
            code=batch["code"],
            age=batch["age"],
            abspos=batch["abspos"],
            segment=batch["segment"],
        )

        x = self.drop(x)

        if self.hparams["attn_type"] == "flash":
            return self.forward_flash(x, batch["attention_mask"])
        elif self.hparams["attn_type"] == "sdpa":
            return self.forward_sdpa(
                x,
                attn_mask=batch["attention_mask"][:, None, None, :]
                if not self.hparams["causal"]
                else None,  # Causal SDPA needs attn_mask to be None
            )

    def forward_sdpa(self, x, attn_mask):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.layernorm(x)

        return x

    def forward_flash(self, x, attn_mask):
        batch_size, padded_seqlen = x.shape[:2]
        (x, indices, cu_seqlens, max_seqlen, _) = unpad_input(
            x, attn_mask
        )  # x: (batch, padded_seqlen, hidden_size) -> (total_valid_tokens, hidden_size)

        for layer in self.layers:
            x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        x = self.layernorm(x)

        x = pad_input(
            x, indices, batch_size, padded_seqlen
        )  # x: (total_valid_tokens, hidden_size) -> (batch, padded_seqlen, hidden_size)

        return x


class BonsaiPretrain(BonsaiBase):
    def __init__(
        self,
        # Embedding / vocab
        vocab_size,
        max_seqlen,
        # Model dimensions
        hidden_size,
        num_layers,
        num_attention_heads,
        # Attention / behavior
        bias,
        dropout,
        attention_dropout,
        causal,
        attn_type,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seqlen=max_seqlen,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            bias=bias,
            dropout=dropout,
            attention_dropout=attention_dropout,
            causal=causal,
            attn_type=attn_type,
        )
        self.pretrain_head = nn.Linear(hidden_size, vocab_size, bias=bias)

        # Weight tying (shares weights from code embedding to pretrain head)
        self.pretrain_head.weight = self.embeddings.code_embedding.weight

    def forward(self, batch: dict):
        last_hidden_state = super().forward(batch)
        labels = batch["target"]

        # Predicts only on the non-masked tokens
        mask = labels != -100
        last_hidden_state = last_hidden_state[mask]
        labels = labels[mask]

        logits = self.pretrain_head(last_hidden_state)
        return logits, labels


class BonsaiFinetune(BonsaiBase):
    def __init__(
        self,
        # Embedding / vocab
        vocab_size,
        max_seqlen,
        # Model dimensions
        hidden_size,
        num_layers,
        num_attention_heads,
        # Attention / behavior
        bias,
        dropout,
        attention_dropout,
        causal,
        attn_type,
        # Misc
        predict_token_id,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seqlen=max_seqlen,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            bias=bias,
            dropout=dropout,
            attention_dropout=attention_dropout,
            causal=causal,
            attn_type=attn_type,
        )
        self.hparams["predict_token_id"] = predict_token_id
        self.finetune_head = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, batch: dict):
        last_hidden_state = super().forward(batch)

        # Extracts the hidden states corresponding to the predict token for each subject
        pred_tokens = batch["code"] == self.hparams["predict_token_id"]
        last_hidden_state = last_hidden_state[pred_tokens]

        logits = self.finetune_head(last_hidden_state)

        return logits
