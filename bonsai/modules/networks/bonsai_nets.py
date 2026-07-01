import torch.nn as nn

from bonsai.modules.networks.components.embeddings import EhrEmbeddings

try:
    from bonsai.modules.networks.components.blocks import (
        FlashTransformerLayer as TransformerLayer,
    )

    _FLASH_ATTENTION_AVAILABLE = True
except:
    from bonsai.modules.networks.components.blocks_nofa import TransformerLayer

    _FLASH_ATTENTION_AVAILABLE = False


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
        causal,
    ):
        if not _FLASH_ATTENTION_AVAILABLE:
            print(
                "WARNING: flash_attn is not available. Falling back to standard pytorch implementation."
            )
        super().__init__()
        self.embeddings = EhrEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seqlen=max_seqlen,
        )
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(dropout),
                layers=nn.ModuleList(
                    [
                        TransformerLayer(
                            hidden_size=hidden_size,
                            num_heads=num_attention_heads,
                            dropout=dropout,
                            bias=bias,
                            max_seqlen=max_seqlen,
                            causal=causal,
                        )
                        for _ in range(num_layers)
                    ]
                ),
                layernorm=nn.LayerNorm(hidden_size, bias=bias),
            )
        )

    def forward(self, batch):
        x = self.embeddings(
            code=batch["code"],
            age=batch["age"],
            abspos=batch["abspos"],
            segment=batch["segment"],
        )

        x = self.transformer.drop(x)
        for block in self.transformer.layers:
            x = block(
                x, attn_mask=batch.get("attn_mask"), cu_seqlens=batch.get("cu_seqlens")
            )
        x = self.transformer.layernorm(x)

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
        causal,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seqlen=max_seqlen,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            bias=bias,
            dropout=dropout,
            causal=causal,
        )
        self.pretrain_head = nn.Linear(hidden_size, vocab_size, bias=bias)

        # Weight tying (shares weights from code embedding to pretrain head)
        self.embeddings.code_embedding.weight = self.pretrain_head.weight

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
        causal,
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
            causal=causal,
        )
        self.predict_token_id = predict_token_id
        self.finetune_head = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, batch: dict):
        last_hidden_state = super().forward(batch)

        # Extracts the hidden states corresponding to the predict token for each subject
        pred_tokens = batch["code"] == self.predict_token_id
        last_hidden_state = last_hidden_state[pred_tokens]

        logits = self.finetune_head(last_hidden_state)

        return logits
