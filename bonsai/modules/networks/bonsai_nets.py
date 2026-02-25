"""
Module: corebehrt_module

This module defines customized EHR-focused BERT models built on top of ModernBertModel:

- CorebehrtEncoder: replaces token embeddings with temporal EHR embeddings and causal encoder layers.
- CorebehrtForPretraining: extends the encoder for masked language model pretraining on EHR sequences.
- CorebehrtForFineTuning: extends the encoder for downstream classification/regression tasks on EHR data.
"""

from typing import Tuple
import torch
import torch.nn as nn
from bonsai.modules.networks.components.embeddings import EhrEmbeddings
from bonsai.modules.networks.components.heads import FineTuneHead
from bonsai.modules.networks.components.utils import make_attention_causal
from transformers import ModernBertModel
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead


class BonsaiEncoder(ModernBertModel):
    """
    Encoder backbone for EHR data using ModernBert.

    Attributes:
        embeddings (EhrEmbeddings): custom embeddings for codes, segments, age, and absolute position.
        layers (nn.ModuleList): list of causal encoder layers replacing standard BERT layers.
    """

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = EhrEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            embedding_dropout=config.embedding_dropout,
            pad_token_id=config.pad_token_id,
            age_scale=config.age_scale,
            age_shift=config.age_shift,
            abspos_scale=config.abspos_scale,
            abspos_shift=config.abspos_shift,
        )
        self.is_causal = config.is_causal

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass building embeddings and attention mask, then calling ModernBertModel.

        Args:
            batch (dict): must contain:
                - "codes": Tensor of token indices (B, L)
                - "segments": Tensor of segment IDs (B, L)
                - "ages": Tensor of patient ages (B, L)
                - "abspos": Tensor of absolute position values (B, L)
            **kwargs: Additional arguments to pass to the ModernBertModel forward method

        Returns:
            BaseModelOutput: output of ModernBertModel with last_hidden_state, etc.
        """
        # TODO: Abstract this out of the forward method and into the getitem.
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
        else:
            attention_mask = (batch["codes"] != 0).float()
        inputs_embeds = self.embeddings(
            input_ids=batch["codes"],
            segments=batch["segments"],
            age=batch["ages"],
            abspos=batch["abspos"],
        )

        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def _update_attention_mask(
        self, attention_mask: torch.Tensor, output_attentions: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        global_attention_mask, sliding_window_mask = super()._update_attention_mask(
            attention_mask, output_attentions
        )
        if self.is_causal:
            global_attention_mask = make_attention_causal(global_attention_mask)
            sliding_window_mask = make_attention_causal(sliding_window_mask)

        return global_attention_mask, sliding_window_mask


class BonsaiPretrain(BonsaiEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

    def forward(self, batch: dict, **kwargs):
        outputs = super().forward(batch, **kwargs)
        labels = batch["target"]
        last_hidden_state = outputs[0]

        if self.sparse_prediction:
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]
        logits = self.decoder(self.head(last_hidden_state))
        return logits, labels


class BonsaiFinetune(BonsaiEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.cls = FineTuneHead(hidden_size=config.hidden_size)

    def forward(self, batch: dict, **kwargs):
        outputs = super().forward(batch, **kwargs)
        sequence_output = outputs[0]  # Last hidden state
        logits = self.cls(sequence_output, batch["attention_mask"])
        return logits


if __name__ == "__main__":
    from bonsai.modules.networks.bonsai_nets import BonsaiPretrain
    from transformers import ModernBertConfig
    import omegaconf

    cfg = omegaconf.OmegaConf.load(
        "/Users/zcr545/Desktop/Projects/repos/BONSAI/bonsai/configs/pretrain.yaml"
    )
    model = BonsaiPretrain(
        ModernBertConfig(
            **cfg["model"],
            vocab_size=64,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            sparse_prediction=True,
        )
    )
