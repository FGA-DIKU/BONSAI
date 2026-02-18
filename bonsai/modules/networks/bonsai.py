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
        embeddings (EhrEmbeddings): custom embeddings for concepts, segments, age, and absolute position.
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
            age_scale=config.age_scale,  # TODO: Move this from constants into configs
            age_shift=config.age_shift,  # TODO: Move this from constants into configs
            abspos_scale=config.abspos_scale,  # TODO: Move this from constants into configs
            abspos_shift=config.abspos_shift,  # TODO: Move this from constants into configs
        )
        self.is_causal = getattr(config, "is_causal", False)

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass building embeddings and attention mask, then calling ModernBertModel.

        Args:
            batch (dict): must contain:
                - "concept": Tensor of token indices (B, L)
                - "segment": Tensor of segment IDs (B, L)
                - "age": Tensor of patient ages (B, L)
                - "abspos": Tensor of absolute position values (B, L)
            **kwargs: Additional arguments to pass to the ModernBertModel forward method

        Returns:
            BaseModelOutput: output of ModernBertModel with last_hidden_state, etc.
        """
        # TODO: Abstract this out of the forward method and into the getitem.
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
        else:
            attention_mask = (batch["concept"] != 0).float()

        inputs_embeds = self.embeddings(
            input_ids=batch["concept"],
            segments=batch["segment"],
            age=batch["age"],
            abspos=batch["abspos"],
        )

        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def _update_attention_mask(
        self, attention_mask: torch.Tensor, output_attentions: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls super()._update_attention_mask and adds causal masking if self.is_causal=True.
        Returns:
            Tuple of:
            - Global attention mask
            - Sliding window mask for local attention
        """
        global_attention_mask, sliding_window_mask = super()._update_attention_mask(
            attention_mask, output_attentions
        )
        if self.is_causal:
            global_attention_mask = make_attention_causal(global_attention_mask)
            sliding_window_mask = make_attention_causal(sliding_window_mask)

        return global_attention_mask, sliding_window_mask


class BonsaiPretrain(BonsaiEncoder):
    """
    Masked Language Model head for EHR pretraining.

    Adds a prediction head and linear decoder on top of CorebehrtEncoder.
    """

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
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]
        logits = self.decoder(self.head(last_hidden_state))
        return logits, labels


class BonsaiFinetune(BonsaiEncoder):
    """
    Fine-tuning head for downstream classification on EHR sequences.

    Adds a binary classification head (BCEWithLogits) on top of sequence outputs.
    """

    def __init__(self, config):
        super().__init__(config)
        self.cls = FineTuneHead(hidden_size=config.hidden_size)

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass for fine-tuning.

        Args:
            batch (dict): must contain 'concept', 'segment', 'age', 'abspos', 'attention_mask';
                          optional 'target' as labels.
            **kwargs: Additional arguments to pass to the encoder forward method

        Returns:
            BaseModelOutput: with logits and optional loss if target provided.
        """
        outputs = super().forward(batch, **kwargs)

        sequence_output = outputs[0]  # Last hidden state
        logits = self.cls(sequence_output, batch["attention_mask"])
        return logits
