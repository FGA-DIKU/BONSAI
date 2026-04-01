import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    MaskedLMOutput,
    SequenceClassifierOutput,
    BaseModelOutput,
)
from transformers import PretrainedConfig, PreTrainedModel


class BasicTransformerConfig(PretrainedConfig):
    model_type = "basic_transformer"

    def __init__(
        self,
        vocab_size=50368,
        hidden_size=768,
        num_hidden_layers=4,  # Kept small for a "basic" implementation
        num_attention_heads=8,
        intermediate_size=3072,
        max_position_embeddings=4096,
        pad_token_id=0,
        num_labels=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_labels = num_labels


class BasicTransformerModel(PreTrainedModel):
    """
    Base model returning hidden states.
    Equivalent to ModernBertModel.
    """

    config_class = BasicTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Using standard PyTorch TransformerEncoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # BONSAI might inject pre-computed embeddings (e.g., combined token/age/segment embeddings)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            seq_length = input_ids.size(1)
            if position_ids is None:
                position_ids = torch.arange(
                    seq_length, dtype=torch.long, device=input_ids.device
                )
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            inputs_embeds = self.embeddings(input_ids) + self.position_embeddings(
                position_ids.long()
            )

        # PyTorch TransformerEncoder expects src_key_padding_mask where True = padding/ignore
        # HuggingFace attention_mask uses 1 = attend, 0 = ignore
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        hidden_states = self.encoder(
            inputs_embeds, src_key_padding_mask=src_key_padding_mask
        )

        if not return_dict:
            return (hidden_states,)

        return BaseModelOutput(last_hidden_state=hidden_states)


class BasicTransformerForMaskedLM(PreTrainedModel):
    """
    Used for pretraining tasks in BONSAI.
    Equivalent to ModernBertForMaskedLM.
    """

    config_class = BasicTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = BasicTransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        attention_mask = input_ids["attention_mask"]
        position_ids = None
        labels = input_ids["target"]
        input_ids = input_ids["code"]
        inputs_embeds = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits, labels
        # loss = None
        # if labels is not None:
        #    loss_fct = nn.CrossEntropyLoss()
        #    # Flatten the logits and labels to compute cross-entropy loss
        #    loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


#
# if not return_dict:
#    output = (logits,) + outputs[1:]
#    return ((loss,) + output) if loss is not None else output
#
# return MaskedLMOutput(
#    loss=loss,
#    logits=logits,
#    hidden_states=outputs.hidden_states,
#    attentions=outputs.attentions,
# )


class BasicTransformerForSequenceClassification(PreTrainedModel):
    """
    Used for fine-tuning on outcomes in BONSAI.
    Equivalent to ModernBertForSequenceClassification.
    """

    config_class = BasicTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = BasicTransformerModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # For a basic transformer, pool by extracting the first token (mimics [CLS] behavior)
        pooled_output = hidden_states[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
