import torch.nn as nn
from bonsai.modules.networks.components.embeddings import EhrEmbeddings
from bonsai.modules.networks.components.blocks import TransformerBlock


class BonsaiFlashEncoder(nn.Module):
    def __init__(self, config):
        self.embeddings = EhrEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_seqlen=config.max_seqlen,
            pad_token_id=config.pad_token_id,
        )

        self.encoder = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                layers=nn.ModuleList(
                    [
                        TransformerBlock(
                            hidden_size=config.hidden_size,
                            num_heads=config.num_attention_heads,
                            dropout=config.dropout,
                            bias=config.bias,
                            max_seqlen=config.max_seqlen,
                            causal=config.causal
                        )
                        for _ in range(config.num_layers)
                    ]
                ),
                layernorm=nn.LayerNorm(config.hidden_size, bias=config.bias)
            )
        )

    def forward(self, batch):
        x = self.embeddings(
            code=batch["code"],
            age=batch["age"],
            abspos=batch["abspos"],
            segment=batch["segment"]
        )

        x = self.encoder.drop(x)
        for block in self.encoder.layers:
            x = block(x, cu_seqlens=batch["cu_seqlens"])
        x = self.encoder.layernorm(x)

        return x
            
