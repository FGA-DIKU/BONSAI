from typing import Optional

import torch
import torch.nn as nn


class EhrEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seqlen: int,
        value_embedding_mode: str = None,
    ):
        super().__init__()

        # Initialize embeddings
        self.code_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(max_seqlen, hidden_size, padding_idx=0)
        self.age_embedding = Time2Vec(hidden_size, clip_range=100)
        self.abspos_embedding = Time2Vec(hidden_size, clip_range=100)

        self.value_embedding_mode = value_embedding_mode
        if self.value_embedding_mode is not None:
            self.numeric_value_embedding = ContinuousEmbedding(
                hidden_size, self.value_embedding_mode
            )

    def forward(
        self,
        code: torch.LongTensor,
        age: torch.Tensor,
        abspos: torch.Tensor,
        segment: torch.LongTensor,
        numeric_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        embeddings = self.code_embedding(code)
        if self.value_embedding_mode is not None and numeric_value is not None:
            embeddings = self.numeric_value_embedding(numeric_value, embeddings)

        embeddings += self.age_embedding(age)
        embeddings += self.abspos_embedding(abspos)
        embeddings += self.segment_embedding(segment)

        return embeddings


class Time2Vec(nn.Module):
    """Time2Vec embedding layer that combines linear and periodic components.

    This layer transforms temporal inputs using a combination of linear and periodic embeddings:
    - First component (i=0): linear transformation w0*t + phi0
    - Remaining components: periodic transformations f(w*t + phi)

    The linear component can be clipped to a specified range.

    Parameters:
        output_dim: int
            Dimension of the output embedding vector. Default: 768
        function: callable
            Periodic function to use (e.g., torch.cos). Default: torch.cos
        clip_range: float, optional
            -Minimum/maximum value for clipping the linear component

    Forward Input:
        tau: torch.Tensor
            Input temporal values of shape (batch_size, sequence_length)

    Returns:
        torch.Tensor: Concatenated linear and periodic embeddings
            of shape (batch_size, sequence_length, output_dim)
    """

    def __init__(
        self,
        output_dim: int = 768,
        function: callable = torch.cos,
        clip_range: Optional[float] = None,
    ):
        """
        Parameters:
            output_dim: int - dimension of the output
            function: callable - function to use for the time2vec transformation
            clip_min: float - minimum value of the output
            clip_max: float - maximum value of the output
        """
        super().__init__()
        self.f = function
        self.clip_range = clip_range
        # for i = 0
        self.w0 = torch.nn.Parameter(torch.randn(1, 1))
        self.phi0 = torch.nn.Parameter(torch.randn(1))
        # for 1 <= i <= k (output_dim)
        self.w = torch.nn.Parameter(torch.randn(1, output_dim - 1))
        self.phi = torch.nn.Parameter(torch.randn(output_dim - 1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.unsqueeze(2)  # (batch_size, sequence_length, 1)

        linear_1 = torch.matmul(tau, self.w0) + self.phi0
        linear_2 = torch.matmul(tau, self.w)

        if self.clip_range is not None:
            linear_1 = torch.clamp(linear_1, -self.clip_range, self.clip_range)

        periodic = self.f(linear_2 + self.phi)

        return torch.cat((linear_1, periodic), dim=-1)


class ContinuousEmbedding(nn.Module):
    def __init__(self, hidden_size: int, value_embedding_mode: str = None):
        super().__init__()
        self.value_embedding_mode = value_embedding_mode
        self.hidden_size = hidden_size

        self.value_proj = nn.Sequential(
            nn.Linear(1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
        )

        if self.value_embedding_mode == "film":
            self.gamma_layer = nn.Linear(hidden_size, 1)
            self.beta_layer = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(
                f"Unknown value_embedding_mode: {self.value_embedding_mode}"
            )

    def forward(
        self, values: torch.Tensor, concept_embeds: torch.Tensor
    ) -> torch.Tensor:
        mask = (~torch.isnan(values)).float().unsqueeze(-1)
        values_safe = torch.where(torch.isnan(values), torch.zeros_like(values), values)
        value_embed = self.value_proj(values_safe.unsqueeze(-1)) * mask

        if self.value_embedding_mode == "film":
            gamma = self.gamma_layer(concept_embeds)
            beta = self.beta_layer(concept_embeds)
            fused = (gamma * value_embed + beta).to(dtype=concept_embeds.dtype)
            return fused * mask + concept_embeds * (1 - mask)

        raise ValueError(
            f"Unknown value_embedding_mode: {self.value_embedding_mode}"
        )
