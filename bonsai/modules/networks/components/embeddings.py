import math
from typing import Optional

import torch
import torch.nn as nn


class EhrEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seqlen: int,
        abspos_encoding: str = "scaled_time2vec",
    ):
        super().__init__()

        # Initialize embeddings
        self.code_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(max_seqlen, hidden_size, padding_idx=0)
        self.age_embedding = Time2Vec(hidden_size, clip_range=100)
        if abspos_encoding == "scaled_time2vec":
            # This branch stores absolute position in thousands of hours, so
            # ordinary Time2Vec exactly preserves the existing abs_pos model.
            self.abspos_embedding = Time2Vec(hidden_size, clip_range=100)
        elif abspos_encoding == "fourier":
            self.abspos_embedding = AbsposFourierEncoding(
                hidden_size, input_unit="thousand_hours"
            )
        else:
            raise ValueError(
                "Unknown abspos_encoding "
                f"{abspos_encoding!r}; expected 'scaled_time2vec' or 'fourier'."
            )
        self.abspos_encoding = abspos_encoding

    def forward(
        self,
        code: torch.LongTensor,
        age: torch.Tensor,
        abspos: torch.Tensor,
        segment: torch.LongTensor,
    ) -> torch.Tensor:

        embeddings = self.code_embedding(code)
        embeddings += self.age_embedding(age)
        embeddings += self.abspos_embedding(abspos)
        embeddings += self.segment_embedding(segment)

        return embeddings


class EhrValueEmbeddings(EhrEmbeddings):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seqlen: int,
        value_embedding_mode: str,
        abspos_encoding: str = "scaled_time2vec",
    ):
        super().__init__(
            vocab_size, hidden_size, max_seqlen, abspos_encoding=abspos_encoding
        )
        self.numeric_value_embedding = ContinuousEmbedding(
            hidden_size, value_embedding_mode
        )

    def forward(
        self,
        code: torch.LongTensor,
        age: torch.Tensor,
        abspos: torch.Tensor,
        segment: torch.LongTensor,
        numeric_value: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.code_embedding(code)
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


class AbsposFourierEncoding(nn.Module):
    """Fixed-frequency calendar encoding for absolute timestamps.

    The ``abs_pos`` branch stores timestamps in thousands of Unix-epoch hours;
    conversion back to hours here gives this module the same calendar semantics
    and state-dict layout as OPERA's Fourier implementation.
    """

    def __init__(
        self,
        output_dim: int = 768,
        min_period_years: float = 1.0,
        max_period_years: float = 80.0,
        linear_ref_years: float = 12.0,
        linear_scale_years: float = 15.0,
        input_unit: str = "thousand_hours",
    ):
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive.")
        if min_period_years <= 0 or max_period_years <= min_period_years:
            raise ValueError(
                "Period range must satisfy 0 < min_period_years < max_period_years."
            )
        if linear_scale_years <= 0:
            raise ValueError("linear_scale_years must be positive.")
        if input_unit not in {"hours", "thousand_hours"}:
            raise ValueError("input_unit must be 'hours' or 'thousand_hours'.")

        num_pairs = (output_dim - 1) // 2
        self.output_dim = output_dim
        self.num_pairs = num_pairs
        self.input_unit = input_unit
        self.register_buffer("epoch_2000_hours", torch.tensor(30.0 * 8766.0))
        self.register_buffer("hours_per_year", torch.tensor(8766.0))
        self.register_buffer("linear_ref", torch.tensor(float(linear_ref_years)))
        self.register_buffer("linear_scale", torch.tensor(float(linear_scale_years)))

        if num_pairs:
            periods = torch.logspace(
                math.log10(min_period_years),
                math.log10(max_period_years),
                num_pairs,
            )
            frequencies = (2.0 * math.pi) / periods
        else:
            periods = torch.empty(0)
            frequencies = torch.empty(0)
        self.register_buffer("periods", periods)
        self.register_buffer("frequencies", frequencies)
        self.phi = nn.Parameter(torch.zeros(num_pairs))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        output_dtype = self.phi.dtype
        with torch.autocast(device_type=tau.device.type, enabled=False):
            tau_hours = tau.float()
            if self.input_unit == "thousand_hours":
                tau_hours = tau_hours * 1000.0
            tau_years = (
                tau_hours - self.epoch_2000_hours.float()
            ) / self.hours_per_year.float()
            linear = (
                (tau_years - self.linear_ref.float()) / self.linear_scale.float()
            ).unsqueeze(-1)
            angles = (
                tau_years.unsqueeze(-1) * self.frequencies.float() + self.phi.float()
            )
            periodic = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
            output = torch.cat((linear, periodic.flatten(start_dim=-2)), dim=-1)
            if output.shape[-1] < self.output_dim:
                output = torch.cat((output, torch.zeros_like(linear)), dim=-1)
        return output.to(dtype=output_dtype)


class ContinuousEmbedding(nn.Module):
    def __init__(self, hidden_size: int, value_embedding_mode: str):
        super().__init__()
        self.value_embedding_mode = value_embedding_mode
        self.hidden_size = hidden_size

        self.value_proj = nn.Sequential(
            nn.Linear(1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
        )

        if self.value_embedding_mode == "film":
            self.gamma_layer = nn.Linear(hidden_size, hidden_size)
            self.beta_layer = nn.Linear(hidden_size, hidden_size)
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

        raise ValueError(f"Unknown value_embedding_mode: {self.value_embedding_mode}")
