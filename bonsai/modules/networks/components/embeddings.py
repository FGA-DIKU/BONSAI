import torch
import torch.nn as nn


class EhrEmbeddings(nn.Module):
    """
    Forward inputs:
        input_ids: torch.LongTensor             - (batch_size, sequence_length)
        token_type_ids: torch.LongTensor        - (batch_size, sequence_length)
        position_ids: dict(str, torch.Tensor)   - (batch_size, sequence_length)
            We abuse huggingface's standard position_ids to pass additional information (age, abspos)
            This makes BertModel's forward method compatible with our EhrEmbeddings

    Parameters:
        vocab_size: int                         - size of the vocabulary
        hidden_size: int                        - size of the hidden layer
        type_vocab_size: int                    - size of max segments
        embedding_dropout: float                - dropout probability
        pad_token_id: int                       - token ID used for padding
        age_scale: float                        - scaling factor for age embeddings
        abspos_scale: float                     - scaling factor for absolute position embeddings
        age_shift: float                        - shift value for age embeddings
        abspos_shift: float                     - shift value for absolute position embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        type_vocab_size: int,
        embedding_dropout: float,
        pad_token_id: int = 0,
        age_scale: float = 1e-2,
        abspos_scale: float = 1e-4,
        age_shift: float = -50,
        abspos_shift: float = -200_000,
    ):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(embedding_dropout)

        # Initialize embeddings
        self.code_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.segment_embedding = nn.Embedding(type_vocab_size, hidden_size)
        self.age_embedding = Time2Vec(
            hidden_size,
            shift=age_shift,
            scale=age_scale,
            clip_min=-100,
            clip_max=100,
        )
        self.abspos_embedding = Time2Vec(
            hidden_size,
            shift=abspos_shift,
            scale=abspos_scale,
            clip_min=-100,
            clip_max=100,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # code
        segments: torch.LongTensor = None,
        age: torch.Tensor = None,
        abspos: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        embeddings = self.code_embedding(input_ids)

        embeddings += self.segment_embedding(segments)
        embeddings += self.age_embedding(age)
        embeddings += self.abspos_embedding(abspos)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Time2Vec(torch.nn.Module):
    """Time2Vec embedding layer that combines linear and periodic components.

    This layer transforms temporal inputs using a combination of linear and periodic embeddings:
    - First component (i=0): linear transformation w0*t + phi0
    - Remaining components: periodic transformations f(w*t + phi)

    The input can optionally be shifted and scaled before transformation, and the linear
    component can be clipped to a specified range.

    Parameters:
        output_dim: int
            Dimension of the output embedding vector. Default: 768
        function: callable
            Periodic function to use (e.g., torch.cos). Default: torch.cos
        init_scale: float
            Scaling factor applied to input values before transformation. Default: 1
        clip_min: float, optional
            Minimum value for clipping the linear component
        clip_max: float, optional
            Maximum value for clipping the linear component
        shift: float, optional
            Constant shift applied to input values before scaling and transformation

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
        shift: float = 0,
        scale: float = 1,
        clip_min: float = None,
        clip_max: float = None,
    ):
        """
        Parameters:
            output_dim: int - dimension of the output
            function: callable - function to use for the time2vec transformation
            init_scale: float - scale of the initial parameters
            clip_min: float - minimum value of the output
            clip_max: float - maximum value of the output
            shift: float - shift of the output
        """
        super().__init__()
        self.f = function
        self.clip_min = clip_min
        self.clip_max = clip_max
        # for i = 0
        self.w0 = torch.nn.Parameter(torch.randn(1, 1))
        self.phi0 = torch.nn.Parameter(torch.randn(1))
        # for 1 <= i <= k (output_dim)
        self.w = torch.nn.Parameter(torch.randn(1, output_dim - 1))
        self.phi = torch.nn.Parameter(torch.randn(output_dim - 1))
        self.shift = shift
        self.scale = scale

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = (tau + self.shift) * self.scale

        tau = tau.unsqueeze(2)  # (batch_size, sequence_length, 1)

        linear_1 = torch.matmul(tau, self.w0) + self.phi0
        linear_2 = torch.matmul(tau, self.w)

        if self.clip_min is not None or self.clip_max is not None:
            linear_1 = torch.clamp(linear_1, self.clip_min, self.clip_max)

        periodic = self.f(linear_2 + self.phi)

        return torch.cat((linear_1, periodic), dim=-1)
