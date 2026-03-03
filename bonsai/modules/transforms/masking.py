import torch
from typing import Tuple


class CodeMasker:
    def __init__(
        self,
        vocabulary: dict,
        select_ratio: float,
        masking_ratio: float = 0.8,
        random_ratio: float = 0.1,
        ignore_special_tokens: bool = True,
    ) -> None:
        """Mask codes for MLM.
        Args:
            vocabulary: Vocabulary
            select_ratio: Ratio of tokens to consider in the loss
            masking_ratio: Ratio of tokens to replace with [MASK]
            random_ratio: Ratio of tokens to replace with random word
        """

        self.vocabulary = vocabulary
        self.n_special_tokens = (
            len([token for token in vocabulary if token.startswith("[")])
            if ignore_special_tokens
            else 0
        )
        self.select_ratio = select_ratio
        self.masking_ratio = masking_ratio
        self.random_ratio = random_ratio

    def mask_patient_codes(
        self, codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = codes.clone()
        probability_vector = torch.full(target.shape, self.select_ratio)

        # Ignore special tokens
        special_token_mask = codes < self.n_special_tokens
        probability_vector.masked_fill_(special_token_mask, value=0.0)

        # Get MLM mask
        selected_indices = torch.bernoulli(probability_vector).bool()
        target[~selected_indices] = -100

        # Replace with [MASK]
        indices_mask = (
            torch.bernoulli(torch.full(target.shape, self.masking_ratio)).bool()
            & selected_indices
        )
        codes[indices_mask] = self.vocabulary["[MASK]"]

        # Replace with random word and Account for already masked tokens
        random_ratio = self.random_ratio / (1 - self.masking_ratio)
        indicies_random = (
            torch.bernoulli(torch.full(target.shape, random_ratio)).bool()
            & selected_indices
            & ~indices_mask
        )
        random_words = torch.randint(
            self.n_special_tokens,
            len(self.vocabulary),
            target.shape,
            dtype=codes.dtype,
        )
        codes[indicies_random] = random_words[indicies_random]
        return codes, target
