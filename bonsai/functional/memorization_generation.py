"""Probability sampling from a public token with no patient features."""

import time

import numpy as np
import torch


@torch.inference_mode()
def sample_codes(
    model, start_id, count, horizon, batch_size, temperature, seed, output
):
    """Sample full-softmax next tokens, appending each draw; only code inputs exist."""
    from bonsai.run.memorization_check import hidden_states

    if (
        min(count, horizon, batch_size) < 1
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Positive counts and finite positive temperature required.")
    if horizon > model.hparams["max_seqlen"]:
        raise ValueError("Seed plus generated inputs exceed checkpoint context.")
    device = next(model.parameters()).device
    random = torch.Generator(device=device).manual_seed(seed)
    samples = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.int32, shape=(count, horizon)
    )
    model.eval()
    started = time.perf_counter()
    for offset in range(0, count, batch_size):
        n = min(batch_size, count - offset)
        codes = torch.full((n, 1), start_id, dtype=torch.long, device=device)
        for step in range(horizon):
            batch = {
                "code": codes,
                "attention_mask": torch.ones_like(codes, dtype=torch.bool),
            }
            states = hidden_states(model, batch, code_only=True)
            logits = model.pretrain_head(states[:, -1]).float() / temperature
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite logits; sampled run is invalid.")
            # No top-k, token masks, patient features, or supplied separators.
            drawn = torch.multinomial(
                torch.softmax(logits, dim=-1), 1, generator=random
            )
            codes = torch.cat((codes, drawn), dim=1)
        # Transfer a completed batch once, rather than synchronize a CPU copy per token.
        samples[offset : offset + n] = codes[:, 1:].cpu().numpy()
        elapsed = time.perf_counter() - started
        remaining = elapsed * (count - offset - n) / (offset + n)
        print(
            f"Public-seed sampling: {offset + n:,}/{count:,} sequences; {elapsed / 60:.1f} min elapsed; estimated sampling remaining {remaining / 60:.1f} min (setup excluded; early batches include warm-up)",
            flush=True,
        )
    samples.flush()
    return samples
