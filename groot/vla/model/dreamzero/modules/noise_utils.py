from collections.abc import Iterable

import torch


def generate_noise_samples(
    shapes: Iterable[tuple[int, ...] | torch.Size],
    seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, ...]:
    """Draw reproducible samples without restarting the stream between shapes."""
    generator = (
        None if seed is None else torch.Generator(device=device).manual_seed(seed)
    )
    return tuple(
        torch.randn(shape, generator=generator, device=device, dtype=dtype)
        for shape in shapes
    )
