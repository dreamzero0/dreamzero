import torch

from groot.vla.model.dreamzero.modules.noise_utils import generate_noise_samples


def test_seeded_modalities_consume_disjoint_reproducible_draws():
    video_shape = (1, 48, 1, 10, 20)
    action_shape = (1, 48, 32)

    video_noise, action_noise = generate_noise_samples(
        (video_shape, action_shape), seed=1140, dtype=torch.bfloat16
    )
    repeated_video_noise, repeated_action_noise = generate_noise_samples(
        (video_shape, action_shape), seed=1140, dtype=torch.bfloat16
    )

    assert torch.equal(video_noise, repeated_video_noise)
    assert torch.equal(action_noise, repeated_action_noise)

    generator = torch.Generator().manual_seed(1140)
    expected_video_noise = torch.randn(
        video_shape, generator=generator, dtype=torch.bfloat16
    )
    expected_action_noise = torch.randn(
        action_shape, generator=generator, dtype=torch.bfloat16
    )

    assert torch.equal(video_noise, expected_video_noise)
    assert torch.equal(action_noise, expected_action_noise)
    assert not torch.equal(
        video_noise.flatten()[: action_noise.numel()], action_noise.flatten()
    )
