"""CPU attention-contract checks for the mismatch reported in issue 61.

Run from the repository root with:
    python -m pytest -q tests/test_causal_state_attention.py

The source loader executes complete upstream definitions, unchanged, and the
upstream torch attention backend. It avoids importing unrelated model loaders.
This is not a full package, checkpoint, CUDA, TensorRT or robot qualification.
"""

import ast
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, create_mask

SOURCE = Path(__file__).resolve().parents[1]
MODULES = Path("groot/vla/model/dreamzero/modules")
MODEL = MODULES / "wan_video_dit_action_casual_chunk.py"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def attention_class(source):
    spec = importlib.util.spec_from_file_location(
        "dreamzero_test_attention", SOURCE / MODULES / "wan2_1_attention.py"
    )
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    namespace = {
        "torch": torch,
        "nn": nn,
        "Any": Any,
        "BlockMask": BlockMask,
        "create_block_mask": create_block_mask,
        "create_mask": create_mask,
        "ENABLE_TENSORRT": False,
        "AttentionModule": backend.AttentionModule,
    }
    definitions(
        SOURCE / MODULES / "wan2_1_submodule.py",
        [
            "WanRMSNorm",
            "rope_action_apply",
            "rope_action_apply_polar",
            "rope_action_apply_no_polar",
        ],
        namespace,
    )
    definitions(
        source / MODEL,
        [
            "causal_rope_action_apply",
            "causal_rope_action_apply_polar",
            "causal_rope_action_apply_no_polar",
            "CausalWanSelfAttention",
        ],
        namespace,
    )
    return namespace["CausalWanSelfAttention"], namespace["rope_action_apply"]


Attention, rope = attention_class(SOURCE)


@pytest.fixture(autouse=True)
def torch_backend(monkeypatch):
    monkeypatch.setenv("ATTENTION_BACKEND", "torch")


def model(*, heads=1, states=1, actions=2, dyadic=False):
    torch.manual_seed(835)
    layer = Attention(
        dim=8,
        num_heads=heads,
        frame_seqlen=1,
        num_action_per_block=actions,
        num_state_per_block=states,
        qk_norm=not dyadic,
    )
    if dyadic:
        with torch.no_grad():
            for projection in (layer.q, layer.k):
                projection.weight.zero_()
                projection.bias.zero_()
            for projection in (layer.v, layer.o):
                projection.weight.copy_(torch.eye(8))
                projection.bias.zero_()
    return layer.eval()


def case(model, *, batch=1, prior=1, dyadic=False):
    blocks, frames = prior, prior + 1
    actions, states = model.num_action_per_block, model.num_state_per_block
    width = model.head_dim // 2

    def frequencies(length):
        phase = torch.arange(length * width, dtype=torch.float64).reshape(length, width) / 10
        return torch.polar(torch.ones_like(phase), phase)

    fv, fa, fs = (
        frequencies(frames).unsqueeze(1),
        frequencies(blocks * actions),
        frequencies(blocks * states),
    )
    generator = torch.Generator().manual_seed(707)
    x = torch.randn(batch, 2 * frames + blocks * (actions + states), 8, generator=generator)
    if dyadic:
        x.zero_()
        x[:, -states:] = 6
    indices = [frames + prior]
    indices += list(range(2 * frames + (blocks - 1) * actions, 2 * frames + blocks * actions))
    state_start = 2 * frames + blocks * actions + (blocks - 1) * states
    indices += list(range(state_start, state_start + states))
    return dict(x=x, fv=fv, fa=fa, fs=fs, indices=indices, prior=prior)


def train(model, data):
    return model(
        data["x"],
        data["fv"],
        data["fa"],
        data["fs"],
        data["prior"] * (model.num_action_per_block + model.num_state_per_block),
        kv_cache=None,
        is_tf=True,
    )[0]


def cached(model, data, *, live=None, prefix=None):
    x, prior = data["x"], data["prior"]
    prefix = x[:, :prior] if prefix is None else prefix
    batch = x.shape[0]
    key = model.norm_k(model.k(prefix)).view(batch, prior, model.num_heads, model.head_dim)
    key = rope(key, data["fv"][:prior], data["fa"], data["fs"], None).type_as(x)
    value = model.v(prefix).view(batch, prior, model.num_heads, model.head_dim)
    cache = torch.stack([key, value])
    before = cache.clone()
    output, updated = model(
        x[:, data["indices"]] if live is None else live,
        data["fv"][prior : prior + 1],
        data["fa"],
        data["fs"],
        model.num_action_per_block + model.num_state_per_block,
        kv_cache=cache,
        current_start_frame=prior,
    )
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    return output, updated


def assert_close(left, right):
    torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_dyadic_state_visibility_witness():
    layer = model(actions=1, dyadic=True)
    data = case(layer, dyadic=True)
    expected = train(layer, data)[:, data["indices"]]
    actual, _ = cached(layer, data)
    # Uniform attention gives 6 within the state block, but 1.5 over all four tokens.
    torch.testing.assert_close(expected[:, -1], torch.full((1, 8), 6.0), rtol=0, atol=0)
    torch.testing.assert_close(actual[:, -1], torch.full((1, 8), 6.0), rtol=0, atol=0)
    assert_close(actual, expected)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("states", [1, 2])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("prior", [1, 3])
@torch.no_grad()
def test_cached_final_block_matches_teacher_forcing(batch, states, heads, prior):
    layer = model(heads=heads, states=states)
    data = case(layer, batch=batch, prior=prior)
    expected = train(layer, data)[:, data["indices"]]
    actual, _ = cached(layer, data)
    assert_close(actual, expected)


@pytest.mark.parametrize("states", [1, 2])
@pytest.mark.parametrize("prior", [1, 3])
@torch.no_grad()
def test_state_does_not_attend_to_video_or_actions(states, prior):
    layer = model(states=states)
    data = case(layer, prior=prior)
    before, _ = cached(layer, data)
    live = data["x"][:, data["indices"]].clone()
    live[:, :-states] += 4
    prefix = data["x"][:, :prior] - 3
    after, _ = cached(layer, data, live=live, prefix=prefix)
    assert_close(before[:, -states:], after[:, -states:])


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("cache_frames", [0, 3])
@torch.no_grad()
def test_image_only_path_matches_dense_attention(batch, cache_frames):
    layer = model()
    generator = torch.Generator().manual_seed(80)
    x = torch.randn(batch, 1, 8, generator=generator)
    cache = torch.randn(2, batch, cache_frames, 1, 8, generator=generator)
    before = cache.clone()
    frequencies = torch.ones(1, 1, 4, dtype=torch.complex128)
    actual, updated = layer(
        x,
        freqs=frequencies,
        freqs_action=frequencies[:, 0],
        freqs_state=frequencies[:, 0],
        action_register_length=None,
        kv_cache=cache,
        current_start_frame=cache_frames,
    )
    # Identity rotary phases leave these projected keys and queries unchanged.
    query = layer.norm_q(layer.q(x)).view(batch, 1, 1, 8)
    key = layer.norm_k(layer.k(x)).view(batch, 1, 1, 8)
    value = layer.v(x).view(batch, 1, 1, 8)
    key = torch.cat([cache[0], key], dim=1)
    value = torch.cat([cache[1], value], dim=1)
    expected = layer.o(layer.attn(query, key, value).flatten(2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(updated, torch.stack([key, value]), rtol=0, atol=0)
    torch.testing.assert_close(cache, before, rtol=0, atol=0)


@torch.no_grad()
def test_state_mismatch_reaches_actions_in_a_second_attention_layer():
    layer = model(actions=1, dyadic=True)
    data = case(layer, dyadic=True)
    expected_first = train(layer, data)
    actual_first, _ = cached(layer, data)
    next_data = dict(data, x=expected_first)
    expected = train(layer, next_data)[:, data["indices"]]
    actual, _ = cached(layer, next_data, live=actual_first)
    # The original context mismatch gives action 1.125 instead of 2.25 here.
    torch.testing.assert_close(expected[:, 1], torch.full((1, 8), 2.25), rtol=0, atol=0)
    assert_close(actual, expected)
