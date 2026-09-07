# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import _MossCodecStreamSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CausalCodec(nn.Module):
    downsample_rate = 1

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def initialize_decoder_state_pool(self, capacity, scratch):
        self.state = torch.zeros(capacity + scratch)

    def reset_decoder_state_slots(self, slots):
        self.state[slots] = 0

    def decode_streaming_batch(self, codes, lengths, slots, valid_rows):
        audio = codes.sum(0).float().cumsum(-1) + self.state[slots, None]
        self.state[slots] = audio[:, -1]
        return SimpleNamespace(audio=audio[:, None], audio_lengths=lengths)


def session():
    return _MossCodecStreamSession(CausalCodec(), state_capacity=3, n_vq=2, vllm_config=None)


@pytest.mark.parametrize("tail_length", [1, 3, 7, 14])
def test_terminal_padding_preserves_audio_live_state_and_reused_slots(tail_length):
    batched, separate = session(), session()
    for s in (batched, separate):
        assert [s.acquire(), s.acquire()] == [0, 1]
        s.step({0: torch.ones(2, 15), 1: torch.full((2, 15), 3)})
    short = torch.arange(2 * tail_length).reshape(2, tail_length)
    long = torch.full((2, 15), 5)
    expected_short = separate.step({0: short}, terminal_slots={0})[0]
    expected_long = separate.step({1: long})[1]
    actual = batched.step({0: short, 1: long}, terminal_slots={0})
    torch.testing.assert_close(actual[0], expected_short, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected_long, rtol=0, atol=0)
    for s in (batched, separate):
        s.release(0, state_already_reset=True)
        assert s.acquire() == 0
    next_step = {0: torch.ones(2, 1), 1: torch.ones(2, 1)}
    a, b = batched.step(next_step), separate.step(next_step)
    for slot in (0, 1):
        torch.testing.assert_close(a[slot], b[slot], rtol=0, atol=0)


def test_nonterminal_row_cannot_silently_advance_through_padding():
    s = session()
    s.acquire()
    s.acquire()
    with pytest.raises(ValueError, match="Only terminal"):
        s.step({0: torch.ones(2, 3), 1: torch.ones(2, 15)})
    assert s._codec.state.count_nonzero() == 0


@pytest.mark.parametrize("graph_capacity, expected_calls", [(None, 5), (4, 5), (8, 3)])
def test_only_existing_compatible_graph_buckets_are_coalesced(graph_capacity, expected_calls):
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder

    calls = []
    leased: list[int] = []

    def acquire():
        slot = len(leased)
        leased.append(slot)
        return slot

    def step(plan, *, terminal_slots):
        calls.append((dict(plan), set(terminal_slots)))
        return {slot: torch.zeros(1, codes.shape[1]) for slot, codes in plan.items()}

    graph = (
        None
        if graph_capacity is None
        else SimpleNamespace(
            batch_sizes=[1, graph_capacity],
            frame_sizes=[1, 8, 15],
            _select_frame_size=lambda size, allow_padding: next((b for b in [1, 8, 15] if b >= size), None),
        )
    )
    fake_session = SimpleNamespace(
        _cudagraph_wrapper=graph,
        acquire=acquire,
        step=step,
        release=lambda slot, **kwargs: None,
    )
    decoder = object.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(decoder)
    decoder._stream_max_step_frames = 15
    decoder._stream_state_capacity = 8
    decoder._stream_req_slots = {}
    decoder._ensure_stream_session = lambda: fake_session
    lengths = [3, 7, 8, 1, 15]
    finished = [True, True, False, True, False]
    result = decoder._decode_streaming_batch(
        [
            (i, str(i), torch.ones(2, length), done)
            for i, (length, done) in enumerate(zip(lengths, finished, strict=True))
        ]
    )
    assert len(calls) == expected_calls
    assert [result[i].shape[-1] for i in range(5)] == lengths
    if graph_capacity == 8:
        assert set(calls[0][0]) == {0, 1, 2}
        assert calls[0][1] == {0, 1}
        assert set(calls[1][0]) == {3}  # Preserve the exact one-frame graph.
