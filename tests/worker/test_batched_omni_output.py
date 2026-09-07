# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration
from vllm_omni.worker.gpu_ar_model_runner import _snapshot_tensor_payload_to_cpu_async

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
@pytest.mark.parametrize("kinds", [("emit", "stop", "prefill"), ("prefill", "emit"), ("stop",), ("prefill",)])
def test_local_stop_flags_and_output_rows_stay_request_aligned(kinds):
    model = object.__new__(MossTTSLocalTalkerForGeneration)
    torch.nn.Module.__init__(model)
    model.n_vq = 12
    model.audio_pad_token_id = 1024
    infos = []
    originals = []
    for index, kind in enumerate(kinds):
        codes = torch.full((1, 12), 1024 if kind == "stop" else index, dtype=torch.long)
        originals.append(codes)
        infos.append({} if kind == "prefill" else {"audio_codes": {"current": codes}})
    result = model.make_omni_output(torch.ones((len(kinds), 4)), runtime_additional_information=infos)
    if all(kind == "prefill" for kind in kinds):
        assert model._batch_should_continue is None
        assert result.multimodal_outputs == {}
        return
    assert model._batch_should_continue.tolist() == [kind != "stop" for kind in kinds]
    rows = result.multimodal_outputs["codes"]["audio"]
    assert len(rows) == len(kinds)
    for index, kind in enumerate(kinds):
        if kind == "prefill":
            assert rows[index].shape == (0, 12)
        else:
            torch.testing.assert_close(rows[index], originals[index], rtol=0, atol=0)
    assert all("current" not in info.get("audio_codes", {}) for info in infos)


@pytest.mark.cuda
@pytest.mark.parametrize("as_tuple", [False, True])
def test_async_packed_snapshot_survives_source_reuse(as_tuple):
    values = [
        torch.arange(12, device="cuda").view(1, 12),
        torch.empty((0, 12), device="cuda", dtype=torch.long),
        torch.arange(24, device="cuda").view(2, 12),
    ]
    expected = [v.cpu().clone() for v in values]
    payload = tuple(values) if as_tuple else values
    snapshot = _snapshot_tensor_payload_to_cpu_async(
        {"codes": payload, "metadata": "kept"}, copy_stream=torch.cuda.Stream(), pin_memory=True
    )
    for v in values:
        v.fill_(-99)
    snapshot.wait()
    actual = snapshot.payload["codes"]
    assert isinstance(actual, tuple if as_tuple else list)
    assert snapshot.payload["metadata"] == "kept"
    for out, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    # A second output must not alias the first output's host snapshot either.
    later = _snapshot_tensor_payload_to_cpu_async(values, copy_stream=torch.cuda.Stream(), pin_memory=True)
    later.wait()
    for out, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("offsets", [[0, 1], [1, 4]])
def test_owned_code_batch_and_embeddings_survive_next_graph_output(monkeypatch, offsets):
    import vllm_omni.worker.gpu_model_runner as mod
    from tests.worker.test_omni_gpu_model_runner import _make_runner, _noop_forward_context

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)
    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)
    runner.model.gpu_resident_buffer_keys = {("codes", "audio")}
    original = runner.talker_mtp
    returned = []

    def mtp(*args, **kwargs):
        result = original(*args, **kwargs)
        returned.append(result)
        return result

    runner.talker_mtp = mtp
    embeddings = torch.full((6, 4), -10.0)
    runner._talker_mtp_forward(["r1", "r2"], embeddings, offsets)
    assert torch.equal(embeddings[offsets], torch.ones(2, 4))
    for i in set(range(6)) - set(offsets):
        assert torch.equal(embeddings[i], torch.full((4,), -10.0))
    returned[0][1].fill_(999)
    assert runner.model_intermediate_buffer["r1"]["codes"]["audio"].item() == 0
    assert runner.model_intermediate_buffer["r2"]["codes"]["audio"].item() == 1


@pytest.mark.cuda
def test_mtp_mixed_offsets_do_not_synchronize_cuda(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod
    from tests.worker.test_omni_gpu_model_runner import _make_runner, _noop_forward_context

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)
    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)
    runner.model.gpu_resident_buffer_keys = {("codes", "audio")}
    for name in ("talker_mtp_input_ids", "talker_mtp_inputs_embeds", "last_talker_hidden", "text_step"):
        buffer = getattr(runner, name)
        buffer.gpu = buffer.gpu.to("cuda")
    embeds = torch.ones((2, 4), device="cuda")
    codes = torch.arange(2, device="cuda").view(2, 1)
    runner.talker_mtp = lambda *args, **kwargs: (embeds, codes)
    output = torch.full((5, 4), -10.0, device="cuda")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        runner._talker_mtp_forward(["r1", "r2"], output, [1, 4])
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    expected = torch.full((5, 4), -10.0)
    expected[[1, 4]] = 1.0
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    codes.fill_(999)
    assert runner.model_intermediate_buffer["r1"]["codes"]["audio"].item() == 0
    assert runner.model_intermediate_buffer["r2"]["codes"]["audio"].item() == 1


@pytest.mark.cuda
def test_mixed_stop_flags_do_not_synchronize_cuda():
    model = object.__new__(MossTTSLocalTalkerForGeneration)
    torch.nn.Module.__init__(model)
    model.n_vq = 12
    model.audio_pad_token_id = 1024
    infos = [
        {"audio_codes": {"current": torch.zeros((1, 12), device="cuda", dtype=torch.long)}},
        {},
        {"audio_codes": {"current": torch.full((1, 12), 1024, device="cuda", dtype=torch.long)}},
    ]
    hidden = torch.ones((3, 4), device="cuda")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        output = model.make_omni_output(hidden, runtime_additional_information=infos)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert model._batch_should_continue.tolist() == [True, True, False]
    assert [row.shape for row in output.multimodal_outputs["codes"]["audio"]] == [(1, 12), (0, 12), (1, 12)]
