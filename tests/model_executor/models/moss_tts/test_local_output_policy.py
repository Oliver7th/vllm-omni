# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.model_executor.models.moss_tts.pipeline import MOSS_TTS_LOCAL_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("use_defaults", [False, True])
def test_local_talker_control_outputs_preserve_codec_streaming(mocker, use_defaults):
    base = object.__new__(OmniBase)
    base.engine = mocker.Mock(num_stages=2)
    base.sampling_constraints_list = [dict(stage.sampling_constraints) for stage in MOSS_TTS_LOCAL_PIPELINE.stages]
    base.default_sampling_params_list = [
        base._apply_sampling_constraints(SamplingParams(max_tokens=8, seed=11), constraints)
        for constraints in base.sampling_constraints_list
    ]
    caller = coerce_param_message_types(
        [SamplingParams(max_tokens=8, seed=11), SamplingParams(max_tokens=8, seed=11)], True
    )
    result = base.resolve_sampling_params_list(None if use_defaults else caller, allow_delta_coercion=True)

    assert result[0].output_kind == RequestOutputKind.FINAL_ONLY
    assert result[0].detokenize is False
    assert result[0].max_tokens == 8
    assert result[0].seed == 11
    assert result[1].output_kind == RequestOutputKind.DELTA
    assert caller[0].output_kind == RequestOutputKind.DELTA
    assert base.default_sampling_params_list[0].output_kind == RequestOutputKind.FINAL_ONLY
    assert MOSS_TTS_LOCAL_PIPELINE.stages[0].final_output is False
    assert MOSS_TTS_LOCAL_PIPELINE.stages[1].final_output_type == "audio"
