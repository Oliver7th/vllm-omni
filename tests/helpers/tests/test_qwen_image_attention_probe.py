# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_flash3_probe_matches_diffusers_kernel_version(monkeypatch):
    import kernels
    from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY, AttentionBackendName

    from tests.e2e.accuracy import test_qwen_image as accuracy

    config = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
    calls = []

    def get_kernel(repo_id, *, version):
        calls.append((repo_id, version))
        return object()

    monkeypatch.setattr(kernels, "get_kernel", get_kernel)
    monkeypatch.setattr(accuracy, "_FLASH_ATTN3_HUB_AVAILABLE", None)

    assert accuracy._flash_attn3_hub_available()
    assert accuracy._omni_server_env() is None
    assert calls == [(config.repo_id, config.version)]


def test_unavailable_flash3_keeps_both_sides_on_sdpa(monkeypatch):
    import kernels

    from tests.e2e.accuracy import test_qwen_image as accuracy

    loader = Mock(side_effect=RuntimeError("No compatible build variant"))
    monkeypatch.setattr(kernels, "get_kernel", loader)
    monkeypatch.setattr(accuracy, "_FLASH_ATTN3_HUB_AVAILABLE", None)
    reference = Mock()

    assert not accuracy._flash_attn3_hub_available()
    assert accuracy._omni_server_env() == {"DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA"}
    accuracy._set_reference_attention_backend(reference)
    reference.transformer.set_attention_backend.assert_called_once_with("native")
    loader.assert_called_once()
