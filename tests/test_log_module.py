"""Regression coverage for independent per-variable log buffers."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from hydroforge.data.input import InputProxy
from hydroforge.kernels.registry import KERNEL_BACKEND

from cmfgpu.models import CaMaFlood


_PARAMETERS = Path(__file__).parents[1] / "inp/glb_06min_japan/parameters.nc"


def _device() -> torch.device:
    if KERNEL_BACKEND in {"cuda", "triton"} and torch.cuda.is_available():
        return torch.device("cuda")
    if KERNEL_BACKEND == "metal" and torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("log kernel regression requires an accelerator backend")


@pytest.mark.skipif(not _PARAMETERS.exists(), reason="Japan parameters unavailable")
def test_log_variables_have_independent_buffers_and_write_values(tmp_path) -> None:
    proxy = InputProxy.from_nc(_PARAMETERS)
    catchment_id = np.asarray(proxy.data["catchment_id"])
    proxy.data["output_catchment_id"] = catchment_id[[0]]
    proxy.visible_vars.add("output_catchment_id")

    model = CaMaFlood(
        rank=0,
        world_size=1,
        device=_device(),
        experiment_name="independent_log_buffers",
        input_proxy=proxy,
        output_dir=tmp_path,
        opened_modules=["base", "log"],
        variables_to_save={},
        output_workers=0,
        BLOCK_SIZE=128,
        mixed_precision=False,
    )
    try:
        buffers = [getattr(model.log, name) for name in model.log.log_vars]
        assert len(buffers) == 11
        assert len({buffer.data_ptr() for buffer in buffers}) == 11
        assert not hasattr(model.log, "log_sums")
        runoff = torch.ones(model.base.num_catchments, device=model.device)
        model.step_advance(runoff, 3600.0, 1, datetime(2000, 1, 1))
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elif model.device.type == "mps":
            torch.mps.synchronize()

        lines = model.log_path.read_text().splitlines()
        assert len(lines) == 3
        values = lines[-1].split()[2:]
        assert len(values) == 11
        assert all(np.isfinite(float(value)) for value in values)
    finally:
        model.close()
