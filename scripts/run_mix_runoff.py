# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from hydroforge.data import InputProxy
from hydroforge.parallel import setup_distributed
from hydroforge.data.datasets import NetCDFDataset
from hydroforge.model import OutputConfig
from torch.utils.data import DataLoader

from cmfgpu.models import CaMaFlood


def main() -> None:

    ### Configuration Start ###
    resolution = "jpn_03min"
    # Windows path example: runoff_dir = r"C:\Users\YourName\cmf_v420_pkg\map\jpn_runoff"
    experiment_name = f"{resolution}_nc"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ("base", "adaptive_time", "bifurcation")
    num_sub_steps = 360 if "adaptive_time" not in opened_modules else None
    loader_workers = 3
    output_workers = 2
    unit_factor = 86400000
    prefetch_factor = 2
    BLOCK_SIZE = 128
    save_state = False

    # Spin-up configuration
    spin_up_start_date = datetime(1950, 1, 1)
    spin_up_end_date = datetime(1950, 12, 31)
    spin_up_cycles = 1

    start_date = datetime(1950, 1, 1)
    end_date = datetime(1950, 12, 31)
    runoff_dir = "/home/eat/cmf_v420_pkg/map/jpn_runoff"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_nc.npz"
    runoff_time_interval = timedelta(days=1)
    prefix0 = "baseflow_"
    prefix1 = "runoff_"
    suffix = ".nc"
    var_name0 = "baseflow"
    var_name1 = "runoff"
    output_split_by_year = False
    ### Configuration End ###

    distributed = setup_distributed(
        allowed_devices=("cuda", "mps"),
    )
    world_size = distributed.world_size
    device = distributed.device

    input_proxy = InputProxy.from_nc(input_file)

    dataset_time = dict(
        start_date=start_date,
        end_date=end_date,
        time_interval=runoff_time_interval,
        spin_up_cycles=spin_up_cycles,
        spin_up_start_date=spin_up_start_date if spin_up_cycles > 0 else None,
        spin_up_end_date=spin_up_end_date if spin_up_cycles > 0 else None,
    )
    dataset0 = NetCDFDataset(
        **dataset_time,
        base_dir=runoff_dir,
        model_step=runoff_time_interval,
        unit_factor=unit_factor,
        var_name=var_name0,
        prefix=prefix0,
        suffix=suffix,
        clip_negative=True,
    )
    dataset1 = NetCDFDataset(
        **dataset_time,
        base_dir=runoff_dir,
        model_step=runoff_time_interval,
        unit_factor=unit_factor,
        var_name=var_name1,
        prefix=prefix1,
        suffix=suffix,
        clip_negative=True,
    )
    if dataset0.simulation_schedule != dataset1.simulation_schedule:
        raise ValueError("forcing datasets generated different schedules")
    schedule = dataset0.simulation_schedule
    # DataLoader returns source rows; it does not repeat them for short model steps.
    reuse_count = dataset0.time_interval // dataset0.model_step

    model = CaMaFlood(
        device=device,
        output=OutputConfig(
            experiment=experiment_name,
            dir=output_dir,
            workers=output_workers,
            split_by_year=output_split_by_year,
            variables={
                "mean": ["total_outflow"],
                "last": ["river_depth"],
            },
        ),
        block_size=BLOCK_SIZE,
        input_proxy=input_proxy,
        opened_modules=opened_modules,
        simulation_schedule=schedule,
    )
    model.materialize()

    desired_catchment_ids = model.base.catchment_id.to("cpu").numpy()
    dataset0, local_mapping0 = dataset0.build_local_mapping(
        runoff_mapping_file, desired_catchment_ids, device=device
    )
    dataset1, _ = dataset1.build_local_mapping(
        runoff_mapping_file, desired_catchment_ids, device=device
    )
    loader0 = DataLoader(
        dataset0,
        batch_size=None,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=prefetch_factor if loader_workers > 0 else None,
    )
    loader1 = DataLoader(
        dataset1,
        batch_size=None,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=prefetch_factor if loader_workers > 0 else None,
    )

    stream_ctx = (
        torch.cuda.stream(torch.cuda.Stream(device=device))
        if device.type == "cuda"
        else nullcontext()
    )
    for runoff_chunk0, runoff_chunk1 in zip(loader0, loader1, strict=True):
        with stream_ctx:
            runoff_chunk = dataset0.shard_forcing(
                runoff_chunk0.to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                + runoff_chunk1.to(
                    device,
                    non_blocking=device.type == "cuda",
                ),
                local_mapping0,
            )
            for runoff in runoff_chunk:
                model.set_inputs(runoff=runoff)
                for _ in range(reuse_count):
                    model.step_advance(
                        num_sub_steps=num_sub_steps,
                    )
    if save_state:
        model.save_state()
    model.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
