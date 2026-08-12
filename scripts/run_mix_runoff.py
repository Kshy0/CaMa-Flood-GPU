# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from hydroforge.data.datasets import NetCDFDataset
from hydroforge.data.distributed import setup_distributed
from hydroforge.data.input import InputProxy
from torch.utils.data import DataLoader
from hydroforge.contracts.temporal import (
    EveryStep,
    StatisticsPlan,
)

from cmfgpu.models import CaMaFlood


def main():
    ### Configuration Start ###
    resolution = "jpn_03min"
    experiment_name = f"{resolution}_nc"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ["base", "adaptive_time", "bifurcation"]
    num_sub_steps = 360 if "adaptive_time" not in opened_modules else None
    variables_to_save = {"mean": ["total_outflow"], "last": ["river_depth"]}
    runoff_chunk_len = 48
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

    local_rank, _, world_size = setup_distributed()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    input_proxy = InputProxy.from_nc(input_file)

    dataset0 = NetCDFDataset(
        base_dir=runoff_dir,
        start_date=start_date,
        end_date=end_date,
        model_step=runoff_time_interval,
        unit_factor=unit_factor,
        var_name=var_name0,
        chunk_len=runoff_chunk_len,
        time_interval=runoff_time_interval,
        prefix=prefix0,
        suffix=suffix,
        spin_up_cycles=spin_up_cycles,
        spin_up_start_date=spin_up_start_date,
        spin_up_end_date=spin_up_end_date,
        clip_negative=True,
    )
    dataset1 = NetCDFDataset(
        base_dir=runoff_dir,
        start_date=start_date,
        end_date=end_date,
        model_step=runoff_time_interval,
        unit_factor=unit_factor,
        var_name=var_name1,
        chunk_len=runoff_chunk_len,
        time_interval=runoff_time_interval,
        prefix=prefix1,
        suffix=suffix,
        spin_up_cycles=spin_up_cycles,
        spin_up_start_date=spin_up_start_date,
        spin_up_end_date=spin_up_end_date,
        clip_negative=True,
    )
    if dataset0.simulation_schedule != dataset1.simulation_schedule:
        raise ValueError("forcing datasets generated different schedules")
    schedule = dataset0.simulation_schedule

    model = CaMaFlood(
        device=device,
        experiment_name=experiment_name,
        input_proxy=input_proxy,
        output_dir=output_dir,
        opened_modules=opened_modules,
        variables_to_save=variables_to_save,
        output_workers=output_workers,
        output_netcdf_options={"compression": "zlib", "complevel": 4},
        BLOCK_SIZE=BLOCK_SIZE,
        output_split_by_year=output_split_by_year,
        simulation_schedule=schedule,
        statistics_plan=StatisticsPlan(inner=EveryStep()),
    )

    desired_catchment_ids = model.base.catchment_id.to("cpu").numpy()
    local_mapping0 = dataset0.build_local_mapping(
        mapping_file=runoff_mapping_file,
        desired_catchment_ids=desired_catchment_ids,
        device=device,
    )
    dataset1.build_local_mapping(
        mapping_file=runoff_mapping_file,
        desired_catchment_ids=desired_catchment_ids,
        device=device,
    )
    loader0 = DataLoader(
        dataset0,
        batch_size=None,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if loader_workers > 0 else None,
    )
    loader1 = DataLoader(
        dataset1,
        batch_size=None,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if loader_workers > 0 else None,
    )

    stream_ctx = torch.cuda.stream(torch.cuda.Stream(device=device)) if device.type == "cuda" else nullcontext()
    for runoff_chunk0, runoff_chunk1 in zip(loader0, loader1, strict=True):
        with stream_ctx:
            runoff_chunk = dataset0.shard_forcing(
                runoff_chunk0.to(device) + runoff_chunk1.to(device),
                local_mapping0,
            )
            for runoff in runoff_chunk:
                model.set_inputs(runoff)
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
