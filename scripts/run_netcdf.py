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
    resolution = "glb_15min"
    # Windows path example: input_file = fr"C:\Users\YourName\CaMa-Flood-GPU\inp\{resolution}\parameters.nc"
    experiment_name = f"{resolution}_nc"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ("base", "adaptive_time", "bifurcation")
    num_sub_steps = 360 if "adaptive_time" not in opened_modules else None
    variables_to_save = {
        "mean": ["total_outflow"],
        "last": ["river_depth"],
    }
    loader_workers = 2
    output_workers = 2
    # Runoff depth flux; the mapping weights (catchment areas, m2) turn it
    # into the model's runoff in m3 s-1. Source units come from the files'
    # units attribute (set source_units, e.g. "mm day-1", if they have none).
    source_units = None
    target_units = "m s-1"
    prefetch_factor = 2
    BLOCK_SIZE = 128
    save_state = True

    start_date = datetime(2000, 1, 1)
    end_date = datetime(2000, 12, 31)
    runoff_dir = "/home/eat/E2O_ecmwf"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_nc.npz"
    runoff_time_interval = timedelta(days=1)
    prefix = "e2o_ecmwf_wrr2_glob15_day_Runoff_"
    suffix = ".nc"
    var_name = "Runoff"
    output_split_by_year = False

    # Set cycles to 0 to disable spin-up.
    spin_up_start_date = datetime(2000, 1, 1)
    spin_up_end_date = datetime(2000, 12, 31)
    spin_up_cycles = 0
    ### Configuration End ###

    distributed = setup_distributed(
        allowed_devices=("cuda", "mps"),
    )
    world_size = distributed.world_size
    device = distributed.device

    input_proxy = InputProxy.from_nc(input_file)

    dataset = NetCDFDataset(
        base_dir=runoff_dir,
        start_date=start_date,
        end_date=end_date,
        time_interval=runoff_time_interval,
        spin_up_cycles=spin_up_cycles,
        spin_up_start_date=spin_up_start_date if spin_up_cycles > 0 else None,
        spin_up_end_date=spin_up_end_date if spin_up_cycles > 0 else None,
        model_step=runoff_time_interval,
        source_units=source_units,
        target_units=target_units,
        var_name=var_name,
        prefix=prefix,
        suffix=suffix,
        clip_negative=True,
    )
    schedule = dataset.simulation_schedule
    # DataLoader returns source rows; it does not repeat them for short model steps.
    reuse_count = dataset.time_interval // dataset.model_step

    model = CaMaFlood(
        device=device,
        output=OutputConfig(
            experiment=experiment_name,
            dir=output_dir,
            variables=variables_to_save,
            workers=output_workers,
            split_by_year=output_split_by_year,
        ),
        block_size=BLOCK_SIZE,
        input_proxy=input_proxy,
        opened_modules=opened_modules,
        simulation_schedule=schedule,
    )
    model.materialize()
    dataset, local_mapping = dataset.build_local_mapping(
        runoff_mapping_file,
        model.base.catchment_id.to("cpu").numpy(),
        device=device,
    )

    loader = DataLoader(
        dataset,
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
    for runoff_chunk in loader:
        with stream_ctx:
            runoff_chunk = dataset.shard_forcing(
                runoff_chunk.to(
                    device,
                    non_blocking=device.type == "cuda",
                ),
                local_mapping,
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
