# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import time
from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from hydroforge.data.datasets import DailyBinDataset
from hydroforge.data.distributed import setup_distributed
from hydroforge.data.input import InputProxy
from torch.utils.data import DataLoader

from cmfgpu.models import CaMaFlood

BLOCK_SIZE_LIST = [64, 128, 256, 512, 1024]

def benchmark_block_sizes():
    ### Benchmark Configuration ###
    experiment_name = "benchmark_run"
    resolution = "glb_15min"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"
    output_dir = "/home/eat/CaMa-Flood-GPU/out"
    opened_modules = ["base", "adaptive_time", "bifurcation"]
    num_sub_steps = 360 if "adaptive_time" not in opened_modules else None
    variables_to_save = {}
    runoff_time_interval = timedelta(days=1)
    loader_workers = 3
    prefetch_factor = 2
    save_state = False
    output_split_by_year = False

    runoff_dir = "/home/eat/cmf_v420_pkg/inp/test_1deg/runoff"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_bin.npz"
    runoff_shape = [180, 360]
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2000, 4, 1)
    unit_factor = 86400000
    bin_dtype = "float32"
    prefix = "Roff____"
    suffix = ".one"

    local_rank, rank, world_size = setup_distributed()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    input_proxy = InputProxy.from_nc(input_file)

    dataset = DailyBinDataset(
        base_dir=runoff_dir,
        shape=runoff_shape,
        start_date=start_date,
        end_date=end_date,
        model_step=runoff_time_interval,
        unit_factor=unit_factor,
        bin_dtype=bin_dtype,
        prefix=prefix,
        suffix=suffix,
    )
    schedule = dataset.simulation_schedule
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if loader_workers > 0 else None,
    )

    results = []
    if rank == 0:
        print("Benchmarking BLOCK_SIZE...")

    for block_size in BLOCK_SIZE_LIST:
        # BLOCK_SIZE is part of the compiled model specialization.  A fresh
        # model also guarantees identical hydrological initial state for every
        # benchmark candidate.
        model = CaMaFlood(
            device=device,
            experiment_name=f"{experiment_name}_bs{block_size}",
            input_proxy=input_proxy,
            output_dir=output_dir,
            opened_modules=opened_modules,
            variables_to_save=variables_to_save,
            output_workers=0,
            output_netcdf_options={},
            BLOCK_SIZE=block_size,
            output_split_by_year=output_split_by_year,
            simulation_schedule=schedule,
        )
        local_mapping = dataset.build_local_mapping(
            mapping_file=runoff_mapping_file,
            desired_catchment_ids=model.base.catchment_id.to("cpu").numpy(),
            device=device,
        )
        if rank == 0:
            print(f"Benchmarking BLOCK_SIZE={block_size}...")
        if device.type == "cuda":
            stream_ctx = torch.cuda.stream(torch.cuda.Stream(device=device))
            torch.cuda.synchronize()
        else:
            stream_ctx = nullcontext()
        start = time.time()

        for runoff_chunk in loader:
            with stream_ctx:
                runoff_chunk = dataset.shard_forcing(
                    runoff_chunk.to(device),
                    local_mapping,
                )
                for runoff in runoff_chunk:
                    model.set_inputs(runoff)
                    model.step_advance(
                        num_sub_steps=num_sub_steps,
                    )

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        elapsed_ms = (end - start) * 1000
        results.append((block_size, elapsed_ms))
        if save_state and block_size == BLOCK_SIZE_LIST[-1]:
            model.save_state()
        model.close()
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        print("\n=== Benchmark Results ===")
        for bs, t in results:
            print(f"BLOCK_SIZE={bs} --> {t:.2f} ms")

if __name__ == "__main__":
    benchmark_block_sizes()
