import shutil
import runpy
import os
import torch
import torch.distributed as dist
from CMF_GPU.phys.triton.StepAdvance import advance_step
from CMF_GPU.utils.Dataset import DailyBinDataset
from CMF_GPU.utils.Aggregator import generate_triton_aggregator_script
from CMF_GPU.utils.Logger import Logger
from CMF_GPU.utils.Datadumper import DataDumper
from CMF_GPU.utils.Spliter import split_runoff_input_matrix
from CMF_GPU.utils.Preprocessor import check_input_h5, load_input_h5, make_order
from CMF_GPU.utils.utils import get_local_rank
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime, timedelta
from omegaconf import OmegaConf


def setup_distributed():
    torch.multiprocessing.set_start_method("spawn", force=True)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = get_local_rank()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        torch.set_default_device(local_rank)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        torch.cuda.set_device(local_rank)
        torch.set_default_device(local_rank)
    
    return local_rank, rank, world_size

def broadcast_orders(inp_dir, simulation_config, rank, world_size):
    """Compute on rank 0, send to others."""
    statistics = simulation_config.get("statistics", [])
    modules = simulation_config["modules"]
    if rank == 0:
        generate_triton_aggregator_script(inp_dir / "Aggregate.py", statistics)
        check_input_h5(inp_dir / "parameters.h5", inp_dir / "init_states.h5", modules)
        orders = make_order(inp_dir / "parameters.h5",
                            modules,
                            statistics, world_size)
    else:
        orders = None
    obj_list = [orders]
    if world_size > 1:
        dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def main(config):
    local_rank, rank, world_size = setup_distributed()
    
    # ---------- load config -----------
    cfg = OmegaConf.load(config)
    simulation_config      = cfg.simulation_config
    runoff_config   = cfg.runoff_config
    statistics   = simulation_config.get("statistics", [])
    inp_dir = Path(simulation_config["inp_dir"])
    out_dir = Path(simulation_config["out_dir"])
    orders = broadcast_orders(inp_dir, simulation_config, rank, world_size)
    params, states, dim_info = load_input_h5(inp_dir / "parameters.h5", inp_dir / "init_states.h5", orders, rank)
    if rank == 0:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    runoff_input_matrix = split_runoff_input_matrix(inp_dir / "runoff_input_matrix.npz", orders, rank)
    ns = runpy.run_path(inp_dir / "Aggregate.py")
    agg_fn = ns["update_statistics"]
    start_date = datetime.strptime(simulation_config["start_date"], "%Y-%m-%d")
    device = torch.device(f"cuda:{local_rank}")
    ds = DailyBinDataset(**runoff_config["params"])
    loader = DataLoader(
        dataset=ds,
        batch_size=32,
        num_workers=3,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
    )
    dumper = DataDumper(
        base_dir=out_dir,
        file_format="nc",
        statistics=statistics,
        dim_info=dim_info,
        num_workers=3,
    )

    logger = Logger(base_dir=out_dir,
        disabled=False if "log" in simulation_config["modules"] else True
    )
    default_num_sub_steps = simulation_config["default_num_sub_steps"]
    current_time = start_date
    for batch in loader:
        runoff_datas, runoff_time_steps = batch
        runoff_datas = (runoff_datas.to(device) @ runoff_input_matrix).contiguous()
        for i in range(runoff_datas.shape[0]):
            runoff_time_step = runoff_time_steps[i]
            runoff_data = runoff_datas[i]
            time_step = simulation_config["time_step"]
            if runoff_time_step % time_step != 0:
                raise ValueError(
                    f"Runoff step length {runoff_time_step} cannot be evenly divided by time_step={time_step}, please check your data or configuration."
                )

            iters_per_runoff_step = int(runoff_time_step // time_step)
            dT_def = time_step / default_num_sub_steps

            for _ in range(iters_per_runoff_step):
                logger.set_current_time(current_time)
                advance_step(
                    simulation_config, params, states, runoff_data, dT_def, logger, agg_fn
                )
                aggregator_cpu = dumper.gather_stats(states)
                dumper.submit_data(current_time, aggregator_cpu)
                current_time += timedelta(seconds=time_step)
    dumper.close()
    logger.close()
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)