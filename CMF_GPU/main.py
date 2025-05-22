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
from CMF_GPU.utils.Preprocesser import check_input_h5, load_input_h5, make_order
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime, timedelta
from omegaconf import OmegaConf

def setup_distributed():
    torch.multiprocessing.set_start_method("spawn", force=True)
    dist.init_process_group(backend="nccl", init_method="env://")     
    local_rank = int(os.environ["LOCAL_RANK"])      
    rank        = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.set_default_device(rank)
    return local_rank, rank, world_size

def broadcast_orders(inp_dir, simulation_config, rank, world_size):
    """Compute on rank 0, send to others."""
    statistics = simulation_config.get("statistics", [])
    modules = simulation_config["modules"]
    precision = simulation_config["precision"]
    if rank == 0:
        generate_triton_aggregator_script(inp_dir / "Aggregate.py", statistics)
        check_input_h5(inp_dir / "parameters.h5", inp_dir / "init_states.h5", modules, precision)
        orders = make_order(inp_dir / "parameters.h5",
                            ["base","adaptive_time_step","log"],
                            statistics, world_size)
    else:
        orders = None
    obj_list = [orders]
    dist.broadcast_object_list(obj_list, src=0)     # ③ broadcast
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
    orders = broadcast_orders(Path(simulation_config["inp_dir"]), simulation_config, rank, world_size)

    params, states = load_input_h5(inp_dir / "parameters.h5", inp_dir / "init_states.h5", orders, rank)
    runoff_input_matrix = split_runoff_input_matrix(inp_dir / "runoff_input_matrix.npz", orders, rank)
    ns = runpy.run_path(inp_dir / "Aggregate.py")
    agg_fn = ns["update_statistics"]
    start_date = datetime.strptime(simulation_config["start_date"], "%Y-%m-%d")

    ds = DailyBinDataset(**runoff_config["params"])
    loader = DataLoader(
        dataset=ds,
        batch_size=1,
        num_workers=1,
    )
    dumper = DataDumper(
        base_dir=out_dir,
        statistics=statistics,
        file_format="nc",
        num_workers=1,
    )
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    logger = Logger(base_dir=out_dir,
        disabled=False if "log" in simulation_config["modules"] else True
    )
    # TODO: multi-GPU support
    default_num_sub_steps = simulation_config["default_num_sub_steps"]
    current_time = start_date
    for (runoff_data, runoff_time_step) in loader:
        # TODO: tranfer runoff_t to GPU here
        time_step = simulation_config["time_step"]
        runoff_data = runoff_input_matrix @ runoff_data[0].to(f"cuda:{local_rank}")
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
    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)