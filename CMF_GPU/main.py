import os
import shutil
import runpy
from CMF_GPU.phys.triton.StepAdvance import advance_step
from CMF_GPU.utils.Checker import prepare_model_and_function
from CMF_GPU.utils.Dataloader import DataLoader
from CMF_GPU.utils.Dataset import DailyBinDataset
from CMF_GPU.utils.Datadumper import DataDumper
from CMF_GPU.utils.Aggregator import generate_triton_aggregator_script
from CMF_GPU.utils.Logger import Logger
from CMF_GPU.utils.utils import snapshot_to_npz, gather_device_dicts, load_from_npz
from scipy.sparse import load_npz
from datetime import datetime, timedelta
from omegaconf import OmegaConf


def main(config_file):

    config = OmegaConf.load(config_file)
    # config = OmegaConf.to_container(config, resolve=True)
    runoff_config = config["runoff_config"]
    simulation_config = config["simulation_config"]
    params = load_from_npz(os.path.join(simulation_config["inp_dir"], "parameters.npz"), "param", simulation_config["modules"], omit_hidden=True)
    states = load_from_npz(os.path.join(simulation_config["inp_dir"], "init_states.npz"), "state", simulation_config["modules"], omit_hidden=True)
    generate_triton_aggregator_script(os.path.join(simulation_config["inp_dir"], "Aggregate.py"), simulation_config["statistics"])
    ns = runpy.run_path(os.path.join(simulation_config["inp_dir"], "Aggregate.py"))
    agg_fn = ns["update_statistics"]
    params, states, parallel_step_fn = prepare_model_and_function(
        params,
        states,
        advance_step,
        simulation_config,
        simulation_config,
    )
    params["runoff_input_matrix"] = load_npz(os.path.join(simulation_config["inp_dir"], "runoff_input_matrix.pkl"))
    start_date = datetime.strptime(simulation_config["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(simulation_config["end_date"], "%Y-%m-%d")
    time_starts = [start_date + timedelta(seconds=simulation_config["time_step"]) * i for i in range((end_date - start_date).days + 1)]

    ds = DailyBinDataset(**runoff_config["params"])

    loader = DataLoader(
        time_starts=time_starts,
        dataset=ds,
        unit_factor=simulation_config["unit_factor"],
        num_workers=3,
        max_cache_steps=100,
        precision="float32",
        runoff_mask=ds.get_mask(),
    )
    if os.path.exists(simulation_config["out_dir"]):
        shutil.rmtree(simulation_config["out_dir"])
    os.makedirs(simulation_config["out_dir"])
    dumper = DataDumper(
        base_dir=simulation_config["out_dir"],
        statistics=simulation_config["statistics"],
        file_format="nc",
        num_workers=3,
    )
    logger = Logger(base_dir=simulation_config["out_dir"],
        buffer_size=params[0].get("log_buffer_size", None), 
        disabled=False if "log" in simulation_config["modules"] else True
    )
    # TODO: multi-GPU support
    default_num_sub_steps = simulation_config["default_num_sub_steps"]
    for time_start in time_starts:
        runoff_t, time_length = loader.get_data(time_start)
        # TODO: tranfer runoff_t to GPU here
        current_time = time_start
        time_step = simulation_config["time_step"]
        if time_length % time_step != 0:
            raise ValueError(
                f"Runoff step length {time_length} cannot be evenly divided by time_step={time_step}, please check your data or configuration."
            )
        iters_per_runoff_step = int(time_length // time_step)
        dT_def = time_step / default_num_sub_steps
        # print("total_storage:", f"{states["total_storage"].sum():.4e}")
        for _ in range(iters_per_runoff_step):
            logger.set_current_time(current_time)
            parallel_step_fn(
                simulation_config, params, states, runoff_t, dT_def, logger, agg_fn
            )
            aggregator_cpu = dumper.gather_stats(states)
            dumper.submit_data(current_time, aggregator_cpu)
            current_time += timedelta(seconds=time_step)

    states_cpu = gather_device_dicts(states)
    snapshot_to_npz(states_cpu, "state", simulation_config["modules"], os.path.join(simulation_config["out_dir"], "final_states.npz"))
    dumper.close()
    logger.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CMF_GPU simulation.")
    parser.add_argument("config_file", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()
    main(args.config_file)