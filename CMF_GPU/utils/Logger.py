import logging
from datetime import datetime, timedelta
from pathlib import Path

class Logger:
    def __init__(self, base_dir, disabled = False):
        self.disabled = disabled
        if self.disabled:
            return
        self.start_time = datetime.now()
        self.scale = 1e-9
        self.log_path = Path(base_dir) / "log_CaMa_GPU.txt"
        # Prepare logger
        self.logger = logging.getLogger("CaMaAligned")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_path, mode='a')
        # Formatter: all fields left-aligned ('<'), fixed widths
        fmt = (
            "{StepStartTime:<18}"
            "{StepSeconds:<14.6f}"
            "{RiverStorage:<14.6f}"
            "{FloodStorage:<14.6f}"
            "{RiverOutflow:<14.6f}"
            "{FloodOutflow:<14.6f}"
            "{RiverInflow:<14.6f}"
            "{FloodInflow:<14.6f}"
            "{TotalStorage:<14.6f}"
            "{Runoff:<14.6f}"
            "{MassBalanceError:<18.6e}" 
        )
        formatter = logging.Formatter(fmt=fmt, style='{')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.prev_total_storage = None
        self.cumulative_mass_error = 0.0

    def write_header(self, states):
        if self.disabled:
            return
        states = {k: v.clone() for k, v in states.items()}
        with open(self.log_path, 'w') as f:
            f.write("Program Start Time: {:<24}\n".format(' '))
            f.write("Program End Time:   {:<24}\n".format(' '))
            f.write("Duration:           {:>10}\n".format(' '))
            f.write("\n")

            # Initial stats
            river_storage = states["river_storage"].sum().item() * self.scale
            flood_storage = states["flood_storage"].sum().item() * self.scale
            river_outflow = states["river_outflow"].mean().item()
            flood_outflow = states["flood_outflow"].mean().item()
            self.prev_total_storage = states["river_storage"] + states["flood_storage"]

            f.write(f"# Initial Total River Storage: {river_storage: .6f} (10^9 m3)\n")
            f.write(f"# Initial Total Flood Storage: {flood_storage: .6f} (10^9 m3)\n")
            f.write(f"# Initial Average River Outflow: {river_outflow: .6f} (m^3/s)\n")
            f.write(f"# Initial Average Flood Outflow: {flood_outflow: .6f} (m^3/s)\n")
            f.write("\n")

            # Column headers, left-aligned
            headers = [
                "StepStartTime", "StepSeconds", "RiverStorage", "FloodStorage",
                "RiverOutflow", "FloodOutflow", "RiverInflow", "FloodInflow",
                "TotalStorage", "Runoff", "MassBalanceError"
            ]
            widths = [18, 14, 14, 14, 14, 14, 14, 14, 14, 14, 18]
            header_line = ''.join(f"{h:<{w}}" for h, w in zip(headers, widths))
            f.write(header_line + "\n")

    def write_step(self, states):
        if self.disabled:
            return
        states = {k: v.clone() for k, v in states.items()}
        expected = self.prev_total_storage + (states["river_inflow"] + states["flood_inflow"] - states["river_outflow"] - states["flood_outflow"] + self.forcing) * self.time_step 
        
        err = states["total_storage"] - expected
        self.prev_total_storage = states["total_storage"]

        record = {
            "StepStartTime":    self.current_time.strftime('%Y-%m-%d %H:%M'),
            "StepSeconds":      self.time_step,
            "RiverStorage":     states["river_storage"].sum().item() * self.scale,
            "FloodStorage":     states["flood_storage"].sum().item() * self.scale,
            "RiverOutflow":     states["river_outflow"].sum().item() * self.time_step * self.scale,
            "FloodOutflow":     states["flood_outflow"].sum().item() * self.time_step * self.scale,
            "RiverInflow":      states["river_inflow"].sum().item() * self.time_step * self.scale,
            "FloodInflow":      states["flood_inflow"].sum().item() * self.time_step * self.scale,
            "TotalStorage":     states["total_storage"].sum().item() * self.scale,
            "Runoff":           self.forcing.sum().item() * self.time_step * self.scale,
            "MassBalanceError": err.sum().item() * self.scale,
        }
        self.logger.info('', extra=record)
        self.current_time += timedelta(seconds=self.time_step)

    def set_current_time(self, current_time):
        if self.disabled:
            return
        self.current_time = current_time

    def set_time_step(self, time_step):
        if self.disabled:
            return
        self.time_step = time_step

    def set_forcing(self, forcing):
        if self.disabled:
            return
        forcing = forcing.clone()
        self.forcing = forcing

    def close(self):
        if self.disabled:
            return
        end_time = datetime.now()
        # Rewrite timing placeholders
        with open(self.log_path, 'r+') as f:
            lines = f.readlines()
            f.seek(0)
            f.write(f"Program Start Time: {self.start_time:%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Program End Time:   {end_time:%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Duration:           {(end_time - self.start_time).total_seconds():.2f}\n")
            f.writelines(lines[3:])

    def __del__(self):
        self.close()