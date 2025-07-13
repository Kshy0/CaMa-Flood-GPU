from pydantic import Field, FilePath, DirectoryPath
from datetime import datetime
from cmfgpu.configs.template_config import TemplateConfig

class YearlyNetCDFConfig(TemplateConfig):
    runoff_dir: DirectoryPath = Field(description="Path to the runoff directory")
    runoff_mapping_file: FilePath = Field(description="Path to the runoff mapping file")
    unit_factor: float = Field(default=86400000, description="mm/day divided by unit_factor to get m/s")
    var_name: str = Field(default="Runoff", description="Variable name in the NetCDF files")
    prefix: str = Field(default="e2o_ecmwf_wrr2_glob15_day_Runoff_", description="Prefix for NetCDF files")
    suffix: str = Field(default=".nc", description="Suffix for NetCDF files")