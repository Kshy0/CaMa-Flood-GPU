from pydantic import Field, FilePath, DirectoryPath
from datetime import datetime
from cmfgpu.configs.template_config import TemplateConfig

class DailyBinConfig(TemplateConfig):
    runoff_dir: DirectoryPath = Field(description="Path to the runoff directory")
    runoff_mapping_file: FilePath = Field(description="Path to the runoff mapping file")
    runoff_shape: tuple = Field(description="Shape of the runoff data")
    unit_factor: float = Field(default=86400000, description="mm/day divided by unit_factor to get m/s")
    bin_dtype: str = Field(default="float32", description="Data type of the binary runoff files")
    prefix: str = Field(default="Roff____", description="Prefix for binary files")
    suffix: str = Field(default=".one", description="Suffix for binary files")