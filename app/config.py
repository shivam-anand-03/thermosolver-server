"""Environment-driven configuration (see ``.env.example``)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; override with environment variables prefixed ``THERMOSOLVE_``."""

    model_config = SettingsConfigDict(env_prefix="THERMOSOLVE_", env_file=".env", extra="ignore")

    app_name: str = "ThermoSolve API"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_nodes: int = 2000
    max_time_steps: int = 20000
    max_node_steps: int = 20_000_000
    record_limit: int = 12  # store every augmented matrix when N <= this
    matrix_payload_limit: int = 100  # send the dense matrix when N <= this
    max_snapshots: int = 200  # transient output sampling target
    max_snapshot_values: int = 300_000

    # three-phase workflow engine (/api/v2)
    engine_max_per_axis: int = 401
    engine_max_unknowns_direct: int = 2500  # Gauss elimination / LU work on a dense band copy
    engine_max_unknowns_iterative: int = 10000  # Gauss–Seidel
    engine_max_node_steps: int = 3_000_000
    engine_max_snapshots: int = 60
    engine_max_snapshot_values: int = 200_000
    engine_matrix_preview: int = 30  # send the dense [A] and {B} when n <= this
    engine_pattern_limit: int = 900  # send the sparsity pattern when n <= this

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
