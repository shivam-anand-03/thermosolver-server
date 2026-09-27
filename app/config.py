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

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
