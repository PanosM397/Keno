from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    gwosc_host: str = "https://gwosc.org"
    default_detector: str = "H1"
    default_duration_seconds: int = 4

    model_checkpoint_path: str = "checkpoints/unet_denoiser.pt"
    device: str = "cpu"


settings = Settings()
