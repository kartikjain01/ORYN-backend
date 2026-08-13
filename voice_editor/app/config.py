from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Audio Editing System"
    TARGET_SR: int = 44100
    MONO: bool = True
    NOISE_REDUCTION_STRENGTH: float = 0.6
    NORMALIZE_TARGET_DB: float = -1.0
    SILENCE_THRESHOLD: float = 0.01
    EXPORT_FORMAT: str = "wav"

    model_config = SettingsConfigDict(
        env_file=".env"
    )


settings = Settings()
