from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM configuration. Tool calling needs a capable model (gpt-4o or newer).
    model_name: str = "gpt-4o"
    model_provider: str = "openai"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 2

    # Optional override for OpenAI-compatible endpoints.
    openai_base_url: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    anthropic_base_url: str = "https://api.anthropic.com/v1/"

    # Credentials. Optional so the app can start and let the CLI collect the key
    # at runtime instead of requiring it in the environment.
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()