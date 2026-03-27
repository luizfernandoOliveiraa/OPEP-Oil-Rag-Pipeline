from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Configurações globais parametrizáveis via Env Vars."""

    # Google API
    google_api_key: str = ""

    # Modelos Parametrizáveis
    embedding_model: str = "gemini-embedding-2-preview"
    llm_model: str = "gemini-2.5-flash"

    # Qdrant Database
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "oil_reports_hybrid"

    # Langfuse Observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://us.cloud.langfuse.com"

    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton global das configurações instanciado apenas uma vez
settings = Settings()
