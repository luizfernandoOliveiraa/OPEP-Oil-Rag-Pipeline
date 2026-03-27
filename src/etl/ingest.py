import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import Settings
from src.core.logger import get_logger
from src.schemas.models import ReportMetadata
from src.vectorstore.qdrant import get_qdrant_vectorstore

settings = Settings()

logger = get_logger(__name__)
DATA_DIR = Path("data")
PROCESSED_REGISTRY = Path("processed_files.json")


def load_processed_registry() -> dict:
    if PROCESSED_REGISTRY.exists():
        with open(PROCESSED_REGISTRY, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_processed_registry(registry: dict):
    with open(PROCESSED_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4)


def get_file_hash(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def extract_metadata_from_text(text: str) -> ReportMetadata:
    logger.debug("Acessando LLM para extração bruta dos relatórios...")
    llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0)
    structured_llm = llm.with_structured_output(ReportMetadata)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um analista experiente no mercado de commodities energéticas. Extraia RIGOROSAMENTE as informações demandadas, e transcreva em destaque o conteúdo integral da tabela/seção 'Oil Market Highlights'.",
            ),
            ("human", "Analise o texto e devolva o schema Pydantic:\n\n{text}"),
        ]
    )

    chain = prompt | structured_llm
    return chain.invoke({"text": text[:45000]})


def ingest_document(filepath: Path, qdrant_store):
    registry = load_processed_registry()
    file_hash = get_file_hash(filepath)
    filename = filepath.name

    if filename in registry and registry[filename] == file_hash:
        logger.info(
            f"[IDEMPOTÊNCIA] Arquivo {filename} já indexado neste bucket de hash. Pulando..."
        )
        return

    logger.info(f"🚀 [START] Processando Ingestão de {filename}")

    logger.info("  [1/4] Realizando extração raw de texto (Bypass DOCLING)...")
    reader = PdfReader(filepath)
    text_content = ""
    for i in range(min(15, len(reader.pages))):
        text_content += (
            f"--- PAGE {i + 1} ---\n" + reader.pages[i].extract_text() + "\n"
        )

    logger.info(
        f"        -> Extraídos {len(text_content)} caracteres. Feedando GenAI..."
    )

    logger.info("  [2/4] Minorando Metadados com Pydantic+Gemini...")
    try:
        metadata = extract_metadata_from_text(text_content)
        logger.info(
            f"        -> Extração Positiva [Mês: {metadata.report_month}/{metadata.report_year}] | [Indicador: {metadata.oil_price_movement}]"
        )
    except Exception as e:
        logger.error(f"Falha crítica de parse no documento {filename}: {e}")
        return

    logger.info("  [3/4] Aplicando Split Recursivo com inserção em metadados...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.create_documents(
        texts=[metadata.oil_market_highlights_content]
    )

    for chunk in chunks:
        chunk.metadata.update(
            {
                "source_file": filename,
                "report_month": metadata.report_month,
                "report_year": metadata.report_year,
                "oil_price_movement": metadata.oil_price_movement,
                "key_drivers": ", ".join(metadata.key_drivers),
            }
        )
    logger.info(
        f"        -> Encapsulados {len(chunks)} chunks validados no Oil Market Highlights."
    )

    logger.info(
        "  [4/4] Adicionando Chunks à base vetorial e efetuando Commit estrutural..."
    )
    qdrant_store.add_documents(chunks)

    registry[filename] = file_hash
    save_processed_registry(registry)
    logger.info(
        f"✅ [SUCCESS] OPEP Report {filename} incorporado semanticamente ao pipeline!"
    )


def run_pipeline():
    logger.info("=== Booting ETL Pipeline - RAG Market Intelligence ===")

    if not settings.google_api_key:
        logger.error("GOOGLE_API_KEY ausente nas dependencias globais (.env)")
        return

    try:
        qdrant_store = get_qdrant_vectorstore()
    except Exception as e:
        logger.error(f"Halted Pipeline: Base vetorial irresponsiva. {e}")
        return

    if DATA_DIR.exists():
        files = list(DATA_DIR.glob("*.pdf"))
        logger.info(
            f"Descobertos {len(files)} subarquivos injetáveis de extensão .pdf."
        )
        for pdf_file in files:
            ingest_document(pdf_file, qdrant_store)
    else:
        logger.error(f"Pasta target '{DATA_DIR}' intrafegável ou não encontrada.")


if __name__ == "__main__":
    run_pipeline()
