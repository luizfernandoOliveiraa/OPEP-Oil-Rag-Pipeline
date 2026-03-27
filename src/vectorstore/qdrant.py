from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Factory para o modelo de Embeddings parametrizado no .env"""
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model)


def get_qdrant_vectorstore() -> QdrantVectorStore:
    """Fornecedor Injetor da Camada de Banco Vetorial. Desacopla as instâncias directas."""
    try:
        embeddings = get_embeddings_model()
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Inicializa o cliente Qdrant para verificar/criar a coleção
        client = QdrantClient(url=settings.qdrant_url)

        if not client.collection_exists(collection_name=settings.collection_name):
            logger.info(f"💾 Criando nova coleção: {settings.collection_name}")

            # Nota: O tamanho do vetor para o Google Gemini Embedding preview costuma ser 768.
            # Se você usar outro modelo, ajuste esse valor ou use um documento dummy para criar a coleção.
            client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=models.VectorParams(
                    size=3072, distance=models.Distance.COSINE
                ),
                sparse_vectors_config={
                    "fastembed_sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )

        store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="fastembed_sparse",
        )
        return store
    except Exception as e:
        logger.error(
            f"Falha ao conectar na Coleção Qdrant '{settings.collection_name}' na URL {settings.qdrant_url}. ERRO: {e}"
        )
        raise
