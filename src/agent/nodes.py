from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler

from src.core.config import settings
from src.core.logger import get_logger
from src.schemas.models import GraphState
from src.vectorstore.qdrant import get_qdrant_vectorstore

logger = get_logger(__name__)


def retrieve_node(state: GraphState):
    """Nó RAG: Busca os chunks estritos no local Qdrant Vector DB."""
    logger.info("--- 🔍 EXECUTANDO NÓ DE BUSCA QDRANT ---")
    question = state["question"]
    qdrant = get_qdrant_vectorstore()

    docs = qdrant.similarity_search(question, k=3)
    logger.info(
        "        -> Foram resgatados %d fragmentos base para o contexto.", len(docs)
    )
    return {"documents": docs}


def generate_node(state: GraphState):
    """Nó RAG: Extrai a resposta da rede LLM conectando os chunks passados."""
    logger.info("--- 🤖 EXECUTANDO NÓ GERADOR DE INTELIGENCIA (GEMINI) ---")
    question = state["question"]
    documents = state["documents"]

    context_text = "\n\n".join(
        f"FONTE OPEP ({doc.metadata.get('report_month')}/{doc.metadata.get('report_year')}) | TENDÊNCIA MACRO DIRECIONAL CORRELATA: {doc.metadata.get('oil_price_movement')}\nCONTEÚDO EVIDENCIAL:\n{doc.page_content}"
        for doc in documents
    )

    prompt = ChatPromptTemplate.from_template(
        """Olá! Como seu consultor de inteligência para o mercado de energia, analisei os dados mais recentes do relatório MOMR (OPEP) para te ajudar com essa questão.

        CONTEXTO DOS RELATÓRIOS:
        {context}

        DÚVIDA DO CLIENTE:
        {question}

        DIRETRIZES PARA A RESPOSTA (ESTILO CONSULTORIA):
        1. Seja cordial e amigável, mas mantenha o rigor técnico.
        2. Não use sempre a mesma introdução. Varie o início da frase para parecer uma conversa natural entre especialistas.
        3. Use os dados resgatados para embasar sua análise (cite meses, anos e tendências mencionadas nas fontes).
        4. Explicite de onde veio a informação de forma integrada ao texto (ex: "Segundo o levantamento de Março de 2026...", "Os dados da OPEP apontam que...").
        5. Se a resposta não estiver nos documentos fornecidos, admita educadamente que não possuímos dados específicos sobre isso no repositório atual.
        6. Use uma estrutura de parágrafos fluida para facilitar a leitura.
        """
    )

    llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0.3)
    chain = prompt | llm | StrOutputParser()

    callbacks = []
    langfuse_handler = None
    if settings.langfuse_public_key:
        try:
            # Tenta inicializar com a sintaxe padrão do Langfuse SDK para Langchain
            langfuse_handler = CallbackHandler()
            callbacks.append(langfuse_handler)
        except Exception as e:
            logger.warning(f"Não foi possível inicializar o tracing do Langfuse: {e}")

    response = chain.invoke(
        {"context": context_text, "question": question}, 
        config={"callbacks": callbacks}
    )

    if langfuse_handler and hasattr(langfuse_handler, "flush"):
        langfuse_handler.flush()
    elif langfuse_handler and hasattr(langfuse_handler, "langfuse"):
        langfuse_handler.langfuse.flush()

    return {"answer": response}
