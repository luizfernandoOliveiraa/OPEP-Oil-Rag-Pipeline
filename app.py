import streamlit as st

from src.agent.graph import invoke_rag

# Base da Janela e Identidade Visual Mínima
st.set_page_config(
    page_title="Petróleo MOMR | Market Intelligence", page_icon="🛢️", layout="wide"
)

st.title("🛢️ OPEP Market Intelligence Agent")
st.markdown(
    "Plataforma de mineração de inteligência sobre o mercado global de commodities enérgéticas baseada no relatório **MOMR** (OPEP)."
)

# Setup state do Histórico do chat interativo da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as bolhas antigas sempre que o frontend reliza refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura de input e Prompt do Usuário
if prompt := st.chat_input(
    "Ex: Qual o crescimento da demanda no último relatório? A China tem participação primária nessa oscilação?"
):
    # Exibe pergunta e adiciona na sessão
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Inicia a UI de Resposta Automática da IA
    with st.chat_message("assistant"):
        with st.spinner(
            "Pesquisando na Vector DB (Local) e Raciocinando com Google Gemini..."
        ):
            try:
                # Onde a mágica acontece! Submetemos para o Grafo (LangGraph)
                result = invoke_rag(prompt)

                answer = result["answer"]
                docs = result.get("documents", [])

                # Joga a resposta escrita pelo LLM limpa na tela
                st.markdown(answer)

                # Exibição analítica de Referências com Expander elegante
                if docs:
                    with st.expander("📚 Fontes Extraídas pelo Vector Store (Qdrant)"):
                        for idx, doc in enumerate(docs):
                            st.write(
                                f"**Trecho Recuperado {idx + 1}**: {doc.metadata.get('source_file')} - Referência: {doc.metadata.get('report_month')}/{doc.metadata.get('report_year')}"
                            )
                            # Nota mental: doc.metadata.get('oil_price_movement') captura a resposta da extração via schema pydantic
                            st.caption(
                                f"Direcional do Barril Extraído Primariamente: {doc.metadata.get('oil_price_movement')}"
                            )
                else:
                    st.warning(
                        "⚠️ Nenhum documento correlacionado de alta confidência foi achado no VectorDB."
                    )

            except Exception as e:
                answer = f"Ocorreu um erro no pipeline RAG: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
