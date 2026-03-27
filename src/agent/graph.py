from langgraph.graph import StateGraph, START, END

from src.schemas.models import GraphState
from src.agent.nodes import retrieve_node, generate_node


def build_graph():
    """Establece e Compila o pipeline analítico Sênior - Workflow OPEP AI."""
    workflow = StateGraph(GraphState)

    # Injeção Estática de Nós
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Topologia Lógica RAG
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# Singleton injetado e empacotado para o entrypoint isoladamente
rag_agent = build_graph()


def invoke_rag(question: str):
    """Invoker utilitario empacotando e escondendo complexidade de graph"""
    return rag_agent.invoke({"question": question})
