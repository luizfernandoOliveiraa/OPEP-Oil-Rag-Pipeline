from pydantic import BaseModel, Field
from typing import List, TypedDict


class ReportMetadata(BaseModel):
    """Schema para extração estruturada de metadados críticos dos relatórios de mercado."""

    report_month: str = Field(
        description="Mês de referência do relatório ou da previsão (ex: 'February', 'January'). Em inglês."
    )
    report_year: int = Field(description="Ano de referência do relatório (ex: 2026).")
    oil_price_movement: str = Field(
        description="Resumo de 1-3 palavras sobre a movimentação da Cesta de Referência (ex: 'Upward trend', 'Declined', 'Stable')."
    )
    key_drivers: List[str] = Field(
        description="Lista com 1 a 3 fatores macroeconômicos que influenciaram a perspectiva neste mês. Em inglês."
    )
    oil_market_highlights_content: str = Field(
        description="O texto COMPLETO e NA ÍNTEGRA da seção chamada 'Oil Market Highlights', copie e transcreva tudo."
    )


class GraphState(TypedDict):
    """Esquema forte de Estado tipado passado entre os nós do LangGraph."""

    question: str
    documents: list
    answer: str
