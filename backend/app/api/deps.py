from typing import Annotated

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.factory import get_embedding_provider, get_vector_store
from app.knowledge.retriever import KnowledgeRetriever
from app.knowledge.vector_store import VectorStore
from app.graph.builder import get_chat_graph
from app.graph.context import ChatGraphContext, ChatGraphSettings
from app.rag.context_builder import RAGContextBuilder
from app.services.chat_service import ChatService
from app.services.credits import CreditPolicy
from app.services.llm import LLMProvider, get_llm_provider
from app.services.metering import UsageMeter
from app.services.pricing import PricingService
from app.services.usage_service import UsageService

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_current_user_id(settings: SettingsDep) -> str:
    # No authentication yet: every request acts as the fixed development user.
    return settings.dev_user_id


def get_pricing_service() -> PricingService:
    return PricingService()


def get_credit_policy(settings: SettingsDep) -> CreditPolicy:
    return CreditPolicy(settings.credits_per_dollar)


def get_usage_service(db: Annotated[Session, Depends(get_db)]) -> UsageService:
    return UsageService(db)


def get_retriever(
    embedder: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> KnowledgeRetriever:
    return KnowledgeRetriever(embedder, store)


def get_context_builder() -> RAGContextBuilder:
    return RAGContextBuilder()


def get_usage_meter(
    pricing: Annotated[PricingService, Depends(get_pricing_service)],
    credits: Annotated[CreditPolicy, Depends(get_credit_policy)],
    usage: Annotated[UsageService, Depends(get_usage_service)],
) -> UsageMeter:
    return UsageMeter(pricing=pricing, credits=credits, usage=usage)


def get_chat_service(
    settings: SettingsDep,
    graph: Annotated[CompiledStateGraph, Depends(get_chat_graph)],
    retriever: Annotated[KnowledgeRetriever, Depends(get_retriever)],
    builder: Annotated[RAGContextBuilder, Depends(get_context_builder)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
    meter: Annotated[UsageMeter, Depends(get_usage_meter)],
) -> ChatService:
    context = ChatGraphContext(
        retriever=retriever,
        builder=builder,
        llm=llm,
        meter=meter,
        settings=ChatGraphSettings.from_settings(settings),
    )
    return ChatService(graph=graph, context=context, debug=settings.rag_debug)
