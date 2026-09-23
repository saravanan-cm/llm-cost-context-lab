"""LangGraph chat workflow tests: the compiled graph invoked directly.

Fakes: retriever (no Qdrant/OpenAI) and LLM provider. Real: context builder, pricing, credit
policy and usage service over in-memory SQLite, so accounting is exercised for real.
"""

import logging
from decimal import Decimal

import pytest
from sqlalchemy.exc import OperationalError

from app.core.errors import LLMTimeoutError, VectorStoreUnavailableError
from app.graph.builder import create_chat_graph, get_chat_graph
from app.graph.context import ChatGraphContext, ChatGraphSettings
from app.graph.state import initial_state
from app.rag.context_builder import RAGContextBuilder
from app.services.credits import CreditPolicy
from app.services.metering import UsageMeter
from app.services.pricing import PricingService
from app.services.usage_service import UsageService
from tests.conftest import DEV_USER, FakeProvider, FakeRetriever, make_chunk

KAFKA_Q = "How does a Kafka consumer group work?"
KAFKA_CHUNKS = [
    make_chunk("Apache Kafka", 1, 0.62, "A consumer group shares a topic's partitions among consumers."),
    make_chunk("Apache Kafka", 2, 0.61),
    make_chunk("Distributed computing", 4, 0.33),
    make_chunk("Redis", 3, 0.21),  # below the 0.3 threshold
]
OFF_TOPIC = [make_chunk("Large language model", 0, 0.27), make_chunk("Redis", 1, 0.2)]

HAPPY_PATH = [
    "analyze_question",
    "precheck_credits",
    "retrieve_knowledge",
    "build_context",
    "check_credits",
    "generate_answer",
    "usage_accounting",
]


@pytest.fixture
def db(session_factory):
    with session_factory() as session:
        yield session


@pytest.fixture
def usage_service(db) -> UsageService:
    return UsageService(db)


@pytest.fixture
def llm() -> FakeProvider:
    return FakeProvider(input_tokens=850, output_tokens=220)


@pytest.fixture
def kb() -> FakeRetriever:
    return FakeRetriever(list(KAFKA_CHUNKS))


def make_context(retriever, llm, usage_service, **settings) -> ChatGraphContext:
    return ChatGraphContext(
        retriever=retriever,
        builder=RAGContextBuilder(),
        llm=llm,
        meter=UsageMeter(pricing=PricingService(), credits=CreditPolicy(Decimal(1000)), usage=usage_service),
        settings=ChatGraphSettings(**{"top_k": 5, "min_relevance_score": 0.3, **settings}),
    )


def run(context, message=KAFKA_Q):
    state = initial_state(user_id=DEV_USER, conversation_id="c1", user_message=message)
    return create_chat_graph().invoke(state, context=context)


def set_credits(usage_service, total):
    from app.db.models import User

    usage_service._db.get(User, DEV_USER).total_credits = Decimal(total)
    usage_service._db.commit()


# -- Test 1: normal RAG flow -----------------------------------------------------------------


def test_normal_rag_flow(kb, llm, usage_service):
    events: list[str] = []
    kb.events = llm.events = events

    final = run(make_context(kb, llm, usage_service))

    assert final["graph_path"] == HAPPY_PATH
    assert events == ["retrieve", "llm"]
    assert kb.calls == [(KAFKA_Q, 5)]
    assert final["status"] == "answered" and final["answer_type"] == "grounded"
    assert final["assistant_message"].startswith("reply to:")
    assert [s.title for s in final["sources"]] == ["Apache Kafka", "Distributed computing"]
    # provider-reported usage -> pricing -> credits -> ledger
    assert (final["usage"].input_tokens, final["usage"].output_tokens) == (850, 220)
    assert final["cost"].total_cost == Decimal("0.000692")  # 850*0.40/1M + 220*1.60/1M
    assert final["credits_consumed"] == Decimal("0.692")
    summary = usage_service.get_summary(DEV_USER)
    assert summary.request_count == 1 and summary.total_input_tokens == 850
    assert summary.credits_remaining == Decimal("999.308") == final["credits_remaining"]


def test_llm_receives_built_context_and_grounding_instructions(kb, llm, usage_service):
    run(make_context(kb, llm, usage_service))

    prompt = llm.calls[0]
    assert "consumer group shares a topic's partitions" in prompt
    assert "[1] Apache Kafka (Wikipedia)" in prompt and "[2] Distributed computing" in prompt
    assert "Redis" not in prompt  # below threshold, never sent
    assert prompt.rstrip().endswith(f"Question: {KAFKA_Q}")
    assert "Do not invent" in llm.instructions[0]


# -- Test 2: no relevant knowledge -----------------------------------------------------------


def test_no_relevant_knowledge_path(llm, usage_service):
    final = run(
        make_context(FakeRetriever(OFF_TOPIC), llm, usage_service, no_context_message="Not in the KB."),
        "What is quantum mechanics?",
    )

    assert final["graph_path"] == ["analyze_question", "precheck_credits", "retrieve_knowledge", "no_knowledge"]
    assert final["status"] == "no_knowledge" and final["answer_type"] == "no_context"
    assert final["assistant_message"] == "Not in the KB."
    assert final["model"] is None and final["sources"] == []
    assert final["credits_consumed"] == 0 and final["credits_remaining"] == Decimal(1000)
    assert llm.calls == []
    assert usage_service.get_summary(DEV_USER).request_count == 0


def test_no_knowledge_llm_mode_answers_with_flagged_prompt(llm, usage_service):
    final = run(make_context(FakeRetriever(OFF_TOPIC), llm, usage_service, no_context_mode="llm"), "Quantum?")

    assert "no_knowledge" not in final["graph_path"] and final["graph_path"][-1] == "usage_accounting"
    assert final["answer_type"] == "no_context" and final["sources"] == []
    assert "knowledge base does not contain information" in llm.instructions[0]


# -- Test 3: insufficient credits ------------------------------------------------------------


def test_no_credits_stops_before_retrieval_and_llm(kb, llm, usage_service):
    set_credits(usage_service, 0)

    final = run(make_context(kb, llm, usage_service))

    assert final["graph_path"] == ["analyze_question", "precheck_credits"]
    assert final["status"] == "insufficient_credits"
    assert kb.calls == [] and llm.calls == []
    assert usage_service.get_summary(DEV_USER).request_count == 0


def test_context_too_expensive_stops_before_llm(llm, usage_service):
    # Question alone passes the precheck; question + ~24k chars of context does not.
    kb = FakeRetriever([make_chunk("Apache Kafka", i, 0.6, "x" * 6000) for i in range(4)])
    set_credits(usage_service, 3)

    final = run(make_context(kb, llm, usage_service))

    assert final["graph_path"][-1] == "check_credits"
    assert final["status"] == "insufficient_credits"
    assert kb.calls and llm.calls == []
    summary = usage_service.get_summary(DEV_USER)
    assert summary.request_count == 0 and summary.credits_used == 0


# -- Test 4: LLM failure ---------------------------------------------------------------------


def test_llm_failure_raises_and_does_not_deduct(kb, llm, usage_service):
    llm.error = LLMTimeoutError()

    with pytest.raises(LLMTimeoutError):
        run(make_context(kb, llm, usage_service))

    summary = usage_service.get_summary(DEV_USER)
    assert summary.request_count == 0 and summary.credits_used == 0


# -- Test 5: retrieval failure ---------------------------------------------------------------


def test_retrieval_failure_raises_before_llm(kb, llm, usage_service):
    kb.error = VectorStoreUnavailableError()

    with pytest.raises(VectorStoreUnavailableError):
        run(make_context(kb, llm, usage_service))

    assert llm.calls == []
    assert usage_service.get_summary(DEV_USER).credits_used == 0


# -- Test 6: state flow between nodes --------------------------------------------------------


def test_state_is_handed_from_node_to_node(kb, llm, usage_service):
    state = initial_state(user_id=DEV_USER, conversation_id="c1", user_message=f"  {KAFKA_Q}  \n")
    steps = list(create_chat_graph().stream(state, context=make_context(kb, llm, usage_service), stream_mode="updates"))
    updates = {node: update for step in steps for node, update in step.items()}

    assert list(updates) == HAPPY_PATH
    assert updates["analyze_question"]["question"] == KAFKA_Q  # normalised
    assert updates["analyze_question"]["retrieval_query"] == KAFKA_Q
    assert kb.calls[0][0] == KAFKA_Q  # retrieval used the analysed query
    assert len(updates["retrieve_knowledge"]["retrieved_documents"]) == 4
    assert len(updates["retrieve_knowledge"]["relevant_documents"]) == 3
    assert updates["build_context"]["context"].input == llm.calls[0]  # LLM got exactly the built context
    assert updates["generate_answer"]["usage"].total_tokens == 1070
    assert updates["usage_accounting"]["credits_consumed"] == Decimal("0.692")


def test_initial_state_has_no_infrastructure_objects():
    state = initial_state(user_id="u", conversation_id="c", user_message="m")

    assert set(state) == {"user_id", "conversation_id", "user_message", "status", "graph_path"}
    assert all(isinstance(v, (str, list)) for v in state.values())


# -- other routing / error cases -------------------------------------------------------------


def test_empty_question_ends_immediately(kb, llm, usage_service):
    final = run(make_context(kb, llm, usage_service), "   \n\t ")

    assert final["graph_path"] == ["analyze_question"]
    assert final["status"] == "invalid_question"
    assert kb.calls == [] and llm.calls == []


def test_rag_disabled_skips_retrieval(kb, llm, usage_service):
    final = run(make_context(kb, llm, usage_service, rag_enabled=False), "Hi")

    assert final["graph_path"] == [p for p in HAPPY_PATH if p != "retrieve_knowledge"]
    assert final["answer_type"] == "direct" and kb.calls == []
    assert llm.calls == ["Hi"] and llm.instructions == [None]


def test_top_k_and_threshold_come_from_settings(kb, llm, usage_service):
    final = run(make_context(kb, llm, usage_service, top_k=2, min_relevance_score=None))

    assert kb.calls == [(KAFKA_Q, 2)]
    assert len(final["relevant_documents"]) == 2


def test_usage_recording_failure_propagates(kb, llm, usage_service, monkeypatch):
    def broken(entry):
        raise OperationalError("INSERT", {}, Exception("disk full"))

    monkeypatch.setattr(usage_service, "record_usage", broken)

    with pytest.raises(OperationalError):
        run(make_context(kb, llm, usage_service))
    assert llm.calls  # the LLM was called; failure is at accounting (logged for reconciliation)


def test_node_executions_are_logged(kb, llm, usage_service, caplog):
    caplog.set_level(logging.INFO, logger="app.graph.nodes")

    run(make_context(kb, llm, usage_service))

    records = [r for r in caplog.records if r.getMessage() == "graph_node"]
    assert [r.node for r in records] == HAPPY_PATH
    assert all(r.outcome == "ok" and isinstance(r.duration_ms, int) for r in records)
    retrieve = records[2]
    assert (retrieve.retrieved, retrieve.relevant, retrieve.top_score) == (4, 3, 0.62)
    assert records[5].input_tokens == 850 and records[6].total_cost_usd == "0.000692"


def test_compiled_graph_is_reused():
    assert get_chat_graph() is get_chat_graph()
