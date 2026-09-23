"""Chat graph construction. Compiled once and reused; per-request services are passed via
``graph.invoke(state, context=ChatGraphContext(...))``.

    START -> analyze_question
    analyze_question   --(empty question)--------> END
                       --(otherwise)-------------> precheck_credits
    precheck_credits   --(insufficient)-----------> END
                       --(RAG enabled)-----------> retrieve_knowledge
                       --(RAG disabled)----------> build_context
    retrieve_knowledge --(no relevant documents)--> no_knowledge -> END
                       --(relevant documents)-----> build_context
    build_context -> check_credits
    check_credits      --(insufficient)-----------> END
                       --(sufficient)-------------> generate_answer
    generate_answer -> usage_accounting -> END
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph import nodes
from app.graph.context import ChatGraphContext
from app.graph.state import ChatState


def create_chat_graph() -> CompiledStateGraph:
    graph = StateGraph(ChatState, context_schema=ChatGraphContext)

    graph.add_node(nodes.ANALYZE_QUESTION, nodes.analyze_question)
    graph.add_node(nodes.PRECHECK_CREDITS, nodes.precheck_credits)
    graph.add_node(nodes.RETRIEVE_KNOWLEDGE, nodes.retrieve_knowledge)
    graph.add_node(nodes.NO_KNOWLEDGE, nodes.no_knowledge)
    graph.add_node(nodes.BUILD_CONTEXT, nodes.build_context)
    graph.add_node(nodes.CHECK_CREDITS, nodes.check_credits)
    graph.add_node(nodes.GENERATE_ANSWER, nodes.generate_answer)
    graph.add_node(nodes.USAGE_ACCOUNTING, nodes.usage_accounting)

    graph.add_edge(START, nodes.ANALYZE_QUESTION)
    graph.add_conditional_edges(
        nodes.ANALYZE_QUESTION,
        nodes.route_after_analysis,
        {"end": END, nodes.PRECHECK_CREDITS: nodes.PRECHECK_CREDITS},
    )
    graph.add_conditional_edges(
        nodes.PRECHECK_CREDITS,
        nodes.route_after_precheck,
        {"end": END, nodes.RETRIEVE_KNOWLEDGE: nodes.RETRIEVE_KNOWLEDGE, nodes.BUILD_CONTEXT: nodes.BUILD_CONTEXT},
    )
    graph.add_conditional_edges(
        nodes.RETRIEVE_KNOWLEDGE,
        nodes.route_after_retrieval,
        {nodes.NO_KNOWLEDGE: nodes.NO_KNOWLEDGE, nodes.BUILD_CONTEXT: nodes.BUILD_CONTEXT},
    )
    graph.add_edge(nodes.NO_KNOWLEDGE, END)
    graph.add_edge(nodes.BUILD_CONTEXT, nodes.CHECK_CREDITS)
    graph.add_conditional_edges(
        nodes.CHECK_CREDITS,
        nodes.route_after_credit_check,
        {"end": END, nodes.GENERATE_ANSWER: nodes.GENERATE_ANSWER},
    )
    graph.add_edge(nodes.GENERATE_ANSWER, nodes.USAGE_ACCOUNTING)
    graph.add_edge(nodes.USAGE_ACCOUNTING, END)

    return graph.compile(name="chat_graph")


@lru_cache
def get_chat_graph() -> CompiledStateGraph:
    return create_chat_graph()
