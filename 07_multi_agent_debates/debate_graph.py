import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langsmith import traceable

from agents import (
    create_academia_agent,
    create_industry_agent,
    create_moderator_agent,
    get_academia_response,
    get_industry_response,
    get_moderator_decision,
    run_self_debate,
)


class DebateState(TypedDict):
    document_content: str
    topic: str

    academia_initial: str
    industry_response: str
    academia_rebuttal: str
    industry_rebuttal: str

    moderator_decision: str

    round_count: int
    novelty_score: float
    factuality_score: float

    document_improvements: str
    debate_transcript: Annotated[list[str], operator.add]


@traceable(name="academia_opening")
def academia_opening_node(state: DebateState) -> dict:
    agent = create_academia_agent()

    response = get_academia_response(
        agent=agent,
        document_content=state["document_content"],
        topic=state["topic"],
        industry_position=None
    )

    return {
        "academia_initial": response,
        "debate_transcript": [f"=== OPENING ===\n{response}"],
        "round_count": 1
    }


@traceable(name="industry_response")
def industry_response_node(state: DebateState) -> dict:
    agent = create_industry_agent()

    response = get_industry_response(
        agent=agent,
        document_content=state["document_content"],
        topic=state["topic"],
        academia_position=state["academia_initial"]
    )

    return {
        "industry_response": response,
        "debate_transcript": [f"\n=== RESPONSE ===\n{response}"]
    }


@traceable(name="academia_rebuttal")
def academia_rebuttal_node(state: DebateState) -> dict:
    agent = create_academia_agent()

    response = get_academia_response(
        agent=agent,
        document_content=state["document_content"],
        topic=state["topic"],
        industry_position=state["industry_response"]
    )

    return {
        "academia_rebuttal": response,
        "debate_transcript": [f"\n=== REBUTTAL ===\n{response}"],
        "round_count": state["round_count"] + 1
    }


@traceable(name="industry_rebuttal")
def industry_rebuttal_node(state: DebateState) -> dict:
    agent = create_industry_agent()

    response = get_industry_response(
        agent=agent,
        document_content=state["document_content"],
        topic=state["topic"],
        academia_position=state["academia_rebuttal"]
    )

    return {
        "industry_rebuttal": response,
        "debate_transcript": [f"\n=== FINAL REBUTTAL ===\n{response}"]
    }


@traceable(name="moderator_synthesis")
def moderator_synthesis_node(state: DebateState) -> dict:
    agent = create_moderator_agent()

    decision = get_moderator_decision(
        agent=agent,
        document_content=state["document_content"],
        topic=state["topic"],
        academia_position=state["academia_initial"],
        industry_position=state["industry_response"],
        academia_rebuttal=state["academia_rebuttal"],
        industry_rebuttal=state["industry_rebuttal"]
    )

    novelty_score = 0.7
    factuality_score = 0.8

    if "Novelty Score:" in decision:
        try:
            score_line = [l for l in decision.split('\n') if 'Novelty Score:' in l][0]
            novelty_score = float(score_line.split(':')[1].strip().split()[0])
        except:
            pass

    if "Factuality Score:" in decision:
        try:
            score_line = [l for l in decision.split('\n') if 'Factuality Score:' in l][0]
            factuality_score = float(score_line.split(':')[1].strip().split()[0])
        except:
            pass

    return {
        "moderator_decision": decision,
        "novelty_score": novelty_score,
        "factuality_score": factuality_score,
        "debate_transcript": [f"\n=== DECISION ===\n{decision}"],
        "document_improvements": decision
    }


def build_debate_graph() -> StateGraph:
    workflow = StateGraph(DebateState)

    workflow.add_node("academia_opening", academia_opening_node)
    workflow.add_node("industry_response", industry_response_node)
    workflow.add_node("academia_rebuttal", academia_rebuttal_node)
    workflow.add_node("industry_rebuttal", industry_rebuttal_node)
    workflow.add_node("moderator_synthesis", moderator_synthesis_node)

    workflow.set_entry_point("academia_opening")
    workflow.add_edge("academia_opening", "industry_response")
    workflow.add_edge("industry_response", "academia_rebuttal")
    workflow.add_edge("academia_rebuttal", "industry_rebuttal")
    workflow.add_edge("industry_rebuttal", "moderator_synthesis")
    workflow.add_edge("moderator_synthesis", END)

    return workflow.compile()


DEBATE_TOPICS = [
    "What evaluation metrics and frameworks are missing for production LLM agents?",
    "How should we evaluate multi-turn reasoning and credit assignment in practice?",
    "What's the right balance between LLM-as-judge and deterministic evaluation?",
]


@traceable(name="run_full_debate")
def run_debate(document_content: str, topic: str) -> dict:
    graph = build_debate_graph()

    initial_state = {
        "document_content": document_content,
        "topic": topic,
        "academia_initial": "",
        "industry_response": "",
        "academia_rebuttal": "",
        "industry_rebuttal": "",
        "moderator_decision": "",
        "round_count": 0,
        "novelty_score": 0.0,
        "factuality_score": 0.0,
        "document_improvements": "",
        "debate_transcript": []
    }

    result = graph.invoke(initial_state)
    return result


@traceable(name="run_all_debates")
def run_all_debates(document_content: str, use_self_debate: bool = False, num_rounds: int = 2) -> list[dict]:
    results = []

    for topic in DEBATE_TOPICS:
        print(f"\n{'='*60}")
        print(f"DEBATE TOPIC: {topic}")
        if use_self_debate:
            print(f"MODE: Self-Debate ({num_rounds} rounds)")
        else:
            print("MODE: Multi-Agent")
        print('='*60)

        if use_self_debate:
            result = run_self_debate(document_content, topic, num_rounds=num_rounds)
            results.append({
                "topic": topic,
                "result": result,
                "mode": "self-debate",
                "num_rounds": num_rounds
            })
        else:
            result = run_debate(document_content, topic)
            results.append({
                "topic": topic,
                "result": result,
                "mode": "multi-agent",
                "num_rounds": 4
            })

        print(f"\nNovelty Score: {result['novelty_score']}")
        print(f"Factuality Score: {result['factuality_score']}")

    return results


@traceable(name="run_self_debate_all")
def run_all_self_debates(document_content: str, num_rounds: int = 2) -> list[dict]:
    return run_all_debates(document_content, use_self_debate=True, num_rounds=num_rounds)


if __name__ == "__main__":
    test_doc = """
    This is a test document about LLM evaluation.
    """

    result = run_debate(test_doc, DEBATE_TOPICS[0])
    print("\n".join(result["debate_transcript"]))
