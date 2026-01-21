"""Debate Agents for multi-agent debate system."""

from typing import Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class DebateResponse(BaseModel):
    position: str = Field(description="The agent's stance on the topic")
    key_points: list[str] = Field(description="Main arguments supporting the position")
    critiques: list[str] = Field(description="Critiques of the opposing view")
    proposed_additions: list[str] = Field(description="Specific content to add to the document")
    confidence: float = Field(description="Confidence in position (0-1)")


class ModeratorDecision(BaseModel):
    synthesis: str = Field(description="Balanced synthesis of both perspectives")
    accepted_from_academia: list[str] = Field(description="Points accepted from academia")
    accepted_from_industry: list[str] = Field(description="Points accepted from industry")
    rejected_points: list[str] = Field(description="Points rejected with reasoning")
    final_recommendations: list[str] = Field(description="Final recommendations for document")
    novelty_score: float = Field(description="Novelty of proposed changes (0-1)")
    factuality_score: float = Field(description="Factual accuracy assessment (0-1)")


ACADEMIA_SYSTEM_PROMPT = """You are Dr. Elena Chen, a Principal Research Scientist at a top AI research lab (DeepMind/Anthropic/OpenAI Research).

BACKGROUND:
- PhD from Stanford in Machine Learning, postdoc at MIT
- Published 50+ papers on LLM evaluation, reward modeling, and agentic systems
- Known for rigorous empirical methodology and theoretical grounding
- Skeptical of hype, demands reproducibility and statistical significance

DEBATE STYLE:
- Cite specific papers with arxiv IDs when possible
- Emphasize methodology, reproducibility, and theoretical foundations
- Point out when claims lack empirical evidence
- Advocate for standardized benchmarks and metrics
- Consider long-term research implications over short-term gains

YOUR PERSPECTIVE ON LLM/AGENT EVALUATION:
- Process reward models (PRMs) are underexplored vs outcome reward models
- Multi-turn credit assignment needs more theoretical grounding
- Current benchmarks are saturating too quickly - we need harder ones
- LLM-as-judge has fundamental issues with self-preference bias
- Agentic evaluation requires formal verification methods, not just empirical testing

When debating, always ground your arguments in research and theory. Be critical but constructive."""


INDUSTRY_SYSTEM_PROMPT = """You are Marcus Rodriguez, VP of AI Platform at a leading tech company (scale of Anthropic/OpenAI/Google).

BACKGROUND:
- MS from CMU, 15 years building production ML systems
- Led teams shipping AI products used by millions
- Focus on practical deployment, latency, cost, and reliability
- Frustrated by academic solutions that don't scale

DEBATE STYLE:
- Emphasize real-world deployment challenges
- Discuss cost/latency/reliability tradeoffs
- Share production war stories and failure modes
- Focus on what actually works at scale
- Pragmatic about "good enough" vs "theoretically optimal"

YOUR PERSPECTIVE ON LLM/AGENT EVALUATION:
- Academic benchmarks don't predict production performance
- Need eval systems that run in CI/CD pipelines
- LLM-as-judge is practical and scales, despite imperfections
- Turn-level credit assignment is too expensive for production
- Focus on user satisfaction metrics, not abstract accuracy
- Observability and debugging > formal verification

When debating, always bring real-world constraints into the discussion. Push back on ivory tower solutions."""


MODERATOR_SYSTEM_PROMPT = """You are Dr. Yann-Ilya Hassabis, a legendary AI researcher who bridges academia and industry.

BACKGROUND:
- Founded both a top AI research lab and successful AI company
- Known for long-term vision while delivering practical results
- Respected by both academics and practitioners
- Thinks in decades, acts in quarters

MODERATOR PRINCIPLES:
1. Find synthesis between theoretical rigor and practical deployment
2. Identify genuinely novel contributions vs rehashed ideas
3. Assess factual accuracy of both sides' claims
4. Push for concrete, actionable recommendations
5. Maintain intellectual honesty - call out BS from either side

YOUR ROLE:
- Listen to both Academia and Industry perspectives
- Identify where they agree (often more than they admit)
- Find productive middle ground
- Make final decisions on what to include in the document
- Score proposals on novelty and factuality

Be diplomatic but decisive. Your goal is to produce the best possible document, not to keep everyone happy."""


def create_academia_agent(model: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        temperature=0.7,
        max_tokens=4096,
    )


def create_industry_agent(model: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        temperature=0.7,
        max_tokens=4096,
    )


def create_moderator_agent(model: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        temperature=0.5,
        max_tokens=4096,
    )


def get_academia_response(
    agent: ChatAnthropic,
    document_content: str,
    topic: str,
    industry_position: str | None = None
) -> str:
    messages = [SystemMessage(content=ACADEMIA_SYSTEM_PROMPT)]

    if industry_position:
        prompt = f"""You are debating how to improve this research document on LLM and Agentic Evaluation.

CURRENT DOCUMENT:
{document_content}

DEBATE TOPIC: {topic}

INDUSTRY'S POSITION:
{industry_position}

Respond to the industry perspective. Provide your academic counterpoints, supported by research.
Include specific recommendations for improving the document from an academic rigor perspective.

Format your response as:
## My Position
[Your stance]

## Key Research-Backed Arguments
- [Point 1 with citation if possible]
- [Point 2]
...

## Critiques of Industry View
- [Critique 1]
...

## Proposed Document Additions
- [Specific content to add]
...
"""
    else:
        prompt = f"""You are starting a debate on how to improve this research document on LLM and Agentic Evaluation.

CURRENT DOCUMENT:
{document_content}

DEBATE TOPIC: {topic}

Provide your initial academic perspective on what's missing or could be improved.
Focus on theoretical foundations, research gaps, and methodological rigor.

Format your response as:
## My Position
[Your stance]

## Key Research-Backed Arguments
- [Point 1 with citation if possible]
- [Point 2]
...

## What's Missing from Academic Perspective
- [Gap 1]
...

## Proposed Document Additions
- [Specific content to add]
...
"""

    messages.append(HumanMessage(content=prompt))
    response = agent.invoke(messages)
    return response.content


def get_industry_response(
    agent: ChatAnthropic,
    document_content: str,
    topic: str,
    academia_position: str
) -> str:
    messages = [SystemMessage(content=INDUSTRY_SYSTEM_PROMPT)]

    prompt = f"""You are debating how to improve this research document on LLM and Agentic Evaluation.

CURRENT DOCUMENT:
{document_content}

DEBATE TOPIC: {topic}

ACADEMIA'S POSITION:
{academia_position}

Respond to the academic perspective. Provide your industry counterpoints based on real-world experience.
Include specific recommendations for making the document more practical and actionable.

Format your response as:
## My Position
[Your stance]

## Key Practical Arguments
- [Point 1 from production experience]
- [Point 2]
...

## Critiques of Academic View
- [Critique 1]
...

## Proposed Document Additions
- [Specific practical content to add]
...
"""

    messages.append(HumanMessage(content=prompt))
    response = agent.invoke(messages)
    return response.content


def get_moderator_decision(
    agent: ChatAnthropic,
    document_content: str,
    topic: str,
    academia_position: str,
    industry_position: str,
    academia_rebuttal: str,
    industry_rebuttal: str
) -> str:
    messages = [SystemMessage(content=MODERATOR_SYSTEM_PROMPT)]

    prompt = f"""You are moderating a debate to improve this research document on LLM and Agentic Evaluation.

CURRENT DOCUMENT:
{document_content}

DEBATE TOPIC: {topic}

=== DEBATE TRANSCRIPT ===

ACADEMIA (Initial):
{academia_position}

INDUSTRY (Response):
{industry_position}

ACADEMIA (Rebuttal):
{academia_rebuttal}

INDUSTRY (Rebuttal):
{industry_rebuttal}

=== YOUR TASK ===

As the visionary moderator, synthesize this debate and make final decisions.

Provide your ruling in this format:

## Synthesis
[Balanced summary of where both sides agree and disagree]

## Accepted from Academia
- [Specific point 1] - Reason for acceptance
- [Point 2]
...

## Accepted from Industry
- [Specific point 1] - Reason for acceptance
- [Point 2]
...

## Rejected Points
- [Point from either side] - Reason for rejection
...

## Final Recommendations for Document
[Specific, actionable content to add to the document. Write it in a form that can be directly inserted.]

## Scores
- Novelty Score: [0-1] - How novel are the accepted contributions?
- Factuality Score: [0-1] - How factually accurate were the arguments?
"""

    messages.append(HumanMessage(content=prompt))
    response = agent.invoke(messages)
    return response.content


SELF_DEBATE_SYSTEM_PROMPT = """You are an expert AI researcher capable of deeply understanding multiple perspectives.

In self-debate mode, you will argue BOTH sides of an issue:
1. First, take the PROPONENT position (argue FOR the proposition)
2. Then, take the OPPONENT position (argue AGAINST)
3. Finally, synthesize the best insights from both perspectives

This is a rigorous intellectual exercise. Each position should be argued as strongly and fairly as possible,
as if you genuinely believed it. Do not strawman either side.

Be specific, cite research when possible, and provide concrete recommendations."""


def run_self_debate(
    document_content: str,
    topic: str,
    num_rounds: int = 2
) -> dict:
    agent = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=4096,
    )

    transcript = []
    proponent_points = []
    opponent_points = []

    for round_num in range(1, num_rounds + 1):
        # PROPONENT round
        if round_num == 1:
            proponent_prompt = f"""You are in SELF-DEBATE mode on the topic: "{topic}"

DOCUMENT CONTEXT:
{document_content[:4000]}

Take the PROPONENT position (arguing FOR this perspective/approach).

Round {round_num} - PROPONENT:
Make your strongest case. Be specific, cite research, propose concrete additions to the document.

Format:
## Proponent Position (Round {round_num})
[Your argument]

## Key Supporting Evidence
- [Point 1]
- [Point 2]
...

## Proposed Additions
- [Specific content to add]
"""
        else:
            proponent_prompt = f"""Continue the self-debate. You previously argued:

OPPONENT (Round {round_num-1}):
{opponent_points[-1] if opponent_points else 'N/A'}

Now respond as PROPONENT (Round {round_num}). Address the opponent's critiques and strengthen your position.

Format:
## Proponent Rebuttal (Round {round_num})
[Your rebuttal and strengthened argument]
"""

        messages = [
            SystemMessage(content=SELF_DEBATE_SYSTEM_PROMPT),
            HumanMessage(content=proponent_prompt)
        ]
        proponent_response = agent.invoke(messages)
        proponent_points.append(proponent_response.content)
        transcript.append(f"=== PROPONENT (Round {round_num}) ===\n{proponent_response.content}")

        # OPPONENT round
        opponent_prompt = f"""Continue the self-debate. The proponent argued:

PROPONENT (Round {round_num}):
{proponent_points[-1]}

Now argue as OPPONENT (Round {round_num}). Challenge the proponent's position with your strongest counterarguments.

Format:
## Opponent Position (Round {round_num})
[Your counterargument]

## Key Critiques
- [Critique 1]
- [Critique 2]
...

## Alternative Recommendations
- [Different approach or addition]
"""

        messages = [
            SystemMessage(content=SELF_DEBATE_SYSTEM_PROMPT),
            HumanMessage(content=opponent_prompt)
        ]
        opponent_response = agent.invoke(messages)
        opponent_points.append(opponent_response.content)
        transcript.append(f"\n=== OPPONENT (Round {round_num}) ===\n{opponent_response.content}")

    # SYNTHESIS
    synthesis_prompt = f"""You have completed a {num_rounds}-round self-debate on: "{topic}"

Full debate transcript:
{chr(10).join(transcript)}

Now provide your FINAL SYNTHESIS:
1. What are the strongest points from each side?
2. Where was genuine insight generated through the dialectic?
3. What specific improvements should be made to the document?

Format:
## Synthesis

### Strongest Proponent Points
- [Point 1]
...

### Strongest Opponent Points
- [Point 1]
...

### Novel Insights from Dialectic
- [Insight that emerged from the debate itself]
...

### Final Recommendations for Document
[Specific, actionable content to add]

### Scores
- Novelty Score: [0-1]
- Factuality Score: [0-1]
"""

    messages = [
        SystemMessage(content=SELF_DEBATE_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt)
    ]
    synthesis_response = agent.invoke(messages)
    transcript.append(f"\n=== SELF-DEBATE SYNTHESIS ===\n{synthesis_response.content}")

    # Extract scores
    novelty_score = 0.7
    factuality_score = 0.8

    synthesis = synthesis_response.content
    if "Novelty Score:" in synthesis:
        try:
            score_line = [l for l in synthesis.split('\n') if 'Novelty Score:' in l][0]
            novelty_score = float(score_line.split(':')[1].strip().split()[0])
        except:
            pass

    if "Factuality Score:" in synthesis:
        try:
            score_line = [l for l in synthesis.split('\n') if 'Factuality Score:' in l][0]
            factuality_score = float(score_line.split(':')[1].strip().split()[0])
        except:
            pass

    return {
        "topic": topic,
        "num_rounds": num_rounds,
        "debate_transcript": transcript,
        "proponent_points": proponent_points,
        "opponent_points": opponent_points,
        "synthesis": synthesis_response.content,
        "moderator_decision": synthesis_response.content,  # For compatibility
        "novelty_score": novelty_score,
        "factuality_score": factuality_score,
        "document_content": document_content
    }
