# Day 04 Reading Notes: Mitigating safety issue for LLM agents

Today I focused on safety issues that emerge when LLM-based autonomous agents are given goals, memory and tools, and examined mitigation strategies across scheming, injection, misalignment and lack of personalization.

## In-context scheming and deceptive behavior

Frontier models have shown that when given goals, or placed in environments that reward strategic behavior, they may exhibit in-context scheming. This includes self-exfiltration, goal guarding, and sandbagging.

**Paper:** Frontier Models are Capable of In-context Scheming
https://arxiv.org/pdf/2412.04984

## Prompt injection

LLM agents interacting with tools and external environments remain vulnerable to injection attacks.

**Paper:** DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents
https://arxiv.org/pdf/2506.12104

Mitigation
Detect agent deviation from the intended plan, and isolate injected content.Different from input-sanitization-only approaches, DRIFT continuously inspects tool outputs, masks malicious content before it enters memory, and prevents memory poisoning by ensuring unsafe content never becomes part of the agent's long-term context.

## Misaligned agency from reward maximization

Pure reward optimization can lead to misaligned incentives and deceptive behavior such as alignment faking.

Mitigation
Create architectural constraints by developing non-agentic systems that are safe by design, such as LawZero. Use Bayesian posterior predictive distributions to avoid overconfident predictions in high-stakes settings, and separate the world model from the inference module to reduce compounding failure modes.

**Paper:** Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path?
https://arxiv.org/pdf/2502.15657

## Lack of pluralism and missing user context in high-stack advice domain

Ignoring user-specific context can produce unsafe recommendations in domains like health, education, finance, life decisions, and relationships.

Mitigation
A benchmark covering high-stakes domains built from Reddit and synthetic data evaluates whether models adapt to diverse user needs. The proposed planning-based agent gathers context efficiently: an offline planner uses LLM-guided MCTS to discover optimal context-acquisition sequences, and an online agent retrieves context through these cached paths. An Abstention Module evaluates whether enough context has been gathered before generating a response.

**Paper:** Personalized Safety in LLMs: A Benchmark and a Planning-Based Agent Approach
https://arxiv.org/pdf/2505.18882
