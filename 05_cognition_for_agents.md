# Day 05 Reading Notes: Cognition for Agents

Drawing on psychological frameworks such as Theory of Mind and metacognition, recent agentic systems applies these framework to improve social reasoning and human interaction. Below are my notes.

## Theory of Mind

**Paper:** MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems
https://arxiv.org/abs/2505.18943

Part of human social intelligence is to infer about others' beliefs, intents, desires, emotions, knowledge, etc. The paper designs a system to emulate human-like social reasoning through multiple components: a Theory-of-Mind agent that hypothesizes about other users’ mental states, a moral agent that uses social norms and ethics as constraints, and a response agent that generates contextually appropriate outputs while maintaining alignment with the other user’s intent. The authors report significant gains compared to Chain-of-Thought and earlier Theory-of-Mind multi-agent methods on the Sandbox Simulation benchmark, which tests goal-oriented social interaction as well as open-ended interaction and social reasoning. A limitation is that real-world social interactions are multimodal, group dynamics are more complex, and they require long-term relationship building.

## Metacognition

**Paper:** Metacognition is all you need? Using Introspection in Generative Agents to Improve Goal-directed Behavior
https://arxiv.org/abs/2401.10910

This 2024 paper introduces a metacognition module for agents. Unlike reflection, which looks back at past experiences to derive insights, metacognition focuses on introspection and strategizing by considering whether progress is being made or if a new strategy is needed. This is akin to System 2 thinking in Daniel Kahneman’s Thinking, Fast and Slow. One task studied in the paper is surviving a zombie apocalypse, where initially goal-less agents, through encounters with zombies, begin to infer that they need to hide in zombie-free zones to survive. Given a baseline survival rate of 27%, the authors find a 33% improvement in the overall score. Limitations include agents starting from a blank state with relatively simple memory, using LMs rather than VLMs, and treating metacognition as a separate module instead of integrating it directly into the LLM.

**Paper:** Truly Self-Improving Agents Require Intrinsic Metacognitive Learning
https://arxiv.org/abs/2506.05109

This 2025 paper tackles how to generalize and scale self-improving agents. By defining the learning process as leveraging goals, strategies, and capabilities to plan and continuously evaluate progress, the authors identify three metacognitive components:
knowledge: self-assessment and learning strategy development
planning: deciding what and how to learn
evaluation: reflecting on learning experiences to improve future learning

Beyond identifying open-world deployment as a challenge, the authors suggest that shared metacognitive responsibility between humans and agents may be necessary.
