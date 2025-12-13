# Day 05 Reading Notes: Cognition for Agents

Drawing on psychological frameworks such as Theory of Mind and metacognition, recent work explores how incorporating cognitive architectures into multi-agent systems can improve social reasoning, introspection, and goal-directed behavior. Below are my notes on papers applying these concepts to enhance agent capabilities.

## Theory of Mind
Part of human social intelligence is to infer about others' beliefs, intents, desires, emotions, knowledge, etc. and some multi-agent systems are leveraging this Theory of Mind approach to enhance the social reasoning capability of LLM

**Paper:** MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems
https://arxiv.org/abs/2505.18943
the paper designed agentic systems to emulate human-like social reasoning through a Theory-of-mind agent that hypoethsis about other users' mental state, a moral agent that use social norm and ethics as constraint, and a response agent to generate contextually appropriate output while maintaining alignment with the other users' intent. they found significant gain compared to Chain-of-thought and other earlier theory-of-mind multi-agent methods, in Sandbox Simulation benchmark which test goal-oriented social interaction, as well as open-ended interaction/social reasoning. The limitation is that real-world social interaction hare multi-modal , group dynamics are more complex and it require long-term relationship building.

## Metacognition

**Paper:** Metacognition is all you need? Using Introspection in Generative Agents to Improve Goal-directed Behavior
https://arxiv.org/abs/2401.10910
This 2024 paper introduces metacognition module for agents. unlike reflection which is looking back at past experience and derive singiths, metacognition go down the path of introspection and strategizing by considering if they are making progress or if a new strategy is needed. This is akin to the system 2 thinking Dan Kahneman's thinking fast and slow. One of the tasks the authors asks the agents do is to surviving zombie apolycypse where goal-less agents through encountering zombies started to figure out they need to hide in zombie-free zone to survive. given a baseline of 27% survival rate , they found 33% improvement in general overall score. limitation includes agents starting from a blank state with relative simple memory, and using lm rather than vlm,treating metacognition as a separate module instead of directly using it directly in a llm.

**Paper:** Truly Self-Improving Agents Require Intrinsic Metacognitive Learning
https://arxiv.org/abs/2506.05109
This 2025 paper tackles how to generalize and scale self-improving agents. by definiing the learning process as leveraging goals, strategies and capabilities to plan and continuously evaluate progress, the authors considered 3 metacognitive components:
knowledge: self-assessment and learning strategy developpment
planning: decide what and how to learn
evaluation: reflecting on learning experiences to improve future learning
Beside finding open-world deployment a challenge, they suggested shared metacognitive responsibility might be needed between human and agents.
