# Day 02 Reading Notes: World Models

World models underpin embodied agents, physical intelligence, and creative AI, especially in areas like video generation and game engines. By constructing a compressed, simulated environment, they allow agents to rehearse, plan, and self-improve, narrowing the gap between virtual and physical worlds. As real-world interaction data is often hard to collect, world models offer a scalable way to generate synthetic trajectories. For Day 2, I focus on the main components and approaches.

## World Models (Ha & Schmidhuber, 2018)

**Paper:** [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

This concept was formalized in the 2018 paper “World Models” by David Ha and Jürgen Schmidhuber. The system consists of three parts: a vision module for compressing observations into latent states, a memory module that models temporal transitions, and a controller that outputs actions using only these latent states. This demonstrated that an agent could learn an internal simulation and train a policy entirely inside that learned latent world.

## DreamerV3/V4 (Hafner et al.)

**Summary:** [https://vitalab.github.io/article/2023/01/19/DreamerV3.html](https://vitalab.github.io/article/2023/01/19/DreamerV3.html)

DreamerV3 is a recent model-based RL method developed by Danijar Hafner and colleagues at DeepMind in 2023. It uses a recurrent state-space world model (RSSM) paired with an actor-critic. The system maintains a posterior latent state computed from the previous state and real observation for training the world model, and a prior latent state predicted from the previous state and action for policy learning in “dreaming” mode. Also DreamerV4 [https://arxiv.org/pdf/2509.24527] (2025) upgraded RSSM to an efficient transformer and used Preference Optimization as Probabilistic Inference (PMPO), which focus on the sign of the advantage rather than the magnitude, in addition to KL as objective.

## WALL-E 2.0: Neurosymbolic World Models

**Paper:** [https://arxiv.org/pdf/2504.15785](https://arxiv.org/pdf/2504.15785)
**Code:** [https://github.com/elated-sawyer/WALL-E](https://github.com/elated-sawyer/WALL-E)

A limitation of previous world models is that their latent representations lack explicit causal structure. WALL-E 2.0 explores neurosymbolic augmentation to address this. In a POMDP environment where the goal is to mine diamonds on Mars, the system extracts symbolic knowledge from agent exploration: deterministic action rules that capture constraints, knowledge graphs defining prerequisites, and scene graphs providing global information beyond the agent’s local observation. These elements help correct LLM-generated plans. This method is effective but not meant for cross-domain transfer.
