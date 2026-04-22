# Day 09: Agent-Based Behavioral Simulation

Behavioral simulation leverages synthetic agents to stand in for real users to assess ML system. Below are my notes on relevant paper and benchmarks, organized into the three generations of approach that have emerged (hand-crafted/IRL → RL platforms → LLM agents), plus the deployment-bridge and benchmarking work that sits on top.


### RecSim / RecSim NG (Google, 2019 / 2021)

**Code:** https://github.com/google-research/recsim
**Paper:** [arXiv:1909.04847](https://arxiv.org/abs/1909.04847) / [arXiv:2103.08057](https://arxiv.org/abs/2103.08057)

#### Overview

A configurable simulation environment for recommender system research. Provides MDP-style environments with pluggable user models and item models so policies can be trained and benchmarked without live traffic.

#### Method

- User model is a collection of hand-specified latent states (interest vectors, satiation, attention budget) that evolve under a user transition function.
- Item model generates candidate documents with hand-defined feature distributions.
- The environment exposes a standard RL interface (`step`, `reset`) so off-the-shelf agents can be plugged in.
- RecSim NG generalizes to probabilistic-programming semantics (Edward2 / TensorFlow) - user dynamics are expressed as probabilistic programs, allowing differentiable simulation and counterfactual analysis. Supports latent-variable model learning via MCMC and variational inference.

#### Observations

- Useful as a training and benchmarking sandbox. Has been used for slate optimization, long-horizon reward experiments, and fairness studies.
- User models are hand-crafted, not learned from production logs - so the simulated user distribution reflects designer assumptions rather than real behavior.
- No deployment-gating logic, no routing of evaluation units to different fidelity tiers, no notion of evaluation budget.

---

## Generation 2: RL Simulation Platforms with Learned Components (2019–2023)

The next wave replaced hand-crafted user dynamics with models pretrained on real interaction logs, while preserving the RL sandbox structure. These platforms model users as statistical response functions — they can't generate the *why* behind a decision, can't simulate cross-scenario behavior, and can't express intent in natural language. But learned user models from real logs are meaningfully more faithful than hand-crafted ones.

### Virtual Taobao (Shi et al., AAAI 2019)

**Paper:** [arXiv:1805.10000](https://arxiv.org/abs/1805.10000)

#### Overview

A GAN-based user simulator trained on historical Taobao customer behavior data — hundreds of millions of records. The first work to bring real e-commerce scale to simulation.

#### Method

- **GAN-SD (GAN for Simulating Distributions):** Generates customer profiles with an extra distribution constraint (entropy + KL-divergence) to produce diverse customers rather than collapsing to the most frequent type.
- **MAIL (Multi-Agent Adversarial Imitation Learning):** Learns both customer and platform policies jointly via a GAIL-style adversarial objective. A discriminator distinguishes simulated from real interactions; the discrimination signal serves as reward for training both policies simultaneously. This makes the learned customer policy generalizable across different platform strategies, unlike behavior cloning which only works under the historical policy.
- **ANC (Action Norm Constraint):** Penalizes the RL agent when its actions deviate too far from historical action norms, reducing over-fitting to the simulator.
- Customer features: request category, purchase power, user level.

#### Findings

- Virtual Taobao faithfully recovers key properties of the real environment: customer distribution proportions, R2P (Rate of Purchase Page) across categories, and R2P over time.
- MAIL-trained environments generalize over time far better than behavior-cloning environments — after one month, RL policies in the BC environment perform worse than random, while MAIL remains positive.
- RL policies trained in Virtual Taobao achieve >2% improvement in real-world revenue over supervised learning approaches.

#### Observations

- Significant for demonstrating GAN-based simulation at industrial scale.
- Still requires fitting user state generators (GAN-SD) that can amplify distribution mismatch between generated and real user populations.
- No natural language, no cross-scenario behavior, no deployment-gating.

---

### KuaiSim (Zhao et al., NeurIPS 2023)

**Paper:** [arXiv:2309.12645](https://arxiv.org/abs/2309.12645)
**Code:** https://github.com/Applied-Machine-Learning-Lab/KuaiSim

#### Overview

The most comprehensive RL simulation platform for recommender systems to date. Supports three levels of recommendation tasks on the KuaiRand unbiased dataset (27K users, 7.5K items, 1.4M interactions with 12 feedback signals).

#### Method

Three modules simulate the full user-system interaction lifecycle:
- **User Immediate Response Module (UIRM):** Pretrained on log data via binary cross-entropy. Encodes user history with a Transformer, outputs behavioral likelihood for each feedback type (click, like, comment, follow, forward, hate). Includes an item-correlation suppression function that reduces positive behavior for highly correlated items, simulating user demand for diversity.
- **User Leave Module:** Maintains a "temper/patience" factor that decreases based on recommendation satisfaction. When temper drops below threshold, user exits the session. Initial temper, decrease rate, and threshold are adjustable.
- **User Retention Module:** Predicts next-day return probability combining personal retention bias (DNN on user state), response retention bias (proportional to session satisfaction), and global bias. Return time follows a geometric distribution, capped at 10 days.

Three task levels:
- **Request level:** List-wise multi-behavior feedback optimization (click, like, comment, follow, forward, hate)
- **Session level:** Whole-session sequential recommendation under standard MDP formulation
- **Cross-session level:** Retention optimization — predicting and maximizing user return probability

#### Findings

- Outperformed RecSim, RecoGym, Virtual Taobao, and RL4RS on all metrics when reconstructed on the same KuaiRand dataset. AUC 0.7234 vs. best baseline 0.6929 (statistically significant, p < 0.05).
- HAC (Hyper-Action Critic) consistently outperformed other RL methods (TD3, A2C, SA2C, DDPG) on whole-session tasks.
- For retention optimization, RLUR significantly outperformed both CEM and TD3.
- Key advantage over Virtual Taobao: directly samples users from datasets during simulation rather than generating them, avoiding distribution mismatch from fitted user generators.

#### Observations

- The three-level task design (request → session → cross-session) is the most complete formulation I've seen for how simulation should support different optimization horizons.
- Still models users as statistical response functions without natural language or cross-scenario capabilities.
- Demonstrates that log-grounded simulation can meaningfully rank RL algorithms and detect performance differences between methods.

---

### RL4RS (Wang et al., 2021)

**Paper:** [arXiv:2110.11073](https://arxiv.org/abs/2110.11073)

#### Overview

Introduced a real dataset from NetEase Games to bridge the "reality gap" in RL-based recommender research, with anonymized data and a validated simulation environment. Relevant as a benchmark that KuaiSim later superseded.

---

## Generation 3: LLM-Based Agent Simulators (2023–2026)

The paradigm shift. Instead of learning a user response function, give an LLM a profile, a memory, and an action space, and let it *role-play* a user. Nearly every system in this generation adopts the Profile-Memory-Action architecture established by Park et al.'s Generative Agents.

### Generative Agents: Interactive Simulacra of Human Behavior (Park et al., UIST 2023)

**Paper:** [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)

#### Overview

The foundational work for LLM-based behavioral simulation. Created 25 agents in a "Smallville" sandbox, each with a profile, memory stream, reflection mechanism, and planning module.

#### Method

- **Profile:** Name, occupation, relationships, personality traits.
- **Memory stream:** Timestamped observations stored in natural language. Retrieval weighted by recency, importance, and relevance.
- **Reflection:** Periodically generates higher-level insights from accumulated memories ("What have I learned about my neighbor?").
- **Planning:** Generates daily plans, decomposes into hourly actions, reacts to unexpected events.

#### Key results

- Demonstrated emergent social behaviors: agents organized parties, spread information, formed relationships — all from simple role-play prompts.
- Cited 3000+ times. Established the Profile-Memory-Action architecture adopted by nearly every subsequent LLM simulator.
- The architecture was designed for open-ended social simulation, not for rec-sys evaluation — no item interaction, no deployment logic.

---

### Generative Agent Simulations of 1,000 People (Park et al., 2024)

**Paper:** [arXiv:2411.10109](https://arxiv.org/abs/2411.10109) — Stanford, Google DeepMind, UW, Northwestern

#### Overview

Scaled the Generative Agents approach from fictional characters to 1,052 real US residents. The most rigorous test of whether LLM agents can faithfully represent real individuals.

#### Method

- Recruited a stratified sample of 1,052 US residents balanced across 9 demographic factors.
- Each participant completed a 2-hour AI-conducted interview (~6,500 words) covering life stories, values, relationships, daily routines, and social perspectives.
- The interview transcript becomes the agent's memory. When queried, the agent reasons through this transcript to predict how the participant would respond.

#### Findings

- **85% of human test-retest consistency** on the General Social Survey (177 items) — vs. 77% for demographic-only agents and 77% for persona-based agents.
- Similar improvements on Big Five personality inventory and economic behavioral games.
- **Replicated 4 of 5 classic social science experiments** with effect-size correlation of 0.99 (normalized).
- Interview-based agents showed **less demographic bias** — smaller accuracy gaps across gender, race, and political ideology compared to demographic-only agents.

#### Observations

- Strongest evidence that interview-grounded agents can approach human fidelity for survey and experiment simulation.
- The 2-hour interview requirement limits scalability. Scalable diversity generation without per-person data collection remains unsolved.
- Focused on social science experiments, not rec-sys or deployment evaluation.

---

### RecAgent: User Behavior Simulation with Large Language Model based Agents (Wang et al., TOIS 2025)

**Paper:** [arXiv:2306.02552](https://arxiv.org/abs/2306.02552) — Renmin University of China

#### Overview

Applied the Profile-Memory-Action architecture specifically to recommender systems. The canonical LLM user simulation framework for rec-sys.

#### Method

- **Profile module:** ID, name, gender, age, traits (e.g., "compassionate", "ambitious"), career, interests (item categories), plus five behavioral features:
  - *Watcher* — provides feedback and ratings
  - *Explorer* — actively searches for heard-about items
  - *Critic* — demands high standards, criticizes
  - *Chatter* — engages in private conversations, trusts friend recommendations
  - *Poster* — publicly posts and shares on social media
- **Memory module:** Three-tier cognitive memory following Atkinson-Shiffrin theory:
  - *Sensory memory:* Raw observations compressed by LLM, scored for importance
  - *Short-term memory:* Enhanced through repetition; after K enhancements, summarized into long-term insights
  - *Long-term memory:* Forgotten with probability proportional to a power function g(Mᵢ) = 1 - (sᵢ + rᵢ · max(rᵢᵝ, δ)) / 2, where sᵢ and rᵢ are importance and recency scores
- **Action module:** Six behaviors — searching, browsing, clicking, next-page, one-to-one chatting, one-to-many broadcasting
- **Activity distribution:** Pareto-distributed action frequencies matching real-world long-tail patterns

#### Findings

- Behavior sequences judged more human-like than RecSim: 45% vs 33% win rate in human evaluation.
- Reproduced **information cocoon formation** — entropy dropped 8.5% over 50 rounds as MF recommender narrowed diversity.
- Reproduced **conformity behavior** — score distributions converged from uniform (3–8 range) to concentrated around 6–7 as social influence accumulated. Agents with more friends had higher attitude-change probability.
- Two intervention strategies tested: adding random recommendations (Rec-Strategy) and adding diverse-interest friends (Soc-Strategy). Rec-Strategy more effective but reduces user satisfaction.

#### Limitations

- Performance degrades after ~15 rounds as accumulated memory overwhelms the LLM's attention.
- Prompts are brittle across different LLMs (ChatGPT vs GPT-4 require different prompts).
- 10 agents cost ~$0.25/round, scaling roughly linearly.
- Time is discretized into rounds, which limits behavioral realism between rounds.

---

### Agent4Rec: On Generative Agents in Recommendation (Zhang et al., 2024)

**Paper:** [arXiv:2310.10108](https://arxiv.org/abs/2310.10108) — NUS, USTC, Tsinghua

#### Overview

A user simulator with emotion-driven decision-making and post-exit "interviews" where agents explain their satisfaction in natural language.

#### Method

- **Profile module** includes social traits (Activity, Conformity, Diversity) quantified into three tiers from real user distributions, plus natural-language taste descriptions distilled from 25 historical interactions via ChatGPT.
- **Memory module** splits into factual memory (logged actions) and emotional memory (fatigue, satisfaction). Includes emotion-driven self-reflection after a pre-defined number of actions.
- **Action module** separates taste-driven actions (view/ignore, rate, generate feelings) from emotion-driven actions (continue/exit based on Chain-of-Thought reasoning about satisfaction and fatigue). Post-exit interviews capture ratings and explanations.
- Tested on MovieLens-1M, Steam, and Amazon-Book with collaborative filtering algorithms (Random, Most Popular, MF, LightGCN, MultVAE).

#### Findings

- Agents correctly identified liked/disliked items with ~65% accuracy and ~75% recall.
- LightGCN consistently outperformed other algorithms in agent satisfaction metrics — matching known real-world performance rankings.
- Successfully replicated **filter bubble effects**: content diversity decreased over simulation iterations as MF recommender was retrained on agent feedback.
- Used **DirectLiNGAM** for causal discovery: movie quality → ratings, popularity → exposure → views → popularity bias loop.
- Page-by-page feedback enhancement: retraining recommenders on agent-*viewed* movies improved both offline metrics and satisfaction; training on *unviewed* movies degraded experience.

#### Notable weakness

- Precision dropped sharply when the proportion of liked items decreased, suggesting LLMs have a bias toward selecting a fixed number of items regardless of preference distribution — a "positivity bias" consistent with OmniBehavior's hyper-activity finding.
- The 'diversity' social trait showed minimal behavioral differentiation, possibly due to overlapping movie categories in MovieLens.

---

### SimUSER: Simulating User Behavior with LLMs for Recommender System Evaluation (Bougie & Watanabe, 2025)

**Paper:** [arXiv:2504.12722](https://arxiv.org/abs/2504.12722) — Woven by Toyota

#### Overview

An agent framework serving as believable, cost-effective human proxies for RS evaluation. Distinguishes itself from RecAgent through persona matching, knowledge-graph memory, and visual perception.

#### Method

- **Phase 1 — Persona matching via consistency check:** Uses LLM semantic reasoning to extract Big Five personality traits and consistent personas from historical data, rather than assigning random profiles. Each persona encompasses age, personality facets (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism on 1–3 scale), and occupation.
- **Phase 2 — Agent architecture:**
  - *Persona module:* Matched persona drives all behavior
  - *Memory module:* Episodic memory + knowledge-graph memory. The KG represents user-item relationships, enabling graph-based retrieval for action planning
  - *Perception module:* Integrates visual signals (thumbnails) into agent reasoning — the first LLM simulator to model how visual cues affect click behavior
  - *Brain module:* Translates retrieved evidences and KG paths into actions (click, exit). Includes self-reflection to synthesize memories into higher-level inferences
- **Multi-round preference elicitation:** Agents refine preferences over multiple interactions via causal action refinement leveraging retrieved evidence

#### Findings

- Closer alignment with real humans than RecAgent at both micro (individual action) and macro (population distribution) levels.
- Explored effects of thumbnails on click rates, exposure effect, and review impact on engagement — demonstrating the visual perception module adds meaningful signal.
- Applied to **offline A/B tests that led to improved real-world user engagement** — one of very few papers demonstrating sim-to-real transfer.
- Refined RS parameters via offline A/B results, with improvements validated in production.

#### Observations

- The persona-matching approach (inferring personality from behavior rather than assigning it) is a meaningful upgrade over random profile generation. Addresses the "chicken and egg" problem differently from RecAgent.
- Knowledge-graph memory is a promising alternative to pure episodic retrieval for structured domain knowledge about item relationships.
- The sim-to-real transfer claim makes this one of the most practically relevant papers in the space.

---

### Shop-R1: Realistic User Behavior Simulation for E-Commerce (Lu et al., 2025)

#### Overview

Trains an LLM-based agent to simulate realistic shopping behavior (browse, click, cart, purchase) on e-commerce sites. The target is high behavioral fidelity against logged sessions.

#### Method

- Two-stage training: supervised fine-tuning on expert trajectories, then GRPO with a composite reward that mixes task completion, format correctness, and step-level semantic plausibility.
- Action space is HTML-grounded: the agent emits clicks and form inputs on a real page state.
- Evaluated on exact-action-match and top-k action agreement against held-out user sessions.

#### Findings

- SFT does the bulk of the lift: exact-match jumps from roughly 0 to 23 percent before RL is applied.
- The headline RL contribution appears sensitive to compute scale and reward design — abbreviated GRPO runs on a single GPU do not reliably improve over the SFT checkpoint.
- The paper does not address how to *use* a high-fidelity simulator to decide whether a candidate ranking/retrieval/prompt change is safe to deploy. It ends at "the simulator is realistic."

---

### BASES: Large-scale Web Search User Simulation (Wang et al., 2024)

**Paper:** [arXiv:2402.17505](https://arxiv.org/abs/2402.17505) — Renmin University of China, Baidu Inc.

#### Overview

Extended the RecAgent paradigm to web search behavior. Demonstrated that LLM agents can simulate query reformulation, click patterns, and browsing behavior at scale. Relevant as evidence that the Profile-Memory-Action architecture generalizes beyond recommendation to search.

---

### US Patent 12513102 B2: Training Language Models on User's Historical Social Media Interactions

**Publication:** US12513102B2

#### Overview

Fine-tunes a language model on a named user's past social media activity so the model can generate responses in that user's voice.

#### Method

- Collect a user's historical posts, comments, and likes.
- Fine-tune (or prompt-condition) an LM to produce text stylistically consistent with the user's history.
- Optionally moderate before posting.

#### Observations

- The simulated entity is an *individual named user*, not a behavioral archetype sampled from a population distribution.
- The target surface is social text, not ML-driven product surfaces (ranking, search, checkout).
- No regression detection, CI/CD integration, or simulation routing.

---

## Benchmarking Realism: How Good Are These Simulators, Really?

The most important question isn't "can we build simulators?" — it's "can we *trust* them?" A series of 2025-2026 benchmarks have started providing honest answers.

### [OmniBehavior: Benchmarking LLMs on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces (Chen et al., 2026)](https://arxiv.org/abs/2604.08362)

**Code:** https://github.com/icip-cas/OmniBehavior
**Site:** https://OmniBehavior.github.io

#### Overview

The first benchmark that pushes LLM user simulation past single-scenario, single-session settings. Built from authentic Kuaishou traces rather than synthetic prompts, and explicitly tests whether a simulator can carry context across scenarios and over long horizons.

#### Method

- 200 real Kuaishou users × ~8,143 actions each × 90 days × 5 scenarios (short video, live, ads, e-commerce, search) × 22 distinct action types.
- Users are not sampled at random — a 4-axis feature vector (demographics, activity level, interest distribution, scenario preference) is clustered via k-means and the nearest-to-centroid user is taken as each cluster's representative. This is the rebuttal to "N=200 is small."
- 6,000 evaluation tasks with enforced balance across time, scenario, and value.
- Evaluation taxonomy: F1 on binary actions (like, share, click); NMAE normalized by video duration on continuous actions (avoiding the bias of raw MAE toward longer items); LLM-as-judge on textual actions (intent fidelity, persona mimicry, knowledge boundary, semantic alignment).
- Benchmarks 11 frontier models: Claude 4.5 (Opus/Sonnet/Haiku), Claude 4, Gemini 3 Flash, GPT-5.2, GPT-4o, GLM-4.7, DeepSeek-V3, Kimi-K2, Qwen3-235B.

#### Findings

- Best model (Claude-Opus-4.5) scores only 44.55 overall; most cluster in the 32–41 band. No model exceeds 40% F1 on binary actions.
- Extending the context window does not close the gap: 32k → 128k gives inconsistent gains. Off-the-shelf RAG and summarization memory yield only modest improvements.
- Three quantified failure modes:
  - **Hyper-activity.** Real users' positive-interaction rate is under 10%. LLM simulators (Gemini-3-Flash, Qwen3-235B) produce 40–60%. A 4–6× over-engagement that breaks any downstream churn, rec-sys, or ad-targeting calibration.
  - **Persona homogenization.** Real users' behavior vectors separate cleanly in embedding space (intra/inter-user distance ratio ≈ 0.29); LLM-simulated populations collapse into an overlapping cloud (ratio ≈ 0.7–0.87). Long-tail individuality is lost.
  - **Utopian bias.** In customer-service dialogs, real users express strong negative sentiment; LLMs concentrate near neutral-positive. Politeness markers, hedging, and face-saving are 2–3× inflated vs. real. The paper attributes this to alignment training.
- Long-horizon matters: 81.8% of conversion chains span more than one scenario, and over 60% of decisions use cues from more than 3 days prior. Session-scoped benchmarks are not representative of realistic behavior.
- Interest drift: real users have a daily Jaccard drift of 0.63; the LoCoMo synthetic baseline sits at 0.17. Synthetic simulators are roughly 4× too stable over time.
- The paper is explicitly negative-results - it diagnoses the three biases but proposes no fix.
- Closed-source models generally lead, but open-source GLM-4.7 ranked second overall (41.46), outperforming several closed-source models. DeepSeek-V3 beat Claude-Opus-4.5 on e-commerce binary behavior.

#### Why it matters for evaluation

- Gives a ready benchmark for the "is my user simulator actually realistic?" question, independent of any downstream evaluation harness.
- The three failure modes (hyper-activity, homogenization, Utopian bias) are specifically the modes that break deployment-gating - if simulated users are more engaged and more positive than real ones, a candidate model will look safer under simulation than it actually is.
- The NMAE-normalized-by-duration convention is the cleanest rigor upgrade in the paper: raw MAE on dwell/watch-duration metrics biases toward longer items, and almost every behavioral-sim paper I've read uses the raw form.

---

### MirrorBench: Evaluating Conversational User-Proxy Agents for Human-Likeness (Hathidara et al., SAP Labs, 2026)

**Paper:** [arXiv:2601.08118](https://arxiv.org/abs/2601.08118)

#### Overview

Focuses specifically on human-likeness of user proxy agents in conversational settings. Key insight: decouple human-likeness assessment from task success — a proxy can be excellent at completing tasks but still sound distinctly non-human.

#### Method

- Dual metric families:
  - **Lexical diversity metrics** (MATTR, Yule's K, HD-D) z-score normalized against human baselines from the same dataset
  - **LLM-judge realism metrics** (GTEval relative scoring, Pairwise Indistinguishability, Rubric-and-Reason)
- Calibration controls: Human-Human (HH) and Proxy-Proxy (PP) baselines for affine rescaling of judge scores
- Tested across 5 proxy LLMs (GPT-4o, GPT-5, GPT-OSS-120B, Claude-4-Sonnet, Gemini-2.5-Pro) and 4 datasets (ChatbotArena, ClariQ, OASST1, QULAC)

#### Findings

- Gemini-2.5-Pro and Claude-4-Sonnet consistently most human-like by judge metrics.
- **Realism-diversity tension:** Proxies rated most realistic by judges sometimes *exceeded* human lexical diversity (Claude-4-Sonnet on ClariQ) or fell below it (all proxies on clarification-centric QULAC). High judge-based realism ≠ human-level lexical diversity; both metric families are needed.
- Naive "act-as-a-user" prompting produces outputs that are verbose, overly cooperative, and unnaturally polite — **confirming OmniBehavior's utopian bias from a completely different angle** (conversational vs. behavioral).
- Judge choice significantly affects scores and rankings; using multiple judges with calibration controls is essential. Claude-4-Sonnet was the most conservative yet stable judge.
- Human-judge correlation: Spearman ρ = 0.697 for GTEval, 0.608 for PI (using Claude-4-Sonnet judge).

---

### SimulatorArena: Are User Simulators Reliable Proxies? (Dou et al., Georgia Tech + Microsoft, 2025)

**Paper:** [arXiv:2510.05444](https://arxiv.org/abs/2510.05444)

#### Overview

First systematic benchmark asking whether user simulators are reliable proxies for multi-turn evaluation of AI assistants. 909 annotated human-LLM conversations across math tutoring and document creation.

#### Key findings

- **User profile-based simulators** dramatically improve alignment with human judgments: Spearman ρ improved from 0.61 (zero-shot CoT) to **0.77** for math tutoring and from 0.55 to **0.70** for document creation. Statistically significant (p < 0.01).
- Profiles are task-specific: math tutoring benefits most from *interaction style* attributes; document creation needs *full profiles* (preferences + writing style + interaction style).
- Profile-based simulators operate at **<3% of human evaluation cost**.
- Remaining gaps: simulators struggle with natural LaTeX notation, grammar mistakes, and sentence fragments that real users produce.
- Top simulator LLMs: GPT-4o best for math tutoring (ρ = 0.77), Gemini 2.0 Flash for document creation (ρ = 0.74).
- Benchmarked 18 LLM assistants including GPT-5, Claude 4.1 Opus, Gemini 2.5 Pro. GPT-5 led overall.

---

### Adjacent Benchmarks

| Paper | What is being simulated | How they evaluate simulation quality | Details |
|---|---|---|---|
| **AgentRecBench** ([arXiv:2505.19623](https://arxiv.org/abs/2505.19623)) | Interactive textual recommendation environments for agentic/personalized recommendation. | Measures downstream recommendation performance across three scenarios — classic, evolving-interest, cold-start — inside an interactive simulator, rather than the simulator itself. | Measures whether agents perform well *in* the simulated environment, not whether the simulator matches human decision dynamics. |
| **Consistently Simulating Human Personas with Multi-Turn RL** | Persona-conditioned dialogue users across therapy, education, and social-chat domains. | Three automatic consistency metrics — prompt-to-line, line-to-line, and Q&A consistency — validated against human annotations. | Strong for persona consistency, weak for user-choice fidelity in recommender settings. |
| **HORIZON** | In-the-wild user behavior modeling for sequential recommendation. | Long-term temporal generalization, cross-domain transfer, and unseen-user generalization, with splits designed to avoid temporal leakage. | User-*modeling* benchmark rather than simulator benchmark; measures generalization under harder splits. |
| **LLM-Powered User Simulator for Recommender System** | Explicit item-engagement behavior used as training signal for RL-based recommenders. | Ensemble of logical and statistical simulation. | Focused on effectiveness and stability of generated *training data*, not on detecting algorithm degradation. |

---

## The Deployment Bridge: From "Realistic Simulator" to "Ship/No-Ship Decision"

Most simulator papers end with "our simulator is realistic." Very few address the next question: how do you use a simulator to make deployment decisions?

### AgentA/B: Automated and Scalable Web A/B Testing with Interactive LLM Agents (Lu et al., 2025)

**Paper:** [arXiv:2504.09723](https://arxiv.org/abs/2504.09723) — Northeastern University, Amazon

#### Overview

The first end-to-end system for LLM-agent-based A/B testing on live websites.

#### Method

- **Agent generation:** LLM generates persona-driven agents with demographic/behavioral diversity.
- **Traffic allocation:** Agents split into control/treatment groups with statistical balance checks.
- **Interaction loop:** Agents interact with *live web pages* via Selenium/ChromeDriver. Environment parsing extracts key web elements into structured JSON; agents emit actions (Search, Click Product, Click Filter, Purchase, Stop).
- **Post-test analysis:** Summary statistics, behavioral analysis, demographic stratification.

#### Key results

Case study: 1,000 agents tested a filter-panel redesign on Amazon.com.
- Treatment group (reduced filter list) showed statistically significant higher purchase completion: χ²(1)=5.51, p=0.03, approximately 2-3% higher conversion.
- Subgroup analysis: older users (>55) showed the largest spending increase ($40.24 → $79.07); younger users (≤35) showed decreased spending — suggesting reduced filters help with choice overload for older users.
- **Crucially, results aligned directionally with a parallel 2M-user human A/B test.**

#### Limitations

- Agents more goal-directed than humans (6.05 avg actions vs 15.96), with less exploratory behavior.
- Works for intention-driven UX evaluation, not discovery of emergent browsing patterns.
- Does not model affective or meta-cognitive signals.

---

### RecoWorld: Building Simulated Environments for Agentic Recommender Systems (Meta, 2025)

**Paper:** [arXiv:2509.10397](https://arxiv.org/abs/2509.10397) — Meta Platforms (WWW Companion 2026)

#### Overview

A *blueprint* (not yet empirically validated) for simulated environments tailored to instruction-following agentic recommender systems.

#### Key design ideas

- **Dual-view architecture:** Simulated user ↔ Agentic recommender in multi-turn loops.
- **Reflective instructions:** When users sense disengagement, they generate natural language feedback rather than just leaving — enabling "user instructs, recommender responds" paradigm.
- **Three content representation modes:** Text-based (flexible, loses non-textual nuances), multimodal (MLLMs — richer but expensive), and semantic ID (compact, requires co-training).
- **Dynamic memory:** Interaction-wise (fine-grained actions) + session-wise (summarized session representations with mindset shifts).
- **Long-term retention metrics** as reward signals, explicitly orthogonal to NDCG. The paper proposes a 2×2 analysis:
  - High NDCG + High Retention = strong exploitation (relevant and engaging)
  - High NDCG + Low Retention = suboptimal (repetitive, leading to disengagement)
  - Low NDCG + High Retention = effective exploration (novel content, long-term retention)
- **Multi-agent simulation** for population-level dynamics and content diffusion.

#### Observations

- Proposes but does not validate. Still significant as Meta's public statement of strategic direction.
- The framing of retention metrics as orthogonal to accuracy metrics is a useful conceptual contribution.
- Strong interest from Google and Kuaishou RecSys teams mentioned in the paper.

---

## What's Missing Across This Landscape

The IRL and simulator papers answer *how to generate realistic synthetic behavior*; OmniBehavior and the benchmarks above answer *how to measure whether a simulator is realistic*. None of them answer *given a realistic simulator, how to route simulation budget so a CI/CD pipeline can make a PASS/HOLD/FAIL call on a candidate model change without spending unbounded compute* — the control-plane problem of discrepancy detection between offline metrics and behavioral signals, tier routing, budget-aware stopping, and component-combination scoring. That gap is the most interesting open problem in the space.

### Specific Open Problems

1. **The Calibration Problem.** If your simulator is systematically 4× too positive, can you calibrate it? Affine correction works for aggregates but erases granularity. Better approaches: contrastive fine-tuning against known failure cases, explicit negative-sentiment injection, de-alignment or controlled personality temperature adjustment.

2. **Population Diversity Generation.** Current approaches either sample from real profiles (data-dependent) or generate via LLM (collapses to the mean). The 1,000 People approach (interview-based) achieves better diversity but requires expensive per-person data collection. Scalable diversity generation without data dependency is unsolved.

3. **Cross-Scenario Coherence.** 81.8% of real conversion chains span multiple scenarios. No current simulator convincingly handles cross-scenario coherent behavior (a user annoyed by ads in live-streaming becoming more price-sensitive in e-commerce). Requires shared latent state across modalities that current Profile-Memory-Action architectures don't enforce.

4. **The Evaluation Harness Gap.** The entire field has focused on "make the simulator realistic" and barely touched "make the simulator useful for deployment decisions." We need stopping rules, disagreement detection, tiered evaluation routing, and safety guarantees.

5. **Sim-to-Real Transfer Guarantees.** AgentA/B and SimUSER showed directional alignment with real A/B tests, but we have no formal theory of when simulation results transfer. Analogous to the sim-to-real gap in robotics, but less studied for behavioral simulation.

---

## What Consensus Looks Like

- **Profile-based simulation >> zero-shot.** SimulatorArena: ρ 0.61 → 0.77. 1,000 People: 85% vs 77%. The data is consistent.
- **Hyper-activity, homogenization, and utopian bias are structural.** Confirmed independently by OmniBehavior (behavioral traces), MirrorBench (conversational), and Agent4Rec (positivity bias in item selection). These are artifacts of RLHF training, not fixable by prompt engineering alone.
- **Learned response models + LLM reasoning is likely the right hybrid.** KuaiSim's statistical response models are more calibrated; LLM agents can reason about intent. Combining them is the obvious next step that nobody has convincingly executed yet.
- **Context window scaling doesn't fix the hard problems.** OmniBehavior (32K → 128K inconsistent), RecAgent (memory degrades after ~15 rounds). The bottleneck is not input capacity but reasoning over long behavioral sequences.

---

## Dataset Bench

- OmniBehavior demo traces: https://github.com/icip-cas/OmniBehavior/blob/main/data/demo.json
