# LLM and Agentic Evaluation

---

## Table of Contents

1. [Core Evaluation Areas](#core-evaluation-areas)
2. [Common Evaluation Packages](#common-evaluation-packages)
3. [Deep Dive: Evaluation Frameworks](#deep-dive-evaluation-frameworks)
4. [LLM-as-a-Judge: Advanced Topics](#llm-as-a-judge-advanced-topics)
5. [Fine-Tuning LLM Raters](#fine-tuning-llm-raters)
6. [Preference Learning and Reward Modeling](#preference-learning-and-reward-modeling)
7. [Agentic Evaluation Research](#agentic-evaluation-research)
8. [Multi-Turn/Multi-Hop Evaluation as Credit Assignment](#multi-turnmulti-hop-evaluation-as-credit-assignment)
9. [Production Deployment Realities](#production-deployment-realities) _(NEW - from debate)_
10. [LLM-as-Judge: Bias Analysis and Mitigation](#llm-as-judge-bias-analysis-and-mitigation) _(NEW - from debate)_
11. [Statistical Foundations for Agent Evaluation](#statistical-foundations-for-agent-evaluation) _(NEW - from debate)_
12. [Tiered Evaluation Strategy Framework](#tiered-evaluation-strategy-framework) _(NEW - from debate)_
13. [Multimodal and Audio Evaluation](#multimodal-and-audio-evaluation)
14. [Long-Context Evaluation](#long-context-evaluation)
15. [OpenAI Evaluation Flywheel](#openai-evaluation-flywheel)
16. [Other Info](#whats-missing)

---

## Core Evaluation Areas

### LLM Evaluation

- Accuracy, coherence, relevance
- Factuality and hallucination detection
- Safety, toxicity, bias
- Task-specific performance (QA, summarization, code generation)

### Agentic Evaluation

- Tool use accuracy and effectiveness
- Multi-step reasoning and planning
- Goal completion rate
- Error recovery and robustness

---

## Common Evaluation Packages

| Package                   | Purpose           | Key Features                                               |
| ------------------------- | ----------------- | ---------------------------------------------------------- |
| **ragas**                 | RAG evaluation    | Context precision/recall, answer relevance, faithfulness   |
| **deepeval**              | LLM testing       | Unit tests for LLMs, hallucination detection, bias metrics |
| **langsmith**             | LangChain tracing | Monitoring, debugging, dataset curation                    |
| **promptfoo**             | Prompt testing    | A/B testing prompts, regression detection                  |
| **phoenix**               | Observability     | LLM traces, embeddings analysis, drift detection           |
| **autoevals**             | General evals     | Model-graded evaluations (Braintrust)                      |
| **HELM**                  | Benchmarking      | Stanford's holistic evaluation framework                   |
| **lm-evaluation-harness** | Benchmarks        | EleutherAI's unified testing (HuggingFace)                 |

**Package Overview:**

- **ragas** - Measures RAG pipeline quality (retrieval relevance, answer faithfulness)
- **deepeval** - Unit testing framework with G-Eval, toxicity, bias metrics
- **langsmith** - Full lifecycle observability for LangChain apps
- **promptfoo** - CI/CD for prompts with regression testing
- **phoenix** - Real-time monitoring and embedding visualization

---

## Deep Dive: Evaluation Frameworks

### G-Eval: NLG Evaluation via GPT

**Core Innovation**: Uses LLMs as evaluators with chain-of-thought reasoning.

**Method**:

1. Generate evaluation criteria and steps via prompt
2. LLM produces intermediate reasoning
3. Outputs probability-weighted scores (1-5 scale)
4. Aggregates token probabilities for final score

**Strengths**:

- High correlation with human judgments (Spearman 0.514 on **SummEval**)
- Outperforms ROUGE/BERTScore
- Handles nuanced criteria (coherence, consistency, fluency, relevance)

**Weaknesses**:

- Expensive (API costs)
- Slower than metric-based approaches
- Prompt-sensitive
- Potential bias from evaluator LLM

---

### Coherence Metric: Calculation Methods

Coherence is a critical NLG evaluation dimension measuring whether text is well-structured and logically organized. Multiple approaches exist for calculating coherence:

#### G-Eval Coherence (LLM-based)

**Definition** (from [G-Eval paper](https://arxiv.org/abs/2303.16634)):

> "The collective quality of all sentences. The summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic."

**Scoring Scale**: 1-5 (lowest to highest)

**Auto-Generated Evaluation Steps** (Chain-of-Thought):

1. Read the source text and identify main topic and key points
2. Read the generated text and compare. Check if it covers main topic/key points and presents them in clear, logical order
3. Assign coherence score (1-5) based on evaluation criteria

**Score Calculation**:

```
Final Score = Σ(p(token_i) × score_i) / Σ p(token_i)

Where:
- token_i ∈ {1, 2, 3, 4, 5}
- p(token_i) = probability of LLM outputting that score token
```

For models without token probabilities (e.g., GPT-4), sample 20 times with temperature=1 to estimate probabilities.

**Performance**: Spearman correlation of 0.514 with human judgments on SummEval benchmark.

#### UniEval Coherence (Boolean QA-based)

**Approach** ([UniEval, Zhong et al. 2022](https://arxiv.org/abs/2210.07197)): Frames evaluation as Boolean question answering.

**Method**:

```
Question: "Is this a coherent summary of the document?"
Answer probabilities: P(Yes), P(No)
Score = P(Yes) / (P(Yes) + P(No))
```

**Advantages**:

- Unified framework across multiple dimensions (coherence, consistency, fluency, relevance)
- More stable than direct 1-5 scoring (avoids middle-score bias)
- Can evaluate without references

#### BARTScore Coherence

**Approach** ([BARTScore, Yuan et al. 2021](https://arxiv.org/abs/2106.11520)): Measures generation likelihood.

**Method**:

```
Coherence ≈ log P(target | source) using BART encoder-decoder

Higher likelihood = more coherent/natural text
```

**Limitation**: Surface-level metrics (ROUGE, BLEU) perform poorly on coherence perturbation detection because they focus on n-gram overlap, not logical structure.

#### Comparison of Coherence Methods

| Method         | Type       | Human Correlation                 | Cost   | Speed   |
| -------------- | ---------- | --------------------------------- | ------ | ------- |
| G-Eval (GPT-4) | LLM-based  | High (0.514 Spearman on SummEval) | High   | Slow    |
| UniEval        | Boolean QA | Moderate-High                     | Medium | Medium  |
| BARTScore      | Likelihood | Moderate                          | Low    | Fast    |
| ROUGE/BLEU     | N-gram     | Low                               | Free   | Instant |

**Recommendation**: Use G-Eval for high-stakes evaluation, UniEval for balanced cost/quality, BARTScore for rapid iteration.

---

### ragas (Retrieval Augmented Generation Assessment)

**Architecture**: Modular evaluation for RAG pipelines - separates retrieval from generation quality.

**Core Metrics**:

- **Context Precision**: Relevant chunks in top-k (ground truth required)
- **Context Recall**: Coverage of ground truth in retrieved context
- **Faithfulness**: Claims in answer grounded in context (entailment checking)
- **Answer Relevance**: Semantic similarity to question (uses embeddings)

**Technical Details**:

- Uses sentence transformers for embeddings
- NLI models for faithfulness checking
- Can operate without ground truth (faithfulness/relevance only)

**Limitations**:

- Requires structured QA format
- Limited agentic support
- Assumes single-turn interactions
- No tool-use evaluation

---

### deepeval: Unit Testing for LLMs

**Design Philosophy**: Pytest-like assertions for LLM outputs.

**Evaluation Types**:

- **G-Eval Integration**: Customizable LLM-as-judge with CoT
- **Hallucination**: Contradiction detection via NLI + factual consistency
- **Toxicity**: Perspective API + custom classifiers
- **Bias**: Gender/race/religion detection across dimensions
- **Answer Relevance**: Semantic + keyword matching
- **Contextual Metrics**: Precision/recall/relevancy for RAG

**Code Pattern**:

```python
from deepeval import assert_test
from deepeval.metrics import GEval, HallucinationMetric

test_case = LLMTestCase(input="...", actual_output="...", context=["..."])
metric = GEval(criteria="accuracy", evaluation_steps=[...])
assert_test(test_case, [metric])
```

**Advantages**:

- CI/CD integration
- Synthetic dataset generation
- Confidence intervals
- Traces linked to metrics

**Gaps**:

- Limited multi-turn support
- Tool-use evaluation requires custom metrics
- No cost tracking

---

### LangSmith: Production Observability

**Core Functions**:

- **Tracing**: Full execution tree (LLM calls, tool invocations, latency)
- **Datasets**: Version-controlled test sets with annotations
- **Evaluators**: Online (production) + offline (batch) evaluation
- **Monitoring**: Cost, latency, error rates, user feedback

**Evaluation Engine**:

- Custom Python evaluators or LLM-as-judge
- Comparison mode (A/B test prompts/models)
- Human annotation workflows
- Metrics: exact match, semantic similarity, custom scoring

**Production Features**:

- Feedback collection
- Tag-based filtering
- Anomaly detection
- Trace search

**Limitations**:

- Tied to LangChain ecosystem (though supports generic traces via API)
- Expensive at scale
- Limited agentic benchmarks

---

### promptfoo: Systematic Prompt Engineering

**Focus**: Red-teaming and regression testing for prompts.

**Workflow**:

1. Define test cases (inputs + expected outputs/assertions)
2. Configure models/providers/prompts as variables
3. Run evaluations (similarity, factuality, custom functions)
4. Compare outputs in matrix view

**Assertions**:

- `contains`, `not-contains`, `regex`
- `similar` (cosine/Levenshtein)
- `llm-rubric` (LLM-graded)
- `cost`, `latency` thresholds
- Custom JavaScript/Python functions

**Red-Teaming**: Built-in adversarial prompts (jailbreaks, PII extraction, bias probing).

**Strengths**:

- Lightweight, no dependencies
- CI/CD friendly
- Version control for prompts

**Weaknesses**:

- Limited to single-turn
- No observability
- Basic agentic support

---

### phoenix (Arize AI): Embedding and Trace Analysis

**Specialization**: Drift detection and embedding space visualization.

**Key Features**:

- **UMAP Clustering**: Visualize embedding drift over time
- **Trace Analysis**: LangChain/LlamaIndex integration
- **Retrieval Evaluation**: Hit rate, MRR, NDCG for RAG
- **LLM Evals**: Hallucination, toxicity, relevance (model-based)

**Use Cases**:

- Production monitoring
- Identifying underperforming clusters
- Dataset curation from production traces

**Technical**:

- Runs locally or cloud
- Exports to Parquet
- Integrates with notebooks

**Limitations**:

- Primarily observability (not benchmarking)
- Limited prompt optimization
- Agentic metrics minimal

---

### autoevals (Braintrust)

**Design**: Model-graded evaluations without labeled data.

**Evaluators**:

- **Factuality**: Claims grounded in context (NLI)
- **ClosedQA**: Answer correctness for closed-domain QA
- **Security**: PII/credentials leakage detection
- **JSONDiff**: Structured output validation

**Implementation**: Uses Claude/GPT via API, returns binary or scored judgments.

**Philosophy**: "Evals as code" - programmatic, version-controlled, reproducible.

**Constraints**:

- Requires evaluator LLM (cost/latency)
- Limited metric diversity
- No dashboards

---

### HELM (Holistic Evaluation of Language Models)

**Scope**: Stanford's comprehensive benchmark suite.

**Dimensions**:

- **Accuracy**: QA, reasoning, knowledge
- **Calibration**: Confidence vs correctness
- **Robustness**: Adversarial, OOD, fairness
- **Efficiency**: Inference cost, latency
- **Toxicity/Bias**: Social harms

**Scenarios**: 42 tasks across question answering, summarization, sentiment, reasoning.

**Metrics**: 59 metrics including exact match, F1, BLEU, toxicity (Perspective API), bias (demographics).

**Process**: Standardized prompts, multiple models, statistical significance testing.

**Limitations**:

- Static benchmark (no agentic)
- Expensive to run
- Limited domain coverage

---

### lm-evaluation-harness (EleutherAI)

**Purpose**: Unified framework for 200+ benchmarks.

**Supported Tasks**: MMLU, HellaSwag, TruthfulQA, HumanEval, GSM8K, Big-Bench.

**Features**:

- Few-shot prompting
- Multiple choice + generative tasks
- Calibration metrics
- HuggingFace integration

**Usage**:

```bash
lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag --num_fewshot 5
```

**Strengths**:

- Reproducibility
- Standardized prompts
- Broad coverage

**Weaknesses**:

- Primarily academic benchmarks
- Limited real-world tasks
- No agentic evaluation

---

## LLM-as-a-Judge: Advanced Topics

### Core Training Approaches

LLM-as-judge models are trained using two primary strategies:

1. **Supervised Fine-Tuning (SFT)** on chain-of-thought and verdict formats to instill structured analysis
2. **Direct Preference Optimization (DPO)** to tune pairwise discrimination power and increase preference alignment

### Key Training Methodologies

**Prompt Design**: Scenario-based prompts that provide context-awareness for instruction-specific evaluations are preferred over unified approaches.

**Data Construction**: Reference-based questioning and role-playing quizzing are used to generate or supplement instruction data in a controlled manner.

**Meta-Evaluation**: Models are evaluated by comparing their output to human evaluation results using classification metrics for binary outputs.

**Bias Mitigation**: Position randomization, multiple chain-of-thought paths, and tie-handling strategies reduce internal inconsistency and position bias.

### Recent Advances

- State-of-the-art LLM judges can achieve up to **85% agreement with human evaluators**, outperforming inter-human consensus levels
- **Themis** uses human-AI collaboration to develop comprehensive evaluation criteria while maintaining flexibility for continuous development
- Specialized domain-specific finetuned judges can match or exceed proprietary model performance when trained on domain-specific data

### Rating Indeterminacy Problem

**Key Research**: [Validating LLM-as-a-Judge Systems Under Rating Indeterminacy](https://blog.ml.cmu.edu/2025/12/09/validating-llm-as-a-judge-systems-under-rating-indeterminacy/) (CMU ML Blog, December 2025)

**The Problem**: Rating indeterminacy occurs when multiple interpretations of rating instructions produce equally valid answers. For example, a response could reasonably be labeled "toxic" (dismissive tone) or "non-toxic" (direct feedback) depending on interpretation. This affects helpfulness, factuality, relevance, and similar subjective judgments.

**Why It Matters**: Current meta-evaluation approaches use forced-choice instructions that eliminate information about acceptable alternatives. When judges resolve indeterminacy differently from humans, downstream task performance suffers.

**Key Findings**:

- Judge systems vary substantially in how they resolve indeterminacy compared to humans
- Forced-choice metrics frequently select suboptimal judge systems
- Multi-label MSE metrics reliably identify high-performing judges

**Recommended Solutions**:

1. **Rating Elicitation**: Allow raters to select all reasonable options (response set elicitation), not just one
2. **Rating Aggregation**: Convert disagreement into multi-label probability vectors
3. **Agreement Measurement**: Use continuous metrics (MSE) on multi-label vectors instead of categorical metrics

**Practical Implementation**:

- Add "Maybe" options to binary evaluation tasks
- Collect response set ratings for new datasets
- Augment existing forced-choice data with ~100 paired response set samples
- If forced-choice is unavoidable, prefer KL-Divergence over categorical metrics

### Performance Considerations

Key factors affecting finetuned judge performance:

- Quality and size of the base model
- Bias impact on output
- Ability to generalize across different scoring setups (pairwise ranking, multi-turn chat, single answer grading)
- Data quality issues

---

## Fine-Tuning LLM Raters

### Core Building Blocks

An effective LLM judge requires three key layers:

1. **How judges reason** (reasoning mechanisms)
2. **What they evaluate** (evaluation criteria)
3. **How assessments are structured** (output formatting)

### Critical Components

**Input Contextualization**: Understanding task requirements and constraints

**Comparison to Quality Standards**: Benchmarking against defined quality criteria

**Multi-step Reasoning**: Breaking down complex judgments into component assessments

**Explanation Generation**: Justifying evaluations with transparent reasoning

**Score Synthesis**: Converting qualitative assessments into quantitative scores

### Fine-Tuning Approaches

**Labeled Dataset Training**: Fine-tuned LLM judges can be trained on labeled evaluation datasets to assess quality of model outputs.

**Examples**:

- **PandaLM**: Fine-tuned on over 300,000 evaluation examples containing instruction-response pairs with human quality judgments
- **Auto-J**: Emphasizes providing structured explanations for evaluations

**Self-Taught Evaluators**: Alternative approach that uses synthetic training data to iteratively improve performance without human annotations.

### Training Data Quality Assessment

**Automated Scoring**: Each label receives a quality score, allowing human reviewers to focus on items below a certain threshold rather than reviewing everything manually.

**Best Practices**:

- Data normalization and standardization
- Feature selection to ensure features appropriately influence predictions
- Avoiding bias in data processing
- Proper train/validation/test splits without "peeking" at test set answers

---

## Preference Learning and Reward Modeling

### Core RLHF Components

Preference learning and reward models form the foundation of modern RLHF systems. In RLHF, a reward model is trained from human-provided comparative feedback to serve as a proxy for human judgment, then used to guide policy optimization through reinforcement learning.

### Reward Model Training

**Bradley-Terry Preference Modeling**: Most common approach using pairwise comparisons between outputs.

**Canonical Loss Function**:

```
L(θ) = -log(σ(r_θ(y_c|x) - r_θ(y_r|x)))
```

where `y_c` and `y_r` are chosen and rejected completions respectively.

**Key Training Practices**:

- Training for typically only **1 epoch** to avoid overfitting
- Using multi-comparison loss functions like K-wise Plackett-Luce for ranking multiple completions
- Incorporating preference margin terms to distinguish magnitude of preference strength

### Advanced Reward Model Types

**Outcome Reward Models (ORMs)**: Output per-token correctness probabilities for verification-focused tasks

**Process Reward Models (PRMs)**: Score intermediate reasoning steps rather than just final outputs, using per-step cross-entropy loss

**Generative Reward Models**: Use prompted language models as judges instead of trained scalar reward models, though these typically underperform specialized models on benchmarks

### RLHF Evaluation Frameworks (as of January 2026)

**General Chat**: RewardBench2, **RMB **, RM-Bench

**Mathematical Reasoning**: RewardMATH, AceMath-RewardBench

**Domain-Specific**:

- **M-RewardBench (multilingual) **
- **RAG-RewardBench (retrieval augmented) **
- **PRM Bench (process rewards) **

**Multimodal**: VL RewardBench, **Multimodal RewardBench**

**Current Research Focus**: Addressing over-optimization of reward models and managing the trade-off between generation quality and model capabilities.

### Key Reward Benchmark Insights

| Benchmark                  | Key Finding                                                                       | Primary Use Case                                 |
| -------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------ |
| **M-RewardBench**          | English-centric reward models show degraded performance on low-resource languages | Multilingual reward model evaluation             |
| **Multimodal RewardBench** | Models can produce confident but incorrect visual reasoning outputs               | Vision-language model alignment evaluation       |
| **PRM-Bench**              | Process-level rewards capture reasoning quality that outcome rewards miss         | Evaluating chain-of-thought and reasoning models |
| **RAG-RewardBench**        | Reward model performance varies significantly across RAG scenarios                | RAG system alignment and retrieval quality       |

---

## Agentic Evaluation Research

### Comprehensive Surveys

**"Survey on Evaluation of LLM-based Agents"** (IBM Research)
Analyzes evaluation methodologies across four critical dimensions:

1. Fundamental agent capabilities (planning, tool use, self-reflection, memory)
2. Application-specific benchmarks (web, software engineering, scientific, conversational agents)
3. Generalist agent benchmarks
4. Agent evaluation frameworks

**"Evaluation and Benchmarking of LLM Agents: A Survey"**
Introduces a two-dimensional taxonomy organizing work by:

- **Evaluation objectives** (what to evaluate: behavior, capabilities, reliability, safety)
- **Evaluation process** (how to evaluate: interaction modes, datasets, metrics, tooling)

### Enterprise-Focused Benchmarks

**KAMI v0.1 (Kamiwaza Agentic Merit Index)**
Evaluates agentic AI for enterprise deployment using 5.5 billion tokens across 35 model configurations. Key finding: traditional benchmarks fail to predict real-world agentic performance, particularly in multi-step tool use and decision-making under uncertainty.

### Key Evaluation Frameworks

**AgentBench**: Comprehensive evaluation across eight interactive environments including operating systems, databases, and web interfaces, measuring planning, reasoning, tool use, and decision-making.

**Evaluation-Driven Development of LLM Agents**: Proposes embedding evaluation throughout the LLM agent lifecycle with continuous feedback loops and adaptation.

### Multi-Agent Approaches

**"A Multi-Agent Framework for Dynamic LLM Evaluation"**
Presents a benchmark self-evolving framework utilizing multi-agent systems to construct evolving test instances with six reframing operations to test diverse queries and problem-solving abilities.

---

## Multi-Turn/Multi-Hop Evaluation as Credit Assignment

### The Core Problem

Traditional RL approaches for LLM agents rely on **sparse outcome rewards** - a single reward at task completion. This creates a fundamental credit assignment problem: when rewards are assigned across an entire trajectory, it becomes impossible to identify which specific decisions contributed positively or negatively to the final result.

**Key Challenge**: In multi-turn reasoning (10+ steps), treating all actions as equally responsible for success or failure prevents the model from learning which intermediate decisions actually mattered.

### Formulating Multi-Turn Eval as RL Credit Assignment

The 2025-2026 research wave reframes multi-turn/multi-hop evaluation as a **turn-level credit assignment problem**, enabling fine-grained optimization rather than trajectory-level learning.

**Core Insight**: Dense intermediate rewards at each reasoning step allow models to learn which specific decisions contributed to success, rather than only receiving feedback at task completion.

---

### Key Papers: Turn-Level Credit Assignment

#### 1. Turn-Level Reward Design for Multi-Turn LLM Agents (NeurIPS 2025)

**Paper**: "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment"
**Authors**: Quan Wei, Siliang Zeng, Chenliang Li, William Brown, Oana Frunza, Wei Deng, Anderson Schneider, Yuriy Nevmyvaka, Yang Katie Zhao, Alfredo Garcia, Mingyi Hong
**Link**: [arXiv:2505.11821](https://arxiv.org/abs/2505.11821)

**Core Contribution**: Systematic study of turn-level reward design for multi-turn RL algorithms.

**Method**:

- Extends GRPO (Group Relative Policy Optimization) to multi-turn variants
- Extends PPO (Proximal Policy Optimization) to multi-turn variants
- Implements two reward types: **verifiable rewards** and **LLM-as-judge rewards**

**Credit Assignment Innovation**:

```
Turn-level reward r_t assigned at each step t, rather than single R at trajectory end
Enables fine-grained gradient signal: ∇θ Σ_t r_t log π_θ(a_t|s_t)
```

**Results**:

- 100% format correctness in outputs
- Greater training stability and faster convergence than trajectory-level baselines
- Highest answer correctness across diverse QA datasets
- Superior performance on multi-turn search tasks requiring reasoning

---

#### 2. OREO: Offline RL for Multi-Step Reasoning (ACL 2025)

**Paper**: "Offline Reinforcement Learning for LLM Multi-step Reasoning"
**Authors**: Huaijie Wang, Shibo Hao, Hanze Dong, Shenao Zhang, Yilin Bao, Ziran Yang, Yi Wu
**Link**: [ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.464/)

**Core Problem Addressed**: DPO fails for multi-step reasoning because it:

1. Relies on paired preference data (not available for multi-step tasks)
2. Treats all tokens uniformly (ineffective for credit assignment)

**OREO Method**:

- Jointly learns policy model and value function
- Optimizes the **soft Bellman Equation** for credit assignment
- No paired preference data required

**Soft Bellman Formulation**:

```
V(s) = E_π[r(s,a) + γV(s') - α log π(a|s)]
Credit assigned via learned value function V(s) at each state
```

**Results**:

- Superior performance on GSM8K and MATH (mathematical reasoning)
- Strong results on ALFWorld (embodied agent control)
- Value function enables **tree search at inference** for further gains

---

#### 3. T-GRPO: Tree-Structured Credit Assignment (arXiv 2026)

**Paper**: "Reinforcement Learning Enhanced Multi-hop Reasoning for Temporal Knowledge Question Answering"
**Authors**: Wuzhenghong Wen, Chao Xue, Su Pan, Yuwei Sun, Minlong Peng
**Link**: [arXiv:2601.01195](https://arxiv.org/abs/2601.01195)

**Innovation**: Tree-Group Relative Policy Optimization (T-GRPO) - recursive, tree-structured exploration for multi-hop reasoning.

**Three-Phase Approach**:

1. **Prompt engineering** → Generate diverse reasoning trajectories
2. **Supervised fine-tuning** → Cold-start initialization with valid trajectories
3. **T-GRPO execution** → Recursive credit assignment across tree structure

**Multi-Hop Credit Assignment**:

```
At each hop h:
  - Exploration establishes causal dependencies on hop h-1
  - Evaluation informed by multi-path feedback from hops h+1...H
  - Enables identification of globally optimal reasoning paths
```

**Results**:

- Surpasses SOTA on two TKGQA benchmarks
- Improved interpretability of reasoning chains
- Robust to noisy temporal annotations

---

#### 4. Agent Lightning: Hierarchical Credit Assignment (Microsoft Research 2025)

**Paper**: "Agent Lightning: Adding Reinforcement Learning to AI Agents Without Code Rewrites"
**Link**: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/agent-lightning-adding-reinforcement-learning-to-ai-agents-without-code-rewrites/)

**Key Innovation**: Hierarchical decomposition of multi-step agent runs into individual LLM calls, each with assigned credit.

**LightningRL Algorithm**:

```
1. Execute multi-step agent task
2. Credit assignment module scores each LLM request's contribution
3. Each (request, reward) pair trains with standard single-step RL (PPO/GRPO)
```

**Advantages**:

- No long sequence stitching (avoids performance degradation)
- Compatible with existing single-step RL algorithms
- Modular credit assignment

**Results** (three real-world scenarios):

- **Text-to-SQL**: Significant accuracy improvement
- **RAG (MuSiQue)**: Better search query generation and multi-hop reasoning
- **Mathematical QA**: Improved tool invocation decisions

---

Mixed-reward RL combining:

- Traditional binary outcome rewards
- **Citation-aware Rubric Rewards (CaRR)**: Decomposes multi-hop questions into verifiable single-hop rubrics
- Requires agents to identify hidden entities, support with citations, build evidence chains

---

### Comparison: Trajectory vs Turn-Level Rewards

| Aspect                 | Trajectory-Level      | Turn-Level                              |
| ---------------------- | --------------------- | --------------------------------------- |
| **Signal density**     | Sparse (1 reward)     | Dense (N rewards)                       |
| **Credit assignment**  | Uniform across steps  | Fine-grained per step                   |
| **Convergence**        | Slow, unstable        | Faster, more stable                     |
| **Error localization** | Impossible            | Precise identification                  |
| **Compute overhead**   | Lower                 | Higher (reward computation per turn)    |
| **Reward design**      | Simple (success/fail) | Complex (requires intermediate metrics) |

---

### Practical Implications for Evaluation

**For Agentic Eval Frameworks**:

1. **Turn-level metrics** should be standard, not just final-outcome metrics
2. **Intermediate reasoning quality** matters as much as final answers
3. **Error attribution** across steps enables targeted improvement

**For Reward Model Training**:

1. Process Reward Models (PRMs) align well with turn-level credit assignment
2. Consider multi-hop verification rubrics (C-GRPO style)
3. Offline methods (OREO) avoid expensive online data collection

**Open Questions**:

- Optimal granularity: token-level vs turn-level vs sub-goal level?
- Automatic turn-level reward generation without human annotation?
- Transfer of credit assignment policies across tasks?

---

## Production Deployment Realities

_Section added from Academia vs Industry debate synthesis (January 2026)_

### Cost-Performance Analysis for Evaluation Methods

Production evaluation requires explicit cost/latency tradeoffs. Below are realistic estimates for common evaluation approaches:

| Method                                 | Cost per Eval  | Latency      | Best Use Case                        |
| -------------------------------------- | -------------- | ------------ | ------------------------------------ |
| **Deterministic (regex, exact match)** | ~$0            | <1ms         | Format validation, known-answer QA   |
| **Embedding similarity**               | $0.0001        | 10-50ms      | Semantic similarity, clustering      |
| **LLM-as-judge (GPT-4)**               | $0.01-0.05     | 2-5s         | Subjective quality, nuanced criteria |
| **LLM-as-judge (Claude Haiku)**        | $0.001-0.005   | 0.5-1s       | Cost-effective quality assessment    |
| **Turn-level rewards**                 | 3-5x base cost | 3-5x latency | Complex reasoning, credit assignment |
| **Human annotation**                   | $5-50          | Hours-days   | Gold standard, benchmark validation  |
| **G-Eval with CoT**                    | $0.02-0.10     | 3-8s         | High-stakes decisions only           |

**Production Guidance**:

- Use deterministic evaluation wherever objective correctness exists
- Reserve LLM-as-judge for subjective dimensions that can't be rule-based
- Turn-level rewards: only when trajectory rewards show high variance (>30%)
- Human evaluation: 1-5% sampling for calibration, not continuous monitoring

### Latency Constraints Are Non-Negotiable

Production SLAs typically require:

- **p50 latency**: <100ms for evaluation
- **p99 latency**: <500ms
- **Evaluation overhead**: <10% of total request time

Evaluation methods that exceed these bounds cannot run in the critical path. Options:

1. **Async evaluation**: Log traces, evaluate offline, alert on degradation
2. **Sampling**: Evaluate 1-10% of traffic with expensive methods
3. **Tiered approach**: Fast deterministic checks inline, slow LLM checks async

### Integration with Existing Infrastructure

Evaluation frameworks that require rebuilding CI/CD pipelines won't get adopted. Practical patterns:

```yaml
# GitHub Actions integration example
- name: Run prompt regression tests
  run: |
    promptfoo eval --config eval.yaml --output results.json
    if [ $(jq '.summary.failRate' results.json) -gt 0.05 ]; then
      echo "Regression detected: >5% failure rate"
      exit 1
    fi
```

**CI/CD Checkpoints**:

1. **Pre-commit**: Syntax/format validation (<1s)
2. **PR checks**: Regression tests on golden set (30-60s)
3. **Staging**: Full benchmark suite with statistical significance (5-30min)
4. **Canary**: Production sampling with alerting (continuous)

---

## LLM-as-Judge: Bias Analysis and Mitigation

_Section synthesized from Academia-Industry debate on systematic biases_

### Documented Biases in LLM Judges

Research has identified systematic biases that compound in production:

**Position Bias** (Zheng et al., arXiv:2306.05685):

- LLM judges favor responses in certain positions (often first)
- **Mitigation**: Randomize presentation order, average across permutations

**Length Bias**:

- Longer responses rated higher regardless of quality
- Particularly problematic for summarization evaluation
- **Mitigation**: Normalize by length, or use length-controlled prompts

**Self-Preference Bias**:

- Models prefer outputs from same model family
- **Mitigation**: Use cross-family judges, ensemble diverse models

**Verbosity Bias**:

- Detailed explanations scored higher even when wrong
- Conflated with "confidence" in reasoning chains
- **Mitigation**: Separate correctness from explanation quality

### Bias Accumulation in Multi-Turn Evaluation

**Critical insight from debate**: Biases don't just add—they multiply across turns.

```
Per-turn bias: 5%
10-turn conversation: 1 - (0.95)^10 = 40% cumulative bias exposure
20-turn conversation: 1 - (0.95)^20 = 64% cumulative bias exposure
```

This means multi-turn agents evaluated with biased judges will systematically drift toward judge-preferred behaviors, not user-preferred behaviors.

### Practical Bias Mitigation (Industry-Validated)

Despite theoretical limitations, these techniques work in production:

1. **Order randomization**: Always shuffle comparison order
2. **Temperature tuning**: Lower temperature (0.1-0.3) reduces variance
3. **Ensemble judging**: 3 diverse models, majority vote
4. **Calibration sets**: Compare judge rankings to human rankings monthly
5. **Bias monitoring**: Track judge preferences over time for drift

**Cost-effective ensemble**:

```python
judges = ["claude-3-haiku", "gpt-4o-mini", "gemini-1.5-flash"]
# Fast, cheap models for ensemble diversity
# Total cost: ~$0.003 per evaluation (vs $0.05 for single GPT-4)
```

---

## Statistical Foundations for Agent Evaluation

_Section added from Academia recommendations, validated against Industry constraints_

### The Multiple Comparison Problem

When evaluating agents across multiple benchmarks, naive p-value interpretation fails:

- 20 benchmarks at α=0.05 → expect 1 false positive by chance
- Without correction, "improvements" may be statistical noise

**Required corrections**:

- **Bonferroni**: Divide α by number of comparisons (conservative)
- **Holm-Bonferroni**: Sequential rejection (less conservative)
- **Benjamini-Hochberg**: Control false discovery rate (recommended for exploration)

### Sample Size Requirements for Agent Evaluation

Agent evaluations have high variance. Required samples for 80% power:

| Effect Size    | Samples Needed | Practical Interpretation |
| -------------- | -------------- | ------------------------ |
| Large (d=0.8)  | ~25 per group  | Obvious improvement      |
| Medium (d=0.5) | ~65 per group  | Noticeable improvement   |
| Small (d=0.2)  | ~400 per group | Subtle improvement       |

**Multi-turn complication**: Conversations are not i.i.d. Effective sample size is lower than conversation count due to within-conversation correlation.

### Confidence Intervals Over P-Values

Report confidence intervals, not just "statistically significant":

```
Bad:  "Model B is significantly better (p < 0.05)"
Good: "Model B success rate: 78% [74%, 82%] vs Model A: 71% [67%, 75%]"
```

This lets stakeholders assess practical significance, not just statistical significance.

### Error Propagation in Multi-Step Agents

**Mathematical reality**: Per-step errors compound multiplicatively.

```
Step success rate: 85%
5-step task:  0.85^5  = 44% trajectory success
10-step task: 0.85^10 = 20% trajectory success
15-step task: 0.85^15 = 9% trajectory success
```

**Implications**:

1. Small per-step improvements yield large trajectory gains
2. Long-horizon agents need >95% per-step reliability
3. Error recovery mechanisms are essential, not optional

---

## Tiered Evaluation Strategy Framework

_Framework synthesized from debate: balancing rigor with operational feasibility_

### Risk-Based Evaluation Tiers

Not all evaluations need the same rigor. Match evaluation depth to risk:

| Tier       | Risk Level               | Evaluation Approach                     | Latency Budget   |
| ---------- | ------------------------ | --------------------------------------- | ---------------- |
| **Tier 1** | Low (internal tools)     | Deterministic + sampling                | <10ms            |
| **Tier 2** | Medium (user-facing)     | LLM-judge + weekly human audit          | <100ms           |
| **Tier 3** | High (financial/medical) | Full human review + formal verification | Hours acceptable |

### Evaluation Escalation Pattern

```
1. Fast deterministic checks (format, safety filters)
   ↓ Pass
2. LLM-as-judge on sampled traffic (quality scoring)
   ↓ Score < threshold
3. Human review queue (edge cases, failures)
   ↓ Pattern detected
4. Root cause analysis + benchmark update
```

### Connecting to Business Metrics

Academic capability metrics must correlate with business outcomes:

| Eval Metric          | Business Metric           | Expected Correlation |
| -------------------- | ------------------------- | -------------------- |
| Task completion rate | Support ticket deflection | Strong (r > 0.7)     |
| Response relevance   | User satisfaction (CSAT)  | Moderate (r ~ 0.5)   |
| Factual accuracy     | Trust/NPS                 | Moderate-Strong      |
| Latency p95          | User engagement           | Inverse correlation  |

**Validation requirement**: Quarterly correlation analysis between eval metrics and business KPIs. If correlation breaks down, eval metrics need recalibration.

---

## Multimodal and Audio Evaluation

### Vision-Language Model Evaluation

**Core Challenges**:

- Cross-modal alignment: Do visual and textual representations match?
- Grounding accuracy: Can the model correctly locate objects/regions?
- Hallucination in vision: Describing objects not present in images

**Key Benchmarks**:

| Benchmark          | Focus                         | Key Metrics                               |
| ------------------ | ----------------------------- | ----------------------------------------- |
| **VQAv2**          | Visual question answering     | Accuracy on open-ended questions          |
| **GQA**            | Compositional reasoning       | Consistency, validity, plausibility       |
| **POPE**           | Object hallucination          | Precision, recall, F1 on object existence |
| **MMBench**        | Comprehensive VLM evaluation  | 20 ability dimensions                     |
| **MMMU**           | Multi-discipline reasoning    | College-level multimodal tasks            |
| **VL RewardBench** | Vision-language reward models | Preference alignment accuracy             |

**Evaluation Approaches**:

1. **Reference-based**: Compare against ground truth captions/answers
2. **Reference-free**: LLM-as-judge for open-ended visual QA
3. **Human evaluation**: Still gold standard for subjective quality
4. **Automated metrics**: CLIPScore, BLIP-2 similarity for image-text alignment

### Audio and Speech Evaluation

**Speech-to-Text (ASR)**:

| Metric                         | Description                                  | Use Case                                |
| ------------------------------ | -------------------------------------------- | --------------------------------------- |
| **WER** (Word Error Rate)      | Edit distance normalized by reference length | Standard ASR benchmark                  |
| **CER** (Character Error Rate) | Character-level WER                          | Languages without clear word boundaries |
| **SER** (Sentence Error Rate)  | Percentage of sentences with any error       | High-stakes applications                |

**Text-to-Speech (TTS)**:

- **MOS** (Mean Opinion Score): Human ratings 1-5 for naturalness
- **PESQ/POLQA**: Perceptual audio quality metrics
- **Speaker similarity**: Cosine similarity of speaker embeddings
- **Prosody evaluation**: Pitch, rhythm, stress pattern accuracy

**Audio Understanding (Emerging)**:

- **Audio-language models**: Evaluate on audio captioning, sound event detection
- **Multimodal dialogue**: Speech + vision + text coherence
- **Real-time evaluation**: Latency-constrained audio processing

**Key Frameworks**:

- **OpenAI Whisper Eval**: ASR benchmark across 96 languages
- **LibriSpeech**: Standard clean/noisy speech recognition benchmark
- **VoxCeleb**: Speaker verification and diarization
- **AudioCaps**: Audio captioning benchmark

### Multimodal Agent Evaluation

**VAGEN Framework** (2025): Evaluates Vision-Language Model agents with:

- Turn-level visual grounding accuracy
- Cross-modal consistency across interactions
- Action success rate in visual environments

**Evaluation Dimensions for Multimodal Agents**:

1. **Perception accuracy**: Correct interpretation of visual/audio input
2. **Cross-modal reasoning**: Integrating information across modalities
3. **Grounded action generation**: Actions that correctly reference visual elements
4. **Temporal coherence**: Consistent understanding across video frames/audio streams

---

## Long-Context Evaluation

### The Challenge

Standard benchmarks evaluate on contexts of 2K-8K tokens. Production LLMs now support 100K-1M+ tokens, but evaluation hasn't kept pace.

**Why It Matters**:

- Long documents (legal contracts, codebases) require full-context understanding
- Multi-turn conversations accumulate context
- RAG systems may retrieve extensive context

### Key Benchmarks

**RULER** (NVIDIA, 2024):

Synthetic benchmark testing retrieval and reasoning at extreme lengths.

| Task Type                | Description                        | Tested Lengths    |
| ------------------------ | ---------------------------------- | ----------------- |
| **Needle-in-a-Haystack** | Find specific fact in long context | 4K - 128K tokens  |
| **Multi-hop reasoning**  | Connect facts across document      | Up to 64K tokens  |
| **Variable tracking**    | Track entity state changes         | Up to 32K tokens  |
| **Aggregation**          | Summarize distributed information  | Up to 128K tokens |

**Key Finding**: Many models claiming 128K context show significant degradation beyond 32K tokens.

**Needle-In-A-Haystack (NIAH)**:

Simple but effective: Hide a specific fact ("needle") at various positions in a long document ("haystack"), test retrieval.

```
Position tested: [0%, 10%, 20%, ..., 90%, 100%] of document
Context lengths: [1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K]
Metric: Retrieval accuracy at each (position, length) pair
```

**Common Findings**:

- "Lost in the middle" effect: Lower recall for information in document center
- Performance cliff: Sharp accuracy drop at certain context lengths
- Position bias: Beginning and end of context retrieved more reliably

**LongBench** (THUDM):

Real-world long-context tasks across 6 categories:

- Single-document QA
- Multi-document QA
- Summarization
- Few-shot learning
- Code completion
- Synthetic tasks

### Evaluation Best Practices

1. **Test multiple positions**: Don't just test needle at random positions
2. **Use realistic distributions**: Real documents have non-uniform information density
3. **Measure degradation curves**: Plot performance vs. context length
4. **Include reasoning tasks**: Not just retrieval, but multi-hop reasoning over long context
5. **Cost-normalize**: Longer contexts = higher inference cost; report quality per dollar

### Production Considerations

**Chunking strategies** affect long-context evaluation:

- Evaluate both chunked retrieval (RAG) and full-context approaches
- Compare quality vs. cost tradeoffs
- Test context window utilization patterns

---

## OpenAI Evaluation Flywheel

### The Continuous Improvement Loop

OpenAI's evaluation approach emphasizes a **flywheel** where evaluation drives model improvement in a continuous cycle:

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION FLYWHEEL                       │
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ Deploy   │───▶│ Collect  │───▶│ Identify │             │
│   │ Model    │    │ Feedback │    │ Failures │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│        ▲                               │                    │
│        │                               ▼                    │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ Train on │◀───│ Curate   │◀───│ Create   │             │
│   │ New Data │    │ Dataset  │    │ Evals    │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Core Principles

**1. Evals Before Training**

- Write evaluations that define desired behavior BEFORE training
- Evals are specifications, not just tests
- Failed evals reveal capability gaps to target

**2. Production Feedback Loop**

- Real user interactions surface edge cases benchmarks miss
- Thumbs up/down, regenerations, and edits are implicit feedback
- High-value failure cases become new eval examples

**3. Eval-Driven Prioritization**

- Improvement efforts focus on failing evals
- New capabilities require new evals first
- Regression testing prevents capability loss

### Implementation Pattern

**Stage 1: Define Eval Suite**

```python
# OpenAI Evals structure
eval_suite = {
    "capability_evals": [...],    # Core model capabilities
    "safety_evals": [...],        # Harmful content, refusals
    "format_evals": [...],        # JSON, code, structured output
    "domain_evals": [...],        # Task-specific benchmarks
}
```

**Stage 2: Continuous Monitoring**

- Run evals on every model checkpoint
- Track metrics over time for regression detection
- Alert on significant performance changes

**Stage 3: Failure Mining**

- Cluster production failures by type
- Convert representative failures to eval cases
- Weight eval importance by production frequency

**Stage 4: Targeted Improvement**

- Fine-tune or prompt-engineer on weak areas
- Validate improvement on held-out eval split
- Monitor for regressions on other capabilities

### Practical Adoption

**For Teams Without OpenAI Scale**:

1. **Start with production logs**: Your users are your evaluators
2. **Build golden sets**: 100-500 high-quality examples per capability
3. **Automate regression checks**: Run evals on every deployment
4. **Close the loop**: Ensure eval failures lead to improvements

**Eval Quality > Eval Quantity**:

- 50 carefully curated examples beat 5000 noisy ones
- Each eval should test ONE specific behavior
- Ambiguous evals create noise, not signal

### OpenAI Evals Framework

Open-source framework for evaluation: https://github.com/openai/evals

**Key Features**:

- YAML-based eval definitions
- Built-in model-graded evaluations
- Comparison modes for A/B testing
- Extensible with custom evaluators

```yaml
# Example eval definition
example_eval:
  id: example_eval.basic
  metrics: [accuracy]
  description: Test basic capability

example_eval.basic:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: data/example_samples.jsonl
```

---

## Other Info

### Agentic-Specific Gaps

**Evaluation Metrics**:

1. **Multi-turn coherence**: Maintain context across 10+ interactions
2. **Tool selection quality**: Precision/recall for tool invocation
3. **Error recovery**: Graceful handling of failed actions
4. **Cost-benefit analysis**: Trade-offs between accuracy and API cost
5. **Long-horizon planning**: Evaluate 50+ step workflows

### General Gaps

**Cross-Model Standards**:

- Standardized metrics for multi-turn interactions
- Real-world task generalization metrics
- Temporal consistency (same query, different times)
- Adversarial robustness testing
- Domain-specific evaluation suites (medical, legal, finance)

### Infrastructure Needs

**Tooling**:

- Automated synthetic dataset generation
- Multi-modal evaluation tools (vision + text)
- Streaming response quality assessment
- Production monitoring integration
- Sandbox environments for safe execution
- Replay/mock systems for deterministic testing
- Human preference learning for agentic behavior

### Emerging Benchmarks

**AgentBench**: 8 environments (OS, DB, web navigation)

**GAIA**: Real-world assistant tasks requiring tool use

**WebArena**: Autonomous web agents in realistic sites

**InterCode**: Interactive coding challenges

---

## Key Takeaways

1. **LLM-as-judge approaches** have matured significantly, with fine-tuned judges achieving 85% agreement with human evaluators

2. **Reward model training** has evolved beyond simple Bradley-Terry models to include process reward models and multi-comparison approaches

3. **Agentic evaluation remains immature** compared to traditional LLM evaluation, with significant gaps in multi-turn, tool-use, and long-horizon metrics

4. **Enterprise benchmarks** like KAMI demonstrate that academic benchmarks fail to predict real-world agentic performance

5. **Fine-tuning judge models** requires careful attention to bias mitigation, data quality, and generalization across scoring setups

6. **Preference learning** has expanded to include sophisticated approaches like DPO and specialized domain-specific reward models

7. **Production observability** is critical but currently siloed by framework (LangSmith for LangChain, etc.)

8. **Multi-modal evaluation** is emerging but still lacks standardized frameworks

9. **Multi-turn evaluation as credit assignment** is the emerging paradigm (2025-2026): Turn-level rewards via extended GRPO/PPO, offline methods (OREO), and tree-structured approaches (T-GRPO) enable fine-grained optimization that trajectory-level rewards cannot achieve

10. **Hierarchical credit assignment** (Agent Lightning) allows existing single-step RL algorithms to train multi-step agents without architectural changes

---

## References and Further Reading

### Key Papers

- "Survey on Evaluation of LLM-based Agents" (IBM Research, 2025)
- "Evaluation and Benchmarking of LLM Agents: A Survey" (ACL 2025)
- "Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Considerations" (arXiv 2024)
- "Self-Taught Evaluators" (arXiv 2024)
- "RLHF 101: A Technical Tutorial" (CMU ML Blog, 2025)

### Multi-Turn Credit Assignment Papers

- "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment" (NeurIPS 2025): https://arxiv.org/abs/2505.11821
- "Offline Reinforcement Learning for LLM Multi-step Reasoning" (ACL 2025 Findings): https://aclanthology.org/2025.findings-acl.464/
- "Reinforcement Learning Enhanced Multi-hop Reasoning for Temporal Knowledge QA" (arXiv 2026): https://arxiv.org/abs/2601.01195
- "Agent Lightning: Adding RL to AI Agents" (Microsoft Research 2025): https://www.microsoft.com/en-us/research/blog/agent-lightning/
- "Training Task Reasoning LLM Agents for Multi-turn Task Planning via Single-turn RL" (arXiv 2025): https://arxiv.org/abs/2509.20616
- "VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents" (2025): https://vagen-ai.github.io/

### Tools and Frameworks

- ragas: https://github.com/explodinggradients/ragas
- deepeval: https://github.com/confident-ai/deepeval
- langsmith: https://smith.langchain.com
- promptfoo: https://github.com/promptfoo/promptfoo
- phoenix: https://github.com/Arize-ai/phoenix
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness

### Benchmarks

- HELM: https://crfm.stanford.edu/helm/
- AgentBench: https://github.com/THUDM/AgentBench
- GAIA: https://huggingface.co/gaia-benchmark
- KAMI: https://docs.kamiwaza.ai/research/papers/kami-v0-1
- RULER (Long-context): https://github.com/hsiehjackson/RULER
- LongBench: https://github.com/THUDM/LongBench
- MMBench (Vision-Language): https://github.com/open-compass/MMBench
- VL RewardBench: https://huggingface.co/datasets/MMInstruction/VLRewardBench
- OpenAI Evals: https://github.com/openai/evals

---

<div align="center">

◆ ◆ ◆

</div>

---

## Method

This report was generated through a multi-agent workflow:

1. **Baseline Extraction**
   Initial extraction was performed by a deep-search agent to establish the baseline landscape.

2. **Debating Agents**
   Multiple agents, prompted as "Production Engineers" and "Academic Researchers," debated the baseline to expose blind spots. This resulted in the addition of Bias Mitigation, Statistical Foundations, and Deployment Realities sections.

3. **Content Enrichment**
   Selected phrases went through detailed explanation (e.g., decomposing M-RewardBench beyond its surface name into its underlying typological impact).

4. **Fact-Checking Guardrails**
   A fact-check by a different model API was executed against sweeping statements (e.g., "it is the standard for global product launches") and to ensure numbers have sources of reference.

5. **Continuous Freshness**
   This document is linked to a GitHub Action that triggers monthly to refresh the information.
