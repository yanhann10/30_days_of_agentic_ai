# Self-Evolving Agents Knowledge Graph

> **Interactive exploration of how AI agents learn to improve themselves**

<p align="center">
  <img src="reference/v3.png" alt="Evolution Methods Graph" width="700">
</p>

The knowledge graph visualization allows you to:

- **Understand contex** click nodes to see papers that use each method
- **Explore connections** between 38 evolution methods
- **Track temporal trends** - use play button to see when methods emerged

<p align="center">
  <a href="https://selfevolvekg.vercel.app/">
    <img src="https://img.shields.io/badge/Launch-Interactive_Graph-4ECDC4?style=for-the-badge" alt="Launch Graph">
  </a>
</p>

---

## Overview

This project transforms 200 papers from the [Awesome-Self-Evolving-Agents](https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents) repository into an **interactive knowledge graph** that visualizes how different evolution methods connect and enable evovling agents.

### Why This Matters

Self-evolving systems that can:
- Learn from their own outputs
- Optimize their own prompts
- Create and refine their own tools
- Collaborate to discover better solutions


---

## The Evolution Taxonomy

Based on the authoritative taxonomy from the research survey:


---

## Key Evolution Methods

### LLM Behaviour Optimisation
| Method | How It Works |
|--------|--------------|
| **Supervised Fine-Tuning** | Train on curated (input, output) pairs |
| **RL from Feedback** | Train reward model, optimize policy via PPO/DPO |
| **Self-Play Learning** | Agent plays both sides, learns from outcomes |
| **Bootstrapping from Rationales** | Filter correct reasoning, fine-tune on them (STaR) |
| **Tree-of-Thought Search** | Branch into paths, evaluate, prune bad branches |
| **Self-Consistency Decoding** | Sample multiple solutions, majority voting |

### Prompt Optimisation
| Method | How It Works |
|--------|--------------|
| **Edit-Based Prompt Search** | Iteratively edit prompts based on feedback |
| **Evolutionary Prompt Optimization** | Population of prompts, mutation/crossover, selection |
| **Generative Prompt Engineering** | LLM proposes candidates, evaluates, selects best |
| **Text Gradient Optimization** | Natural language feedback as gradient signal |

### Memory Optimization
| Method | How It Works |
|--------|--------------|
| **Agent Workflow Memory** | Store successful action sequences, retrieve for similar tasks |
| **Long-Term Memory Systems** | Store episodes with outcomes, query and apply lessons |
| **Compressive Memory** | Compress long context into retrievable gist representations |

### Tool Optimization
| Method | How It Works |
|--------|--------------|
| **Tool Instruction Fine-Tuning** | Fine-tune on tool-use demos, generalize to new tools |
| **RL for Tool Selection** | Reward for successful tool use, policy learns when/how |
| **Tool Discovery and Retrieval** | Search tools, plan sequence, execute and verify |
| **Tool Creation** | Agent writes code to create new tools |

### Multi-Agent Optimisation
| Method | How It Works |
|--------|--------------|
| **Automatic MAS Construction** | Auto-construct multi-agent systems from task requirements |
| **Workflow Optimization** | Evolve agent communication patterns (MetaGPT, AFlow) |
| **Multi-Agent Debate** | Agents argue positions, refine, converge on answer |
| **Symbolic Learning for Agents** | Agents learn symbolic rules from interactions |

### Domain-Specific Methods
| Domain | Method | How It Works |
|--------|--------|--------------|
| **Medical Diagnosis** | Multi-Agent Medical Systems | Specialized agents collaborate on diagnosis |
| **Molecular Discovery** | Chemistry Tool Agents | LLM augmented with chemistry tools |
| **Code Refinement** | Iterative Code Refinement | Generate, test, refine based on results |
| **Code Debugging** | Self-Debug | Execute code, read errors, fix bugs iteratively |
| **Finance** | Financial Decision Agents | Multi-agent systems with verbal reinforcement |

---


## Technical Implementation

### Pipeline Architecture

```
GitHub README        PDF Download        Method Extraction       Visualization
      |                   |                     |                     |
  200 papers    -->   112 PDFs      -->    38 methods      -->   D3.js graph
  parsed              from arXiv           19 connections        interactive
```

### Tools Used

PyMuPDF, D3.js

---

## References

- [Awesome-Self-Evolving-Agents](https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents) - Source paper repository
- [Self-Evolving AI Agents Survey](https://arxiv.org/abs/2508.07407) - Comprehensive survey paper
- [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) - Open-source evolution framework

