# ReAct++ WebShop Agent

An implementation of **ReAct++** and **Uncertainty-Aware LLM Agents** for the WebShop benchmark, developed as part of my M.Tech thesis at the Department of Computer Science and Engineering, Indian Institute of Technology Kharagpur.

## Overview

Large Language Models (LLMs) have shown remarkable reasoning capabilities, but their performance in sequential decision-making environments remains limited by inconsistent reasoning paths, stochastic action generation, and lack of uncertainty awareness.

This repository explores the evolution of LLM-based agents for the WebShop benchmark through three stages:

### 1. ReAct

Baseline implementation of the ReAct (Reason + Act) framework, where the agent alternates between reasoning and environment actions.

### 2. ReAct++

An enhanced version of ReAct that improves reasoning-action synergy through:

* Few-Shot Prompting
* Multi-Response Generation
* Self-Consistency Aggregation
* Majority Voting
* Semantic Similarity Ranking

### 3. Uncertainty-Aware Agent

A principled framework for uncertainty-aware action selection using:

* State-Action Scoring Functions
* Conformal Prediction
* Conditional Conformal Prediction
* Adaptive Thresholding
* BFS-Based Planning

Instead of committing to a single action, the agent constructs calibrated action sets and explores them efficiently.

---

## Research Contributions

### ReAct++

* Enhanced contextual grounding through few-shot prompting.
* Improved reasoning consistency using self-consistency.
* Majority-voting based action aggregation.
* Semantic similarity based search-action selection.
* Automated evaluation on 500 WebShop instructions.

---





## Ongoing Research

The uncertainty-aware agent represents ongoing research aimed at improving the reliability and efficiency of LLM-based decision-making.

Current work focuses on:
- Calibrated uncertainty estimation
- Conformal prediction for action selection
- State-dependent adaptive thresholding
- Structured exploration using BFS

The implementation and results presented in this repository should be considered an evolving research prototype. Additional improvements and experimental evaluations are currently in progress.




## WebShop Environment

WebShop is a large-scale simulated e-commerce environment designed for evaluating grounded language agents.

Features:

* 1.1M+ real-world products
* 12K+ human-written instructions
* Dynamic action space
* Multi-step reasoning requirements
* Sparse reward signals

The agent must search, navigate, inspect, and purchase products satisfying a natural language instruction.

---

## Repository Structure

```text
reactpp-webshop-agent/
│
├── react.py
│
├── reactpp_selfconsistency.py
├── selfconsistency.py
│
├── uncertainty_aware.py
├── conformal_predictor_webshop.py
├── run_conformal_calibration_webshop.py
│
│
├── MTP_1.pdf
├── MTP_2.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```

Update the structure above if your uncertainty-aware implementation uses different helper file names.

---

## ReAct++ Methodology

The ReAct++ framework extends ReAct using:

### Few-Shot Prompting

Multiple demonstration trajectories are included in the prompt to improve contextual understanding and reasoning quality.

### Self-Consistency

For each decision step:

1. Generate multiple reasoning-action trajectories.
2. Group outputs by action type.
3. Aggregate outputs using:

   * Majority Voting for click actions
   * Semantic Similarity Ranking for search actions
4. Execute the most representative action.

### Action Categories

```text
search[...]
click[...]
think[...]
```

---

## Experimental Setup

### Model

* Meta-Llama-3.1-8B-Instruct
* Qween 3.1
* 4-bit Quantization
* Hugging Face Transformers
* BitsAndBytes

### Inference Parameters

```text
Temperature: 0.9
Top-p: 0.9
Top-k: 50
Responses per step: 10
Max tokens: 100
```

### Hardware

* 4 × NVIDIA A16 GPUs
* CUDA 13
* PyTorch 2.2+

---

## Results

### ReAct++ Performance

| Method    | Score | Success Rate |
| --------- | ----- | ------------ |
| ReAct-1   | 49.6  | 24.2%        |
| ReAct-2   | 60.2  | 28.2%        |
| ReAct-1SC | 53.3  | 28.9%        |
| ReAct-2SC | 60.3  | 33.6%        |

### Improvements Over Baseline

* +9.4% Success Rate
* +21.6% Score Improvement

---

## Preliminary Results: Uncertainty-Aware Agent

The uncertainty-aware framework is currently under active development. Preliminary experiments on 200 WebShop tasks produced the following results:

| Metric | Value |
|----------|----------|
| Success Rate | 52.5% |
| Average Reward | 0.78 |
| Average Trajectory Length | 4.045 |

---


## Thesis Information

### ReAct++: Enhancing Reasoning-Action Synergy through Few-Shot Prompting and Self-Consistency

### Uncertainty-Aware Action Selection for LLM Agents Using Conformal Prediction and BFS-Based Planning

M.Tech Thesis
Department of Computer Science and Engineering
Indian Institute of Technology Kharagpur

Supervisor:
Prof. Sourangshu Bhattacharya

---

---

## License

This repository is released for research and educational purposes.
