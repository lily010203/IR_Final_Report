# IR_Final_Report

# LLM-based Session-Based Recommendation for Instacart

This repository contains the implementation of a **Candidate-Constrained Generative Recommendation Framework** using `Qwen-2.5-7B-Instruct`. The project addresses the cold-start problem in session-based recommendation by leveraging the semantic reasoning capabilities of Large Language Models (LLMs).

## Task Overview
The goal is to predict the next item a user will buy based on their purchase history. We focus on comparing the performance of **Zero-Shot LLM** against a traditional **BERT4Rec baseline** across different user groups (Cold-Start, Warm, Active).

## Prerequisites
- **Python Version:** 3.8+
- **Hugging Face Account:** You need an access token to use the Inference API.

### Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/lily010203/IR_Final_Report.git
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Setup:**
   - Download the [Instacart Dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset).
   - Unzip `orders.csv`, `products.csv`, and `order_products__prior.csv`.
   - Place them into a folder named `instacart/` in the root directory of this project.

### Usage

1. **Configure Token:**
   Open `LLM_version2.py` and replace the placeholder string with your actual Hugging Face Access Token:
   ```python
   MY_HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
   ```

2. **Run the experiment:**
   ```bash
   python LLM_version2.py
   ```

### Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | Qwen-2.5-7B-Instruct | LLM used for zero-shot inference |
| **Top-K** | 10 | Number of recommendations generated |
| **Sample Size** | 150 (50 per group) | Total users evaluated for balance |
| **Context Length** | Last 10 items | History window size |
| **Strategy** | Candidate Constraints | User History + Top-50 Popular Items |
| **Retry Logic** | Exponential Backoff | Max 5 retries (2s, 4s, 8s, 16s, 32s) |

### Experiment Results

We use **Recall@10** as the primary evaluation metric.

| User Group | Baseline (BERT4Rec) | **Ours (LLM)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Cold-Start** (<5 orders) | 0.036 | **0.140** | **+288% (3.8x)** |
| **Warm** (5-20 orders) | 0.014 | **0.140** | **+900% (10x)** |
| **Active** (>20 orders) | 0.0004 | **0.080** | **+19900% (200x)** |

Full evaluation logs can be found in `evaluation_results.csv`.
