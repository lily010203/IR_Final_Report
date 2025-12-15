# IR_Final_Report

# LLM-based Session-Based Recommendation for Instacart

This repository contains the implementation of a **Candidate-Constrained Generative Recommendation Framework** using `Qwen-2.5-7B-Instruct`. The project addresses the cold-start problem in session-based recommendation by leveraging the semantic reasoning capabilities of Large Language Models (LLMs).

##  Task Overview
The goal is to predict the next item a user will buy based on their purchase history. We focus on comparing the performance of **Zero-Shot LLM** against a traditional **BERT4Rec baseline** across different user groups (Cold-Start, Warm, Active).

##  Prerequisites
- **Python Version:** 3.8+
- **Hugging Face Account:** You need an access token to use the Inference API.

### Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/lily010203/IR_Final_Report.git
   ```
2.Install dependencies:
   pip install -r requirements.txt
3.Dataset Setup:
   Download the [Instacart Dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset).
   Unzip orders.csv, products.csv, and order_products__prior.csv.
   Place them into a folder named instacart/ in the root directory of this project.

### Usage
1.Configure Token: Open LLM_version2.py and replace the placeholder string with your actual Hugging Face Access Token:
   MY_HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
2.Run the experiment:
   python LLM_version2.py
### Hyperparameters
