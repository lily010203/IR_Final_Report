# Dense + SASRec Recommendation System (Instacart)

This repository contains an implementation of a **next-item recommendation system** using a hybrid approach:
- Dense Retrieval (Sentence-BERT)
- Sequential Recommendation (SASRec)

The code trains a SASRec model and evaluates recommendations for different user groups on the Instacart dataset.

---

## 1. Directory Structure

```
.
├── team12_DS2.py                  # Main training & evaluation script
├── README.md
└── instacart_data/
    ├── orders.csv
    ├── order_products__prior.csv
    ├── products.csv
    ├── aisles.csv
    └── departments.csv
```

---

## 2. Requirements

### Python Version
- Python **3.8+** recommended

### Required Libraries

Install dependencies with:

```bash
pip install torch sentence-transformers scikit-learn pandas tqdm
```

### Optional (GPU)

- CUDA-enabled GPU is supported
- If CUDA is unavailable, the code will automatically fall back to CPU

---

## 3. Dataset Preparation

Download the Instacart dataset from Kaggle:

https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

Place the following files under `./instacart_data/`:

- `orders.csv`
- `order_products__prior.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

No additional preprocessing is required.

---

## 4. Configuration

Key configuration parameters (editable in `main.py`):

```python
MAX_LEN = 50
MAX_EVAL_USERS = 500
LOG_EVERY = 100
```

---

## 5. How to Run(Usage)

```bash
python team12_DS2.py
```

---
## 6.Hyperparameter
1.系統與全域設定（Global Configuration）
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **device** | "cuda" / "cpu" | 自動選擇運算裝置，若系統支援 GPU 則使用 CUDA，否則使用 CPU |
| **DATA_DIR** | ./instacart_data | Instacart 資料集所在的資料夾路徑 |
| **MAX_LEN	50** | 50 | SASRec 使用的最長使用者行為序列長度，只保留最近的 50 筆互動 |
| **MAX_EVAL_USERS** | 500 | 每一類使用者（Cold / Warm / Active）最多評估的使用者數量 |
| **LOG_EVERY** | 100 | 評估階段中，每處理多少位使用者輸出一次進度紀錄 |

2.Dense Retrieval（Sentence-BERT）
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Sentence-BERT** | model	msmarco-distilbert-base-v4 | 預訓練的 Sentence-BERT 模型，用於將商品文字轉換為向量表示 |
| **Embedding normalization** | True | 對向量進行 L2 正規化，以利後續 cosine similarity 計算 |
| **Similarity metric** | Cosine similarity | 使用餘弦相似度衡量使用者與商品之間的相似程度 |
| **Dense candidate siz** | 200 | Dense Retrieval 階段為每位使用者選取的候選商品數量 |
| **LOG_EVERY** | 100 | 評估階段中，每處理多少位使用者輸出一次進度紀錄 |

3.SASRec 資料集建構（Dataset Construction）
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Padding token** | 0 | 用於補齊序列長度的 padding 索引值 |
| **Sequence truncation** | 最近 MAX_LEN 筆 | 超過最大長度的序列只保留最近的互動行為 |
| **Training strategy** | Sliding window | 使用滑動視窗方式，以前序互動預測下一個商品 |

4.SASRec 模型架構（Model Architecture）
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Item embedding dimension** | 64 | 商品向量的嵌入維度 |
| **Positional embedding dimension** | 64 | 位置向量的嵌入維度，用於表示順序資訊 |
| **Transformer layers** | 2 | Transformer Encoder 的層數 |
| **Attention heads** | 2 | Self-Attention 的注意力頭數 |
| **TFeedforward dimension** | 256 | Transformer 中前饋網路（FFN）的隱藏層維度 |
| **Attention mask** | Causal mask |  使用上三角遮罩，避免模型看到未來的行為|

5️. 訓練參數（Training Hyperparameters）
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | Adam | 使用 Adam 優化器進行模型訓練 |
| **Learning rate** | 1e-3 | Adam 優化器的學習率 |
| **Loss function** | CrossEntropyLoss | 多類別分類損失，用於下一商品預測 |
| **Batch size** | 128 | 每個訓練批次的樣本數 |
| **Epochs** | 5 | SASRec 模型的訓練回合數 |
| **Training users** | ≥ 5  | 僅使用 Warm 與 Active 使用者進行模型訓練 |

6️. 評估設定（Evaluation Configuration）
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Evaluation metrics** | Recall@K、NDCG@K | 用於評估推薦結果準確度與排序品質的指標 |
| **K valuesK  | {5, 10, 20} | 評估前 K 個推薦結果 |
| **Ground truth** | 最後一次互動 | 使用者實際購買的下一個商品 |
| **Cold users** | Dense Retrieval | 冷啟動使用者僅使用 Dense Retrieval |
| **Warm users** | Dense + SASRec | 使用 Dense 產生候選，再由 SASRec 重新排序 |
| **Active users** | SASRec only | 活躍使用者直接使用 SASRec 進行推薦 |

---

## 7. Training Details

- Model: SASRec
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Epochs: 5
- Batch size: 128

---

## 8. Evaluation Output

The script prints Recall@K and NDCG@K for different user groups.

Grup	K	Recall@K	NDCG@K
Active	5	0.152	0.05880
        10	0.20 	0.05781
        20	0.252	0.05737


Grup	K	Recall@K	NDCG@K
Warm	5	0.224	0.08666
        10	0.242 	0.06995
        20	0.254	0.05783

Grup	K	Recall@K	NDCG@K
Warm	5	0.224	0.08666
        10	0.242 	0.06995
        20	0.254	0.05783

---

## 9. Logging

Progress is printed every `LOG_EVERY` users during evaluation.

---

## 10. Notes

- Dense retrieval uses cosine similarity
- SASRec uses only item IDs
- Cold users rely on dense retrieval only

---

## 11. Limitations

- No validation split
- No fine-tuning of Sentence-BERT
- One ground-truth item per user
