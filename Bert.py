"""
Instacart cold-start experiment
- Dense retrieval (pretrained sentence-transformers)
- Transformer (BERT) next-item classifier (on top-N items)
- Evaluate by Recall@k and NDCG@k for user groups (cold/warm/active)

Required packages:
pip install pandas numpy tqdm scikit-learn sentence-transformers transformers torch faiss-cpu

Note:
- Download the Instacart CSVs and put them into DATA_DIR:
  orders.csv, order_products__prior.csv, order_products__train.csv, products.csv, aisles.csv, departments.csv
  (the dataset is available on Kaggle — login required). See: Kaggle Instacart dataset pages. :contentReference[oaicite:2]{index=2}
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import math

# for dense retrieval
from sentence_transformers import SentenceTransformer
import faiss  # make sure faiss-cpu is installed

# for transformer training
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
# evaluation
from sklearn.preprocessing import normalize

# -------------------------
# Config / paths
# -------------------------
DATA_DIR = "./instacart_data"  # <- 修改為你的路徑
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
AISLES_CSV = os.path.join(DATA_DIR, "aisles.csv")
DEPTS_CSV = os.path.join(DATA_DIR, "departments.csv")
ORDERS_CSV = os.path.join(DATA_DIR, "orders.csv")
OP_PRIOR_CSV = os.path.join(DATA_DIR, "order_products__prior.csv")
OP_TRAIN_CSV = os.path.join(DATA_DIR, "order_products__train.csv")

# Experiment params
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Dense retrieval config
SENT_TRANS_MODEL = "all-MiniLM-L6-v2"  # 可改成其它 sentence-transformers model
EMB_DIM = 384  # for the above model

# Transformer training config (if you want to train)
TOP_N_ITEMS = 5000  # only model top-N frequent items (reduce classification head size)
BATCH_SIZE = 32
EPOCHS = 3
MAX_SEQ_LEN = 128
LR = 2e-5

# Evaluation cutoffs
EVAL_K = [5, 10, 20]

# User group thresholds
COLD_TH = 5
WARM_TH_LOW, WARM_TH_HIGH = 5, 20
ACTIVE_TH = 50

# Subsample (if you want faster testing)
SUBSAMPLE_USERS = None  # e.g., 10000 or None for all

# -------------------------
# Utility: metrics
# -------------------------
def recall_at_k(recommended, ground_truth, k):
    """recommended: list of item ids (ordered)
       ground_truth: set of true item ids
    """
    rec_k = recommended[:k]
    hit = len(set(rec_k) & ground_truth)
    return hit / len(ground_truth) if len(ground_truth)>0 else 0.0

def dcg_at_k(recommended, ground_truth, k):
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        rel = 1.0 if item in ground_truth else 0.0
        denom = math.log2(i+2)  # i starts at 0 => position 1 has log2(2)
        dcg += (2**rel - 1) / denom
    return dcg

def idcg_at_k(ground_truth, k):
    # ideal DCG for binary relevance with |ground_truth| ones
    n_rel = min(len(ground_truth), k)
    idcg = sum((2**1 - 1)/math.log2(i+2) for i in range(n_rel))
    return idcg

def ndcg_at_k(recommended, ground_truth, k):
    idcg = idcg_at_k(ground_truth, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(recommended, ground_truth, k) / idcg

# -------------------------
# Load data and build histories
# -------------------------
print("Loading CSVs...")
orders = pd.read_csv(ORDERS_CSV)
op_prior = pd.read_csv(OP_PRIOR_CSV)
op_train = pd.read_csv(OP_TRAIN_CSV)
products = pd.read_csv(PRODUCTS_CSV)
aisles = pd.read_csv(AISLES_CSV)
departments = pd.read_csv(DEPTS_CSV)

# Merge metadata into products
products = products.merge(aisles, on="aisle_id", how="left").merge(departments, on="department_id", how="left")

# Build user histories from prior orders
# orders has order_id, user_id, eval_set (prior, train, test)
prior_orders = orders[orders.eval_set=="prior"]
train_orders = orders[orders.eval_set=="train"]  # this order's products are ground-truth
# sometimes test set exists too, but we'll use order_products__train as ground-truth per user

# Map order_id -> list of products for prior orders
print("Aggregating prior histories...")
prior_group = op_prior.groupby("order_id")["product_id"].apply(list).to_dict()
# Map order_id -> product list for train (ground truth)
train_group = op_train.groupby("order_id")["product_id"].apply(list).to_dict()

# Build user -> history (concatenate their prior orders in chronological order)
orders_sorted = orders.sort_values(["user_id", "order_number"])
user_hist = defaultdict(list)
user_prior_orders = prior_orders.sort_values(["user_id", "order_number"])
for _, row in user_prior_orders.iterrows():
    oid = row.order_id
    uid = row.user_id
    if oid in prior_group:
        user_hist[uid].extend(prior_group[oid])

# Build user -> ground_truth (from train set)
user_ground = {}
train_user_orders = train_orders[train_orders.order_id.isin(op_train.order_id.unique())]
for _, row in train_user_orders.iterrows():
    if row.order_id in train_group:
        user_ground[row.user_id] = set(train_group[row.order_id])

# Filter users with at least 1 prior and having ground truth
common_users = [u for u in user_ground.keys() if len(user_hist[u])>0]
print(f"Users with history + ground-truth: {len(common_users)}")
if SUBSAMPLE_USERS:
    np.random.seed(SEED)
    common_users = list(np.random.choice(common_users, size=min(SUBSAMPLE_USERS, len(common_users)), replace=False))

# -------------------------
# Create user groups
# -------------------------
user_groups = {"cold": [], "warm": [], "active": []}
for u in common_users:
    n_inter = len(user_hist[u])
    if n_inter < COLD_TH:
        user_groups["cold"].append(u)
    elif WARM_TH_LOW <= n_inter <= WARM_TH_HIGH:
        user_groups["warm"].append(u)
    elif n_inter >= ACTIVE_TH:
        user_groups["active"].append(u)

print({k: len(v) for k,v in user_groups.items()})

# -------------------------
# Dense retrieval pipeline
# -------------------------
print("Preparing dense retrieval item texts...")
# item textual representation
products["text"] = products["product_name"].astype(str) + " | aisle: " + products["aisle"].astype(str) + " | dept: " + products["department"].astype(str)
product_id_to_text = dict(zip(products.product_id, products.text))
product_ids = products.product_id.values.astype(int)

# Load sentence-transformers
print(f"Loading sentence-transformers model: {SENT_TRANS_MODEL}")
sbert = SentenceTransformer(SENT_TRANS_MODEL)

# Compute item embeddings (may take time)
ITEM_EMB_PATH = os.path.join(DATA_DIR, "item_embeddings.npy")
ITEM_ORDER_PATH = os.path.join(DATA_DIR, "item_ids.npy")
if os.path.exists(ITEM_EMB_PATH) and os.path.exists(ITEM_ORDER_PATH):
    item_embeddings = np.load(ITEM_EMB_PATH)
    item_ids_array = np.load(ITEM_ORDER_PATH)
    print("Loaded saved item embeddings.")
else:
    texts = [product_id_to_text[int(pid)] for pid in product_ids]
    item_embeddings = sbert.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=64)
    # normalize
    item_embeddings = normalize(item_embeddings, axis=1)
    item_ids_array = product_ids
    np.save(ITEM_EMB_PATH, item_embeddings)
    np.save(ITEM_ORDER_PATH, item_ids_array)
    print("Computed and saved item embeddings.")

# Build FAISS index for fast retrieval
print("Building FAISS index...")
index = faiss.IndexFlatIP(item_embeddings.shape[1])  # inner product for cosine (embeddings normalized)
index.add(item_embeddings)
print("FAISS index ready.")

def dense_recommend_for_user(user_id, top_k=100):
    history_pids = user_hist[user_id]
    # compute user embedding = mean of item embeddings in history (skip items not in product list)
    idx_map = {pid: i for i,pid in enumerate(item_ids_array)}
    hist_idxs = [idx_map[pid] for pid in history_pids if pid in idx_map]
    if len(hist_idxs) == 0:
        return []  # no candidate
    user_emb = item_embeddings[hist_idxs].mean(axis=0)
    user_emb = user_emb / np.linalg.norm(user_emb)
    user_emb = user_emb.reshape(1, -1).astype(np.float32)
    D, I = index.search(user_emb, top_k+len(history_pids))  # fetch extra to filter history
    rec_ids = []
    for idx in I[0]:
        pid = int(item_ids_array[idx])
        if pid in history_pids:
            continue
        rec_ids.append(pid)
        if len(rec_ids) >= top_k:
            break
    return rec_ids

# Evaluate dense retrieval by groups
print("Evaluating dense retrieval...")
dense_results = {}
for group_name, users in user_groups.items():
    if len(users) == 0:
        continue
    recs_per_k = {k: [] for k in EVAL_K}
    ndcgs_per_k = {k: [] for k in EVAL_K}
    for u in tqdm(users, desc=f"Group {group_name}"):
        ground = user_ground[u]
        recs = dense_recommend_for_user(u, top_k=max(EVAL_K))
        if len(recs)==0:
            continue
        for k in EVAL_K:
            recs_k = recs[:k]
            recs_per_k[k].append(recall_at_k(recs_k, ground, k))
            ndcgs_per_k[k].append(ndcg_at_k(recs_k, ground, k))
    dense_results[group_name] = {
        "recall": {k: np.mean(recs_per_k[k]) if len(recs_per_k[k])>0 else 0.0 for k in EVAL_K},
        "ndcg": {k: np.mean(ndcgs_per_k[k]) if len(ndcgs_per_k[k])>0 else 0.0 for k in EVAL_K},
    }

print("Dense retrieval results (means):")
for g, res in dense_results.items():
    print(g)
    print("  Recall:", res["recall"])
    print("  NDCG:  ", res["ndcg"])

# -------------------------
# Transformer-based next-item classifier (optional training)
# -------------------------
# We'll limit classification to the TOP_N_ITEMS most frequent items in training labels.
# Build item frequency and mapping
print("Preparing data for BERT next-item classifier (top-N items)...")
all_train_labels = []
for uid, gt in user_ground.items():
    all_train_labels.extend(list(gt))
freq = Counter(all_train_labels)
most_common = [pid for pid, _ in freq.most_common(TOP_N_ITEMS)]
item2idx = {pid: idx for idx, pid in enumerate(most_common)}
idx2item = {idx: pid for pid, idx in item2idx.items()}

# Create dataset for users where ground truth items are in top-N
train_examples = []
for uid in common_users:
    gt = user_ground.get(uid, set())
    gt_in_top = set([g for g in gt if g in item2idx])
    if len(gt_in_top) == 0:
        continue
    # build sequence text from last N items in history (or all)
    hist = user_hist[uid]
    # represent sequence as concatenated product names (or product ids as tokens)
    seq_names = [product_id_to_text.get(pid, "") for pid in hist[-50:]]  # keep last 50
    seq_text = " [SEP] ".join(seq_names)
    # for multi-label ground truth we'll pick one positive per example for classification simplicity:
    pos_pid = list(gt_in_top)[0]
    train_examples.append((seq_text, item2idx[pos_pid], uid))  # store uid for potential grouping

print(f"Training examples for BERT classifier (labels in top-{TOP_N_ITEMS}): {len(train_examples)}")

class NextItemDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_SEQ_LEN):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        seq_text, label, uid = self.examples[idx]
        tok = self.tokenizer(seq_text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in tok.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# If you want to train the classifier: uncomment training block. (Warning: can be slow on CPU)
TRAIN_BERT = False  # set True to train
if TRAIN_BERT and len(train_examples)>0:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(item2idx))
    dataset = NextItemDataset(train_examples, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = AdamW(model.parameters(), lr=LR)
    total_steps = len(loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)
    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
    torch.save(model.state_dict(), os.path.join(DATA_DIR, "bert_next_item.pt"))
    print("BERT model saved.")

# Evaluation function for BERT model inference
def bert_recommend_for_user(model, tokenizer, user_id, top_k=100):
    hist = user_hist[user_id]
    seq_names = [product_id_to_text.get(pid, "") for pid in hist[-50:]]
    seq_text = " [SEP] ".join(seq_names)
    tok = tokenizer(seq_text, truncation=True, padding='max_length', max_length=MAX_SEQ_LEN, return_tensors="pt")
    device = next(model.parameters()).device
    tok = {k: v.to(device) for k,v in tok.items()}
    model.eval()
    with torch.no_grad():
        out = model(**tok)
        logits = out.logits.squeeze(0).cpu().numpy()
    # logits correspond to top-N items
    ranked_idx = np.argsort(-logits)[:top_k*2]
    recs = []
    for ridx in ranked_idx:
        pid = idx2item[ridx]
        if pid in hist:  # filter
            continue
        recs.append(pid)
        if len(recs) >= top_k:
            break
    return recs

# If you trained BERT and want to evaluate:
bert_results = {}
if TRAIN_BERT:
    # load model (if not in memory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(item2idx))
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "bert_next_item.pt"), map_location=device))
    model.to(device)
    for group_name, users in user_groups.items():
        if len(users) == 0:
            continue
        recs_per_k = {k: [] for k in EVAL_K}
        ndcgs_per_k = {k: [] for k in EVAL_K}
        for u in tqdm(users, desc=f"BERT Group {group_name}"):
            ground = user_ground[u]
            # only evaluate if ground has at least one item inside top-N
            if len([g for g in ground if g in item2idx])==0:
                continue
            recs = bert_recommend_for_user(model, tokenizer, u, top_k=max(EVAL_K))
            for k in EVAL_K:
                recs_k = recs[:k]
                recs_per_k[k].append(recall_at_k(recs_k, ground, k))
                ndcgs_per_k[k].append(ndcg_at_k(recs_k, ground, k))
        bert_results[group_name] = {
            "recall": {k: np.mean(recs_per_k[k]) if len(recs_per_k[k])>0 else 0.0 for k in EVAL_K},
            "ndcg": {k: np.mean(ndcgs_per_k[k]) if len(ndcgs_per_k[k])>0 else 0.0 for k in EVAL_K},
        }
    print("BERT results (means):")
    for g, res in bert_results.items():
        print(g)
        print("  Recall:", res["recall"])
        print("  NDCG:  ", res["ndcg"])

# -------------------------
# Example: print a few example recommendations
# -------------------------
print("\nExample dense recommendations (first 5 users in each group):")
for g, users in user_groups.items():
    for u in users[:5]:
        recs = dense_recommend_for_user(u, top_k=10)
        print(f"Group {g} user {u}: top-10 recommendations: {recs} -- ground: {list(user_ground[u])[:10]}")
    print("----")

# Save evaluation results
import json
with open(os.path.join(DATA_DIR, "dense_results.json"), "w") as f:
    json.dump(dense_results, f, indent=2)
if TRAIN_BERT:
    with open(os.path.join(DATA_DIR, "bert_results.json"), "w") as f:
        json.dump(bert_results, f, indent=2)

print("Done.")
