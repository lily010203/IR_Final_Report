import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Device
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# =========================================================
# Config
# =========================================================
DATA_DIR = "./instacart_data"
MAX_LEN = 50
MAX_EVAL_USERS = 500          # 🔥 先小，確定會動再調大
LOG_EVERY = 100               # 每 N user 印一次 log

# =========================================================
# Load Dataset
# =========================================================
print("[INFO] Loading Instacart dataset...")

orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
order_products = pd.read_csv(f"{DATA_DIR}/order_products__prior.csv")
products = pd.read_csv(f"{DATA_DIR}/products.csv")
aisles = pd.read_csv(f"{DATA_DIR}/aisles.csv")
departments = pd.read_csv(f"{DATA_DIR}/departments.csv")

products = products.merge(aisles, on="aisle_id") \
                   .merge(departments, on="department_id")

products["text"] = (
    products["product_name"] + " " +
    products["aisle"] + " " +
    products["department"]
)

user_orders = (
    orders.merge(order_products, on="order_id")
          .merge(products[["product_id", "text"]], on="product_id")
          .sort_values(["user_id", "order_number"])
)

print("[INFO] Dataset loaded.")

# =========================================================
# User grouping
# =========================================================
user_counts = user_orders.groupby("user_id")["product_id"].count()

groups = {
    "cold": user_counts[user_counts < 5].index,
    "warm": user_counts[(user_counts >= 5) & (user_counts < 20)].index,
    "active": user_counts[user_counts >= 50].index
}

for g in groups:
    print(f"[INFO] {g} users: {len(groups[g])}")

# =========================================================
# Dense Retrieval Model
# =========================================================
print("[INFO] Loading Sentence-BERT...")
dense_model = SentenceTransformer(
    "sentence-transformers/msmarco-distilbert-base-v4"
).to(device)
dense_model.eval()

print("[INFO] Encoding item embeddings...")
item_embeddings = dense_model.encode(
    products["text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

pid2text = dict(zip(products["product_id"], products["text"]))

# =========================================================
# Build user sequences
# =========================================================
user_sequences = user_orders.groupby("user_id")["product_id"].apply(list)

all_pids = products["product_id"].unique().tolist()
pid2idx = {pid: i+1 for i, pid in enumerate(all_pids)}
idx2pid = {i: pid for pid, i in pid2idx.items()}
n_items = len(pid2idx)

# =========================================================
# Dense candidate function (SAFE + FAST)
# =========================================================
def dense_candidates(uid, topk=200):
    seq = user_sequences[uid][:-1]  # exclude GT
    if len(seq) == 0:
        return []

    texts = [pid2text[p] for p in seq if p in pid2text]

    with torch.no_grad():
        user_emb = dense_model.encode(
            " [SEP] ".join(texts),
            normalize_embeddings=True
        ).reshape(1, -1)

    sims = cosine_similarity(user_emb, item_embeddings)[0]
    idx = np.argsort(-sims)[:topk]
    return list(dict.fromkeys(products.iloc[idx]["product_id"].tolist()))

# =========================================================
# SASRec Dataset
# =========================================================
class SASRecDataset(Dataset):
    def __init__(self, sequences):
        self.seqs, self.targets = [], []

        for seq in sequences:
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                self.seqs.append(seq[max(0, i-MAX_LEN):i])
                self.targets.append(seq[i])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        seq = [0]*(MAX_LEN - len(seq)) + [pid2idx[p] for p in seq]
        return torch.LongTensor(seq), torch.LongTensor([pid2idx[self.targets[idx]]])

# =========================================================
# SASRec Model
# =========================================================
class SASRec(torch.nn.Module):
    def __init__(self, n_items):
        super().__init__()
        self.item_emb = torch.nn.Embedding(n_items + 1, 64, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(MAX_LEN, 64)

        layer = torch.nn.TransformerEncoderLayer(
            d_model=64, nhead=2, dim_feedforward=256, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(layer, 2)
        self.fc = torch.nn.Linear(64, n_items + 1)

    def forward(self, seq):
        pos = torch.arange(MAX_LEN, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(MAX_LEN, MAX_LEN, device=seq.device), 1).bool()
        h = self.encoder(x, mask)
        return self.fc(h[:, -1])

# =========================================================
# Train SASRec
# =========================================================
print("[INFO] Training SASRec...")
train_users = user_counts[user_counts >= 5].index
dataset = SASRecDataset(user_sequences.loc[train_users])
loader = DataLoader(dataset, batch_size=128, shuffle=True)

sasrec_model = SASRec(n_items).to(device)
optimizer = torch.optim.Adam(sasrec_model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    sasrec_model.train()
    total_loss = 0
    for seq, target in loader:
        seq, target = seq.to(device), target.squeeze().to(device)
        optimizer.zero_grad()
        loss = loss_fn(sasrec_model(seq), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[INFO] SASRec Epoch {epoch+1} Loss={total_loss/len(loader):.4f}")

# =========================================================
# Evaluation (WITH LOG)
# =========================================================
def ndcg(hit, k):
    return 1 / np.log2(k + 1) if hit else 0

sasrec_model.eval()
K = [5, 10, 20]
results = {}

print("[INFO] Start evaluation")

for gname, users in groups.items():
    print(f"\n[INFO] Evaluating {gname.upper()} users")
    users = list(users)[:MAX_EVAL_USERS]

    recalls = {k: [] for k in K}
    ndcgs = {k: [] for k in K}

    start_time = time.time()

    for i, uid in enumerate(users):
        if uid not in user_sequences or len(user_sequences[uid]) < 2:
            continue

        if i % LOG_EVERY == 0:
            print(f"[LOG] {gname} user {i}/{len(users)} (uid={uid})")

        gt = user_sequences[uid][-1]

        if gname == "cold":
            ranked = dense_candidates(uid)

        elif gname == "warm":
            cand = dense_candidates(uid)
            if not cand:
                continue
            seq = user_sequences[uid][:-1][-MAX_LEN:]
            seq = [0]*(MAX_LEN-len(seq)) + [pid2idx[p] for p in seq]
            with torch.no_grad():
                logits = sasrec_model(torch.LongTensor(seq).unsqueeze(0).to(device)).squeeze()
            ranked = sorted(cand, key=lambda p: logits[pid2idx[p]].item(), reverse=True)

        else:  # active
            seq = user_sequences[uid][:-1][-MAX_LEN:]
            seq = [0]*(MAX_LEN-len(seq)) + [pid2idx[p] for p in seq]
            with torch.no_grad():
                logits = sasrec_model(torch.LongTensor(seq).unsqueeze(0).to(device)).squeeze()
            top = torch.topk(logits, 200).indices.cpu().numpy()
            ranked = [idx2pid[i] for i in top if i in idx2pid]

        for k in K:
            hit = gt in ranked[:k]
            recalls[k].append(int(hit))
            ndcgs[k].append(ndcg(hit, k))

    elapsed = time.time() - start_time
    print(f"[INFO] {gname.upper()} done in {elapsed:.1f}s")

    results[gname] = {
        "Recall": {k: np.mean(recalls[k]) for k in K},
        "NDCG": {k: np.mean(ndcgs[k]) for k in K}
    }

print("\n=== FINAL RESULTS (Dense + SASRec) ===")
for g, m in results.items():
    print(f"\n{g.upper()}")
    print(pd.DataFrame(m).round(5))
