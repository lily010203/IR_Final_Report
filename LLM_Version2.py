import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import math
from huggingface_hub import InferenceClient
import time
import random
import re

# ==========================================
# 1. 資料載入與預處理 (Data Preprocessing)

def load_and_process_data(base_path='./instacart/', n_per_group=20):
    print("Loading data...")
    orders = pd.read_csv(f'{base_path}orders.csv')
    order_products = pd.read_csv(f'{base_path}order_products__prior.csv')
    products = pd.read_csv(f'{base_path}products.csv')
    
    id_to_name = dict(zip(products['product_id'], products['product_name']))
    orders = orders[orders['eval_set'] == 'prior']

    # 計算全域熱門商品 (Top 50) 
    print("Calculating global popular items...")
    popular_ids = order_products['product_id'].value_counts().head(50).index.tolist()
    global_popular_items = [id_to_name[pid] for pid in popular_ids]

    print("Analyzing user activity...")
    user_counts = orders.groupby('user_id').size()
    cold_ids = user_counts[user_counts < 5].index.values
    warm_ids = user_counts[(user_counts >= 5) & (user_counts <= 20)].index.values
    active_ids = user_counts[user_counts > 20].index.values

    # 抽樣邏輯
    selected_users = []
    import numpy as np
    if len(cold_ids) > 0: selected_users.extend(np.random.choice(cold_ids, size=min(len(cold_ids), n_per_group), replace=False))
    if len(warm_ids) > 0: selected_users.extend(np.random.choice(warm_ids, size=min(len(warm_ids), n_per_group), replace=False))
    if len(active_ids) > 0: selected_users.extend(np.random.choice(active_ids, size=min(len(active_ids), n_per_group), replace=False))
    
    orders = orders[orders['user_id'].isin(selected_users)]

    print("Merging data...")
    merged = pd.merge(orders, order_products, on='order_id')
    merged = merged.sort_values(['user_id', 'order_number', 'add_to_cart_order'])
    
    dataset = []
    print("Constructing sequences...")
    final_counts = merged.groupby('user_id')['order_number'].max()

    for uid, group in tqdm(merged.groupby('user_id')):
        # 該用戶所有買過的商品 (去重)
        user_history_set = list(set([id_to_name.get(pid, "Unknown") for pid in group['product_id'].values]))
        
        product_seq = [id_to_name.get(pid, "Unknown") for pid in group['product_id'].values]
        if len(product_seq) < 2: continue
            
        target_item = product_seq[-1]
        history_seq = product_seq[:-1][-10:]
        
        u_count = final_counts.get(uid, 0)
        u_group = 'Cold-Start' if u_count < 5 else ('Warm' if u_count <= 20 else 'Active')
        
        # --- 構建候選集 (Candidates) ---
        # 候選集 = (用戶過去買過的所有東西) + (全域熱門商品)
        candidates = list(set(user_history_set + global_popular_items))
        
        dataset.append({
            'user_id': uid,
            'group': u_group,
            'history': history_seq,
            'target': target_item,
            'candidates': candidates # 把這個傳給模型
        })
    
    return pd.DataFrame(dataset), list(id_to_name.values())


# 2. LLM 推薦模型 HFRecommender
class HFRecommender:
    def __init__(self, api_token, all_products, top_k=10): 
        self.client = InferenceClient(token=api_token)
        self.all_products = all_products
        self.top_k = top_k
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"

    def predict(self, history, candidates, retries=5): 
        history_str = ", ".join(history)
        random.shuffle(candidates)
        candidates_str = ", ".join(candidates[:80]) 
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a grocery recommendation assistant. "
                    f"Select exactly {self.top_k} items from the provided Candidate List that the user is most likely to buy next. "
                    "Prioritize items the user has bought frequently in the past. "
                    "Output ONLY the selected item names separated by commas."
                )
            },
            {
                "role": "user", 
                "content": (
                    f"User History: {history_str}\n"
                    f"Candidate List: {candidates_str}\n"
                    "Recommendation:"
                )
            }
        ]
        
        for attempt in range(retries):
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.model_id,
                    max_tokens=200,
                    temperature=0.1
                )
                
                predicted_text = response.choices[0].message.content.strip()
                clean_text = re.sub(r'[0-9]+\.|-|\n|\[|\]|\*', '', predicted_text) 
                
                recommendations = [item.strip() for item in clean_text.split(',')]
                recommendations = [r for r in recommendations if len(r) > 1]
                
                # 補齊不足
                if len(recommendations) < self.top_k:
                    pool = [c for c in candidates if c not in recommendations]
                    needed = self.top_k - len(recommendations)
                    if len(pool) >= needed:
                        recommendations.extend(random.sample(pool, needed))
                    else:
                        recommendations.extend(pool)
                
                return recommendations[:self.top_k]

            except Exception as e:
                # 指數退避策略：2, 4, 8, 16, 32 秒
                wait_time = 2 ** (attempt + 1)
                print(f"API Busy, waiting {wait_time}s... (Attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
        
        # 真的運氣太差才到這
        print("Fallback to candidate random (Network Issue).")
        return random.sample(candidates[:self.top_k*2], self.top_k)
    
# 3. 評估指標 (Evaluation Metrics)
def calculate_metrics(recommendations, ground_truth, k=20):
    """
    寬鬆版評估指標：
    1. 忽略大小寫
    2. 允許部分匹配 (例如 'Banana' 可以匹配 'Organic Banana')
    """
    # 統一轉小寫並去除空白
    ground_truth = ground_truth.lower().strip()
    preds = [str(r).lower().strip() for r in recommendations[:k]]
    
    recall = 0
    rank = -1
    
    # 檢查是否命中
    for i, pred in enumerate(preds):
        # 1. 完全相同
        if pred == ground_truth:
            rank = i
            break
        # 2. 互相包含 (寬鬆判定)
        # 如果預測是 "banana" 且答案是 "organic banana" -> 算對
        # 如果預測是 "organic banana" 且答案是 "banana" -> 算對
        if len(pred) > 2 and len(ground_truth) > 2: # 避免太短的誤判
            if pred in ground_truth or ground_truth in pred:
                rank = i
                break
    
    # 計算分數
    if rank != -1:
        recall = 1
        ndcg = 1 / math.log2(rank + 2)
    else:
        recall = 0
        ndcg = 0
        
    return recall, ndcg

# 4. 主程式 (Main Execution)
def main():
    MY_HF_TOKEN = "hf_XXXXXXXXXXXXXXX" 
    # 1. 準備資料
    df_test, all_product_names = load_and_process_data(base_path='./instacart/', n_per_group=100)
    
    RECOMMENDATION_K = 10
    recommender = HFRecommender(api_token=MY_HF_TOKEN, all_products=all_product_names, top_k=RECOMMENDATION_K)
    
    results = []
    print("Running evaluation...")
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        history = row['history']
        candidates = row['candidates'] # 取得候選集
        ground_truth = row['target']
        group = row['group']
        
        # 傳入 candidates
        preds = recommender.predict(history, candidates)        
        # 計算指標
        recall, ndcg = calculate_metrics(preds, ground_truth, k=RECOMMENDATION_K)
        
        results.append({
            'group': group,
            'recall': recall,
            'ndcg': ndcg
        })
        
    # 4. 分析結果
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print(" FINAL REPORT: Performance by User Group")
    print("="*40)
    
    # 依群組計算平均指標
    summary = results_df.groupby('group').agg({
        'recall': 'mean',
        'ndcg': 'mean',
        'group': 'count' # 顯示樣本數
    }).rename(columns={'group': 'sample_count'})
    
    print(summary)
    
    # 儲存結果
    summary.to_csv('evaluation_results.csv')
    print("\nResults saved to evaluation_results.csv")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"檔案找不到錯誤: {e}")