import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from tqdm import tqdm
import warnings
import random
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

edges_filepath = "Gowalla_edges.txt"
checkin_filepath = "Gowalla_totalCheckins.txt"

G = nx.Graph()
print("加载网络边数据...")
with open(edges_filepath) as f:
    for line in f:
        node1, node2 = line.strip().split()
        G.add_edge(str(node1), str(node2))
print(f"网络节点数: {G.number_of_nodes():,}, 边数: {G.number_of_edges():,}")

print("加载签到数据...")
checkin_df = pd.read_csv(
    checkin_filepath, 
    sep='\t',
    header=None,
    names=['user', 'time', 'lat', 'lon', 'location_id'],
    parse_dates=['time']
)
print(f"总签到记录数: {len(checkin_df):,}")

# ========== 2. 数据过滤 ==========
print("\n" + "=" * 60)
print("2. 数据过滤")
print("=" * 60)

active_users = set(str(uid) for uid in checkin_df['user'].unique())
print(f"签到用户数: {len(active_users):,}")

G_filtered = G.subgraph(active_users).copy()
print(f"过滤后网络节点数: {G_filtered.number_of_nodes():,}, 边数: {G_filtered.number_of_edges():,}")

# ========== 3. 编码用户和POI ==========
print("\n" + "=" * 60)
print("3. 编码用户和POI")
print("=" * 60)

user_encoder = LabelEncoder()
poi_encoder = LabelEncoder()

checkin_df['user_id'] = user_encoder.fit_transform(checkin_df['user'].values)
checkin_df['poi_id'] = poi_encoder.fit_transform(checkin_df['location_id'].values)

n_users = len(user_encoder.classes_)
n_pois = len(poi_encoder.classes_)
print(f"用户数: {n_users:,}, POI数: {n_pois:,}")

poi_location = checkin_df.groupby('poi_id')[['lat', 'lon']].mean().reset_index()
poi_coords = np.zeros((n_pois, 2), dtype=np.float32)
for _, row in poi_location.iterrows():
    poi_coords[int(row['poi_id'])] = [row['lat'], row['lon']]

# ========== 4. 划分训练集/测试集 ==========
print("\n" + "=" * 60)
print("4. 划分训练集/测试集")
print("=" * 60)

checkin_df = checkin_df.sort_values(['user', 'time'])
user_counts = checkin_df.groupby('user_id').size().reset_index(name='count')
user_counts['train_count'] = (user_counts['count'] * 0.8).astype(int)
user_counts_dict = user_counts.set_index('user_id')['train_count'].to_dict()

checkin_df['row_num'] = checkin_df.groupby('user_id').cumcount() + 1
split_array = np.full(len(checkin_df), 'test', dtype=object)
user_ids = checkin_df['user_id'].values
row_nums = checkin_df['row_num'].values
train_counts = np.array([user_counts_dict.get(uid, 0) for uid in user_ids])
is_train = row_nums <= train_counts
split_array[is_train] = 'train'
checkin_df['split'] = split_array

train_df = checkin_df[checkin_df['split'] == 'train'].copy()
test_df = checkin_df[checkin_df['split'] == 'test'].copy()
print(f"训练集签到数: {len(train_df):,}")
print(f"测试集签到数: {len(test_df):,}")
checkin_df.drop(['row_num', 'split'], axis=1, inplace=True)

# ========== 5. 构建用户历史 ==========
print("\n" + "=" * 60)
print("5. 构建用户历史POI集合")
print("=" * 60)

user_history = defaultdict(set)
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="构建历史"):
    user_history[row['user_id']].add(row['poi_id'])
print(f"有历史记录的用户数: {len(user_history):,}")

# ========== 6. 构建好友映射（加速） ==========
print("\n" + "=" * 60)
print("6. 构建好友映射")
print("=" * 60)

# 将图中好友关系转换为 {user_id: set(friend_ids)}
# 注意：G_filtered中的节点是字符串，user_encoder.transform需要原始user字符串
user_to_id = {str(u): i for i, u in enumerate(user_encoder.classes_)}
friend_map = defaultdict(set)
for u, v in tqdm(G_filtered.edges(), desc="构建好友映射"):
    u_id = user_to_id.get(u)
    v_id = user_to_id.get(v)
    if u_id is not None and v_id is not None:
        friend_map[u_id].add(v_id)
        friend_map[v_id].add(u_id)
print(f"有好友记录的用户数: {len(friend_map):,}")

# ========== 7. 计算好友的POI聚合 ==========
print("\n" + "=" * 60)
print("7. 聚合好友签到的POI")
print("=" * 60)

# 为了加速，只对采样用户计算好友POI
# 这里预先构建一个cache：{user_id: {poi_id: friend_count}}
friend_poi_cache = {}
SAMPLE_USERS_FOR_FRIEND = 5000  # 只对前5000个用户预计算
sample_users_for_friend = list(user_history.keys())[:SAMPLE_USERS_FOR_FRIEND]

for uid in tqdm(sample_users_for_friend, desc="聚合好友POI"):
    poi_weight = defaultdict(int)
    for friend in friend_map.get(uid, set()):
        for poi in user_history.get(friend, set()):
            poi_weight[poi] += 1
    # 只保留权重最高的50个POI（减少存储）
    if poi_weight:
        sorted_pois = sorted(poi_weight.items(), key=lambda x: x[1], reverse=True)[:50]
        friend_poi_cache[uid] = {poi: w for poi, w in sorted_pois}
print(f"预计算好友POI缓存大小: {len(friend_poi_cache)}")

def get_friend_pois(user_id, top_k=20):
    """获取好友签到的POI（带权重）"""
    if user_id in friend_poi_cache:
        items = friend_poi_cache[user_id].items()
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)[:top_k]
        return [poi for poi, _ in sorted_items]
    return []

# ========== 8. 构建POI-KD树 ==========
print("\n" + "=" * 60)
print("8. 构建POI索引")
print("=" * 60)

tree = KDTree(poi_coords)
print(f"KD树构建完成，包含 {n_pois:,} 个POI")

def find_nearby_pois(poi_id, k=50):
    coord = poi_coords[poi_id].reshape(1, -1)
    distances, indices = tree.query(coord, k=min(k+1, n_pois))
    return indices[0][1:], distances[0][1:]

# ========== 9. 计算热门POI ==========
print("\n" + "=" * 60)
print("9. 计算热门POI（冷启动）")
print("=" * 60)

poi_popularity = defaultdict(int)
for pois in tqdm(user_history.values(), desc="统计POI热度"):
    for poi in pois:
        poi_popularity[poi] += 1

HOT_POIS = [poi for poi, _ in sorted(poi_popularity.items(), key=lambda x: x[1], reverse=True)]
print(f"热门POI数量: {len(HOT_POIS)}")

def get_hot_pois(top_k=10, exclude=None):
    """获取热门POI，可选排除某些POI"""
    if exclude is None:
        return HOT_POIS[:top_k]
    result = [p for p in HOT_POIS if p not in exclude]
    return result[:top_k]

# ========== 10. 核心推荐：融合 地理 + 好友 + 热门 ==========
print("\n" + "=" * 60)
print("10. 融合推荐（地理 + 好友 + 热门）")
print("=" * 60)

def compute_geo_scores_for_candidates(user_id, candidate_pois, user_history, poi_coords, bandwidth=0.2):
    if user_id not in user_history or not user_history[user_id]:
        return np.zeros(len(candidate_pois)) if len(candidate_pois) > 0 else np.array([])
    if len(candidate_pois) == 0:
        return np.array([])
    
    hist_pois = list(user_history[user_id])
    if len(hist_pois) > 300:
        hist_pois = np.random.choice(hist_pois, size=300, replace=False).tolist()
    
    hist_coords = poi_coords[hist_pois]
    chunk_size = 2000
    all_scores = []
    
    for i in range(0, len(candidate_pois), chunk_size):
        chunk_pois = candidate_pois[i:i+chunk_size]
        poi_coords_candidates = poi_coords[chunk_pois]
        diff = hist_coords[:, np.newaxis, :] - poi_coords_candidates[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        scores = np.exp(-distances**2 / (2 * bandwidth**2))
        chunk_scores = np.mean(scores, axis=0)
        all_scores.extend(chunk_scores)
    return np.array(all_scores)

def recommend_fusion(user_id, user_history, poi_coords, friend_map, 
                     top_k=10, bandwidth=0.2, nearby_k=50, 
                     max_candidates=3000,
                     alpha_g=0.5, alpha_s=0.3, alpha_h=0.2):
    """
    融合推荐：地理(G) + 好友(S) + 热门(H)
    alpha_g + alpha_s + alpha_h = 1
    """
    user_id = int(user_id)
    
    # 冷启动：无历史 -> 热门POI
    if user_id not in user_history or not user_history[user_id]:
        return get_hot_pois(top_k), [1.0] * top_k
    
    hist_pois = list(user_history[user_id])
    user_pois_set = set(hist_pois)
    
    # --- 候选集生成（取并集） ---
    candidate_pool = set()
    
    # 1. 地理候选：从历史POI附近找
    n_seeds = min(len(hist_pois), 10)
    seeds = np.random.choice(hist_pois, size=n_seeds, replace=False)
    for seed in seeds:
        nearby, _ = find_nearby_pois(seed, k=nearby_k)
        candidate_pool.update(nearby.tolist())
    
    # 2. 好友候选：好友签到的POI
    friend_pois = get_friend_pois(user_id, top_k=20)
    candidate_pool.update(friend_pois)
    
    # 3. 热门候选：补充热门POI（确保候选集够大）
    hot_pois = get_hot_pois(top_k=30, exclude=user_pois_set)
    candidate_pool.update(hot_pois)
    
    # 排除用户已签到的
    candidate_pool = candidate_pool - user_pois_set
    
    # 限制候选数量
    candidate_list = list(candidate_pool)
    if len(candidate_list) > max_candidates:
        candidate_list = np.random.choice(candidate_list, size=max_candidates, replace=False).tolist()
    
    if len(candidate_list) == 0:
        return get_hot_pois(top_k), [1.0] * top_k
    
    # --- 计算三个得分 ---
    candidate_array = np.array(candidate_list)
    
    # G: 地理得分
    geo_scores = compute_geo_scores_for_candidates(
        user_id, candidate_array, user_history, poi_coords, bandwidth
    )
    
    # S: 好友得分（如果在缓存中，用缓存权重；否则为0）
    friend_scores = np.zeros(len(candidate_array))
    if user_id in friend_poi_cache:
        friend_weight = friend_poi_cache[user_id]
        for i, poi in enumerate(candidate_array):
            friend_scores[i] = friend_weight.get(poi, 0)
        # 归一化
        if friend_scores.max() > 0:
            friend_scores = friend_scores / friend_scores.max()
    
    # H: 热门得分（基于全局排名）
    hot_scores = np.zeros(len(candidate_array))
    hot_rank = {poi: idx for idx, poi in enumerate(HOT_POIS)}
    for i, poi in enumerate(candidate_array):
        rank = hot_rank.get(poi, len(HOT_POIS))
        hot_scores[i] = 1.0 / (rank + 1)  # 排名越前得分越高
    if hot_scores.max() > 0:
        hot_scores = hot_scores / hot_scores.max()
    
    # --- 融合 ---
    final_scores = (alpha_g * geo_scores + 
                    alpha_s * friend_scores + 
                    alpha_h * hot_scores)
    
    # --- 排序取Top-K ---
    top_indices = np.argsort(final_scores)[-top_k:][::-1]
    top_pois = candidate_array[top_indices].tolist()
    top_scores = final_scores[top_indices].tolist()
    
    return top_pois, top_scores

# ========== 11. 评估（软匹配） ==========
print("\n" + "=" * 60)
print("11. 评估融合推荐（软匹配）")
print("=" * 60)

def evaluate_fusion_soft(test_df, user_history, poi_coords, friend_map,
                         top_k=10, sample_size=200, distance_threshold=0.02,
                         bandwidth=0.2, nearby_k=50,
                         alpha_g=0.5, alpha_s=0.3, alpha_h=0.2):
    
    available_users = [u for u in test_df['user_id'].unique() if u in user_history]
    if len(available_users) > sample_size:
        sampled_users = np.random.choice(available_users, size=sample_size, replace=False)
    else:
        sampled_users = available_users
    
    test_user_poi = defaultdict(set)
    sampled_set = set(sampled_users)
    for _, row in test_df.iterrows():
        if row['user_id'] in sampled_set:
            test_user_poi[row['user_id']].add(row['poi_id'])
    
    precision_list = []
    recall_list = []
    hit_counts = []
    
    for user_id in tqdm(sampled_users, desc="融合推荐评估"):
        true_pois = test_user_poi.get(user_id, set())
        if len(true_pois) == 0:
            continue
        
        top_pois, _ = recommend_fusion(
            user_id, user_history, poi_coords, friend_map,
            top_k=top_k, bandwidth=bandwidth, nearby_k=nearby_k,
            alpha_g=alpha_g, alpha_s=alpha_s, alpha_h=alpha_h
        )
        
        hits = 0
        for rec_poi in top_pois:
            rec_coord = poi_coords[rec_poi]
            true_coords = poi_coords[list(true_pois)]
            distances = np.sqrt(np.sum((true_coords - rec_coord)**2, axis=1))
            if np.min(distances) <= distance_threshold:
                hits += 1
        
        precision_list.append(hits / top_k)
        recall_list.append(hits / len(true_pois) if len(true_pois) > 0 else 0)
        hit_counts.append(hits)
    
    print(f"\n融合推荐统计 (阈值={distance_threshold}):")
    print(f"  - 有效用户数: {len(precision_list)}")
    print(f"  - 平均命中数: {np.mean(hit_counts):.2f}/{top_k}")
    print(f"  - 命中率 > 0 的用户比例: {np.sum(np.array(hit_counts) > 0) / len(hit_counts):.2%}")
    
    return np.mean(precision_list), np.mean(recall_list), len(precision_list)

# ========== 12. 运行评估 ==========
print("\n" + "=" * 60)
print("12. 运行评估与对比")
print("=" * 60)

# 配置参数
SAMPLE_SIZE = 200
TOP_K = 10
DIST_THRESHOLD = 0.05

# 不同权重配置
configs = [
    # {"name": "纯地理 (G)", "alpha_g": 1.0, "alpha_s": 0.0, "alpha_h": 0.0},
    # {"name": "地理+好友 (G+S)", "alpha_g": 0.6, "alpha_s": 0.4, "alpha_h": 0.0},
    # {"name": "地理+热门 (G+H)", "alpha_g": 0.6, "alpha_s": 0.0, "alpha_h": 0.4},
    # {"name": "地理+好友+热门 (G+S+H)", "alpha_g": 0.5, "alpha_s": 0.3, "alpha_h": 0.2},
    {"name": "纯热门 (H)", "alpha_g": 0.0, "alpha_s": 1.0, "alpha_h": 0.0},
    {"name": "纯好友 (S)", "alpha_g": 0.0, "alpha_s": 0.0, "alpha_h": 1.0},
]

results = []
for cfg in configs:
    print(f"\n--- 测试配置: {cfg['name']} ---")
    prec, rec, n = evaluate_fusion_soft(
        test_df, user_history, poi_coords, friend_map,
        top_k=TOP_K, sample_size=SAMPLE_SIZE, 
        distance_threshold=DIST_THRESHOLD,
        bandwidth=0.2, nearby_k=50,
        alpha_g=cfg["alpha_g"], 
        alpha_s=cfg["alpha_s"], 
        alpha_h=cfg["alpha_h"]
    )
    results.append((cfg["name"], prec, rec))

# 打印对比结果
print("\n" + "=" * 60)
print("对比结果汇总")
print("=" * 60)
print(f"{'配置':<25} {'Precision@10':<15} {'Recall@10':<15}")
print("-" * 60)
for name, prec, rec in results:
    print(f"{name:<25} {prec:<15.4f} {rec:<15.4f}")

# ========== 13. 示例推荐 ==========
print("\n" + "=" * 60)
print("13. 示例推荐展示")
print("=" * 60)

sample_user = list(user_history.keys())[0]
print(f"为用户 {sample_user} 推荐（历史POI: {len(user_history[sample_user])}个）")

top_pois, scores = recommend_fusion(
    sample_user, user_history, poi_coords, friend_map,
    top_k=5, bandwidth=0.2, nearby_k=50,
    alpha_g=0.5, alpha_s=0.3, alpha_h=0.2
)

print("\n推荐的Top-5 POI:")
for i, (poi, score) in enumerate(zip(top_pois, scores), 1):
    lat, lon = poi_coords[poi]
    print(f"  {i}. POI {poi}, 坐标({lat:.4f},{lon:.4f}), 得分={score:.4f}")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)