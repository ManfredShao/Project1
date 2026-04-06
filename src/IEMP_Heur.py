import random
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluator for Information Exposured Maximization')
    parser.add_argument('-n', '--network', type=str, required=True,
                        help='Path to the social network file')
    parser.add_argument('-i', '--initial_seeds', type=str, required=True,
                        help='Path to the initial seed set file (I1, I2)')
    parser.add_argument('-b', '--balanced_seeds', type=str, required=True,
                        help='Path to the balanced seed set file (S1, S2)')
    parser.add_argument('-k', '--budget', type=int, required=True,
                        help='Budget k')
    return parser.parse_args()

def read_network(file_path):
    graph = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return 0,{}
        
        n, m = map(int, lines[0].strip().split())

        for i in range(n):
            graph[i] = []

        for i in range(1,m+1):
            parts = lines[i].strip().split()
            u = int(parts[0])
            v = int(parts[1])
            p1 = float(parts[2])
            p2 = float(parts[3])
            graph[u].append((v, p1, p2))

    return n, graph

def read_seeds(file_path):
    set1, set2 = set(), set()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return set1, set2
        k1, k2 = map(int, lines[0].strip().split())

        for i in range(1, k1+1):
            set1.add(int(lines[i].strip()))
        for i in range(1, k2+1):
            set2.add(int(lines[k1+i].strip()))

    return set1, set2

def write_seeds(file_path, S1, S2):
    # Balanced seed set file should only contain *new* seeds (not initial I1/I2),
    # and must satisfy |S1|+|S2| <= k (checked by caller). This helper only serializes.
    with open(file_path, 'w') as f:
        f.write(f"{len(S1)} {len(S2)}\n")
        for seed in S1:
            f.write(f"{seed}\n")
        for seed in S2:
            f.write(f"{seed}\n")

def sample_graph(n, graph, prob_index):
    live_graph = {i: [] for i in range(n)}
    for u in graph:
        for neighbor in graph[u]:
            v = neighbor[0]
            if prob_index == 1:
                p = neighbor[1]
            else:
                p = neighbor[2]

            if random.random() <= p:
                live_graph[u].append(v)

    return live_graph

def bfs(live_graph, seeds, blocked_set=None):
    if blocked_set is None:
        blocked_set = set()
    
    queue = [s for s in seeds if s not in blocked_set]
    visited = set(queue)
    while queue:
        u = queue.pop()
        for v in live_graph[u]:
            if v not in visited and v not in blocked_set:
                visited.add(v)
                queue.append(v)
    return visited

def get_exposured(graph, activated):
    exposured = set(activated)
    for u in activated:
        for neighbor in graph[u]:
            v = neighbor[0]
            exposured.add(v)
    return exposured

def main():
    start_time = time.time()

    args = parse_args()
    n, graph = read_network(args.network)
    I1, I2 = read_seeds(args.initial_seeds)
    S1, S2 = set(), set()
    
    write_seeds(args.balanced_seeds, S1, S2)

    # Scale-aware MC budget: keep small graphs accurate, large graphs fast enough for TA time limits.
    if n <= 800:
        num_simulations = 120
    elif n <= 2500:
        num_simulations = 80
    elif n <= 8000:
        num_simulations = 40
    else:
        num_simulations = 20
    
    # ==========================================
    # 核心优化：基于概率出度的候选池缩减
    # 计算每个节点的 outgoing probability mass
    # ==========================================
    forbidden = set(I1) | set(I2)
    scores = [0.0] * n
    for u, outs in graph.items():
        if u not in forbidden:
            scores[u] = sum(p1 + p2 for v, p1, p2 in outs)
            
    # 只取概率质量最大的前若干候选人（与预算相关，且对大图更克制）
    top_k_count = min(n, max(200, args.budget * 12))
    candidate_pool = [u for u in range(n) if u not in forbidden]
    candidate_pool.sort(key=lambda x: scores[x], reverse=True)
    candidate_pool = candidate_pool[:top_k_count]

    # 主循环：贪心寻找 k 个种子
    while len(S1) + len(S2) < args.budget:
        # 只为候选池里的节点建立字典
        h1_avg = {i: 0.0 for i in candidate_pool}
        h2_avg = {i: 0.0 for i in candidate_pool}
        
        a1 = I1.union(S1)
        a2 = I2.union(S2)

        # 过滤掉本轮已经被选中的人
        valid_candidates = [i for i in candidate_pool if i not in a1 and i not in a2]

        if not valid_candidates:
            break

        # 蒙特卡洛评估（只针对 valid_candidates）
        for j in range(num_simulations):
            live_g1 = sample_graph(n, graph, 1)
            live_g2 = sample_graph(n, graph, 2)

            a1_base = bfs(live_g1, a1)
            a2_base = bfs(live_g2, a2)

            r1 = get_exposured(graph, a1_base)
            r2 = get_exposured(graph, a2_base)

            for i in valid_candidates:
                # 评估加给 S1 的增量
                if i in a1_base:
                    h1_score = 0
                else:
                    delta_a1 = bfs(live_g1, {i}, blocked_set=a1_base)
                    delta_r1 = get_exposured(graph, delta_a1) - r1
                    h1_score = len(delta_r1.intersection(r2)) - len(delta_r1.difference(r2))
                h1_avg[i] += h1_score / num_simulations

                # 评估加给 S2 的增量
                if i in a2_base:
                    h2_score = 0
                else:
                    delta_a2 = bfs(live_g2, {i}, blocked_set=a2_base)
                    delta_r2 = get_exposured(graph, delta_a2) - r2
                    h2_score = len(delta_r2.intersection(r1)) - len(delta_r2.difference(r1))
                h2_avg[i] += h2_score / num_simulations

        # 从候选池中选出最大收益
        best_i1 = max(valid_candidates, key=lambda i: h1_avg[i])
        best_i2 = max(valid_candidates, key=lambda i: h2_avg[i])

        # 决定加入哪个阵营
        if h1_avg[best_i1] >= h2_avg[best_i2]:
            S1.add(best_i1)
        else:
            S2.add(best_i2)

        # Defensive: never exceed budget
        if len(S1) + len(S2) > args.budget:
            # Remove the last-added seed to satisfy the required constraint.
            if best_i1 in S1 and best_i2 not in S2:
                S1.discard(best_i1)
            else:
                S2.discard(best_i2)

        # 步步存档，防御 SIGKILL
        write_seeds(args.balanced_seeds, S1, S2)

    end_time = time.time()
    print(f"Heuristic time: {end_time - start_time:.2f} s")
    
    # 最终确保写入
    write_seeds(args.balanced_seeds, S1, S2)

if __name__ == '__main__':
    main()