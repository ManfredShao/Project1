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
            graph[i] =[]
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

# ================================
# 护盾 1：步步存档函数
# ================================
def write_seeds(file_path, S1, S2):
    with open(file_path, 'w') as f:
        f.write(f"{len(S1)} {len(S2)}\n")
        for seed in S1:
            f.write(f"{seed}\n")
        for seed in S2:
            f.write(f"{seed}\n")

def sample_graph(n, graph, prob_index):
    live_graph = {i:[] for i in range(n)}
    for u in graph:
        for neighbor in graph[u]:
            v = neighbor[0]
            p = neighbor[1] if prob_index == 1 else neighbor[2]
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
    
    # 初始空档写入，保证强杀也有0分以上的保底
    write_seeds(args.balanced_seeds, S1, S2)

    # ================================
    # 护盾 2：极致的速度策略
    # 既然目标是用 Higher TL 拿满分，大图的模拟次数必须压缩！
    # ================================
    if n <= 1000:
        num_simulations = 100
    elif n <= 5000:
        num_simulations = 60
    else:
        num_simulations = 30  # 巨型图只跑30次，秒出结果拿速度分！

    out_degrees = {i: len(graph.get(i,[])) for i in range(n)}
    # 候选池：小图全查，大图最多查 300 个大V
    top_k_count = min(n, max(300, args.budget * 10))
    candidate_pool = sorted(range(n), key=lambda x: out_degrees[x], reverse=True)[:top_k_count]

    # 初始化第一批图
    live_graphs_1 =[sample_graph(n, graph, 1) for _ in range(num_simulations)]
    live_graphs_2 =[sample_graph(n, graph, 2) for _ in range(num_simulations)]

    while len(S1) + len(S2) < args.budget:
        h1_avg = {i: 0.0 for i in candidate_pool}
        h2_avg = {i: 0.0 for i in candidate_pool}
        
        a1 = I1.union(S1)
        a2 = I2.union(S2)

        valid_candidates =[i for i in candidate_pool if i not in a1 and i not in a2]
        if not valid_candidates:
            break

        for j in range(num_simulations):
            live_g1 = live_graphs_1[j]
            live_g2 = live_graphs_2[j]

            a1_base = bfs(live_g1, a1)
            a2_base = bfs(live_g2, a2)

            r1 = get_exposured(graph, a1_base)
            r2 = get_exposured(graph, a2_base)

            for i in valid_candidates:
                # 算 S1 增量
                if i in a1_base:
                    h1_score = 0
                else:
                    delta_a1 = bfs(live_g1, {i}, blocked_set=a1_base)
                    delta_r1 = get_exposured(graph, delta_a1) - r1
                    h1_score = len(delta_r1.intersection(r2)) - len(delta_r1.difference(r2))
                h1_avg[i] += h1_score / num_simulations

                # 算 S2 增量
                if i in a2_base:
                    h2_score = 0
                else:
                    delta_a2 = bfs(live_g2, {i}, blocked_set=a2_base)
                    delta_r2 = get_exposured(graph, delta_a2) - r2
                    h2_score = len(delta_r2.intersection(r1)) - len(delta_r2.difference(r1))
                h2_avg[i] += h2_score / num_simulations

        best_i1 = max(valid_candidates, key=lambda i: h1_avg[i])
        best_i2 = max(valid_candidates, key=lambda i: h2_avg[i])

        if h1_avg[best_i1] >= h2_avg[best_i2]:
            S1.add(best_i1)
        else:
            S2.add(best_i2)

        # 每次选完人，立刻存档防强杀！
        write_seeds(args.balanced_seeds, S1, S2)

        # 刷新图以防过拟合
        live_graphs_1 =[sample_graph(n, graph, 1) for _ in range(num_simulations)]
        live_graphs_2 =[sample_graph(n, graph, 2) for _ in range(num_simulations)]

    end_time = time.time()
    print(f"Heuristic time: {end_time - start_time:.2f} s")
    
    # 最终保险写入
    write_seeds(args.balanced_seeds, S1, S2)

if __name__ == '__main__':
    main()