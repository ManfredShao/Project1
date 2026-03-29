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
    num_simulations = 100
    while len(S1) + len(S2) < args.budget:
        h1_avg,h2_avg = [0.0]*n, [0.0]*n
        a1 = I1.union(S1)
        a2 = I2.union(S2)

        for j in range(num_simulations):
            live_g1 = sample_graph(n, graph, 1)
            live_g2 = sample_graph(n, graph, 2)

            a1_base = bfs(live_g1, a1)
            a2_base = bfs(live_g2, a2)

            r1 = get_exposured(graph, a1_base)
            r2 = get_exposured(graph, a2_base)

            for i in range(n):
                if i in a1_base:
                    delta_r1 = set()
                else:
                    delta_a1 = bfs(live_g1, {i}, blocked_set=a1_base)
                    delta_r1 = get_exposured(graph, delta_a1) - r1

                h1_score = len(delta_r1.intersection(r2)) - len(delta_r1.difference(r2))
                h1_avg[i] += h1_score / num_simulations

                if i in a2_base:
                    delta_r2 = set()
                else:
                    delta_a2 = bfs(live_g2, {i}, blocked_set=a2_base)
                    delta_r2 = get_exposured(graph, delta_a2) - r2

                h2_score = len(delta_r2.intersection(r1)) - len(delta_r2.difference(r1))
                h2_avg[i] += h2_score / num_simulations

        valid_candidates = [i for i in range(n) if i not in a1 and i not in a2]

        if not valid_candidates:
            break

        best_i1 = max(valid_candidates, key=lambda i: h1_avg[i])
        best_i2 = max(valid_candidates, key=lambda i: h2_avg[i])

        if h1_avg[best_i1] >= h2_avg[best_i2]:
            S1.add(best_i1)
        else:
            S2.add(best_i2)

    end_time = time.time()

    print(f"Heuristic time: {end_time - start_time:.2f} s")

    with open(args.balanced_seeds, 'w') as f:
        f.write(f"{len(S1)} {len(S2)}\n")
        for seed in S1:
            f.write(f"{seed}\n")
        for seed in S2:
            f.write(f"{seed}\n")

if __name__ == '__main__':
    main()
