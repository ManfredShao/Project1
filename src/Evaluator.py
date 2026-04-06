import random
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluator for Information Exposure Maximization')
    parser.add_argument('-n', '--network', type=str, required=True,
                        help='Path to the social network file')
    parser.add_argument('-i', '--initial_seeds', type=str, required=True,
                        help='Path to the initial seed set file (I1, I2)')
    parser.add_argument('-b', '--balanced_seeds', type=str, required=True,
                        help='Path to the balanced seed set file (S1, S2)')
    parser.add_argument('-k', '--budget', type=int, required=True,
                        help='Budget k')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to output the objective value')
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

def get_exposure(graph, activated):
    exposure = set(activated)
    for u in activated:
        for neighbor in graph[u]:
            v = neighbor[0]
            exposure.add(v)
    return exposure

def mc_prepare(graph, seeds, prob_index):
    activated = set(seeds)
    queue = set(seeds)

    while queue:
        u = queue.pop()
        for neighbor in graph[u]:
            v = neighbor[0]
            if prob_index == 1:
                p = neighbor[1]
            else:
                p = neighbor[2]

            if v not in activated:
                if random.random() <= p:
                    activated.add(v)
                    queue.add(v)

    return activated

def evaluate_monte_carlo(n, graph, u1, u2, num_simulations=1000):
    total_score = 0.0
    for _ in range(num_simulations):
        a1 = mc_prepare(graph, u1, 1)
        a2 = mc_prepare(graph, u2, 2)

        r1 = get_exposure(graph, a1)
        r2 = get_exposure(graph, a2)

        symmetric_diff = r1 ^ r2
        score = n - len(symmetric_diff)
        total_score += score

    return total_score / num_simulations

def main():
    start_time = time.time()

    args = parse_args()
    n, graph = read_network(args.network)
    I1, I2 = read_seeds(args.initial_seeds)
    B1, B2 = read_seeds(args.balanced_seeds)

    if len(B1) + len(B2) > args.budget:
        print("Error: Budget exceeded")
        with open(args.output, 'w') as f:
            f.write("0.0/n")
        return

    U1 = I1.union(B1)
    U2 = I2.union(B2)

    value = evaluate_monte_carlo(n, graph, U1, U2, num_simulations=5000)
    end_time = time.time()

    print(f"Evaluation time: {end_time - start_time:.2f} s")
    print("Phi ", value)

    with open(args.output, 'w') as f:
        f.write(f"{value:.3f}\n")

if __name__ == '__main__':
    main()






