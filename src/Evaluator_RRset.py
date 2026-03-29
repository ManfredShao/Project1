import random
import time
import argparse
from turtle import st


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


def read_network_inverse(file_path):
    reverse_graph = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return 0, {}

        n, m = map(int, lines[0].strip().split())

        for i in range(n):
            reverse_graph[i] = []

        for i in range(1, m + 1):
            parts = lines[i].strip().split()
            u = int(parts[0])
            v = int(parts[1])
            p1 = float(parts[2])
            p2 = float(parts[3])
            reverse_graph[v].append((u, p1, p2))

    return n, reverse_graph


def read_seeds(file_path):
    set1, set2 = set(), set()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return set1, set2
        k1, k2 = map(int, lines[0].strip().split())

        for i in range(1, k1 + 1):
            set1.add(int(lines[i].strip()))
        for i in range(1, k2 + 1):
            set2.add(int(lines[k1+i].strip()))

    return set1, set2


def generate_single_rrset(graph, target_node, prob_index):
    start_nodes = [target_node] + [v for v, _, _ in graph[target_node]]
    activated = set(start_nodes)
    queue = list(start_nodes)

    while queue:
        u = queue.pop()
        for prev_node in graph[u]:
            v = prev_node[0]
            if prob_index == 1:
                p = prev_node[1]
            else:
                p = prev_node[2]

            if v not in activated:
                if random.random() <= p:
                    activated.add(v)
                    queue.append(v)

    return activated


def evaluate_rr(n, graph, u1, u2, num_simulations=1000):
    valid_cnt = 0
    for _ in range(num_simulations):
        target_node = random.randint(0, n - 1)
        rr_set_1 = generate_single_rrset(graph, target_node, 1)
        rr_set_2 = generate_single_rrset(graph, target_node, 2)

        receive_1 = not rr_set_1.isdisjoint(u1)
        receive_2 = not rr_set_2.isdisjoint(u2)

        if receive_1 == receive_2:
            valid_cnt += 1

    return valid_cnt/num_simulations * n

def main():
    start_time = time.time()

    args = parse_args()
    n, graph = read_network_inverse(args.network)
    I1, I2 = read_seeds(args.initial_seeds)
    B1, B2 = read_seeds(args.balanced_seeds)

    if len(B1) + len(B2) > args.budget:
        print("Error: Budget exceeded")
        with open(args.output, 'w') as f:
            f.write("0.0/n")
        return

    U1 = I1.union(B1)
    U2 = I2.union(B2)

    value = evaluate_rr(n, graph, U1, U2, num_simulations=10000)
    end_time = time.time()

    print(f"Evaluation time: {end_time - start_time:.2f} s")
    print("Phi ", value)

    with open(args.output, 'w') as f:
        f.write(f"{value:.3f}\n")


if __name__ == '__main__':
    main()
