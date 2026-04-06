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
# The following functions (sample_graph, bfs, get_exposured) are the same as in IEMP_Heur.py and are used for simulating the diffusion process and calculating exposured.
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

def precompute_evol_data(graph, n, num_graphs=100):
    live_graphs_1 = [sample_graph(n, graph, 1) for _ in range(num_graphs)]
    live_graphs_2 = [sample_graph(n, graph, 2) for _ in range(num_graphs)]
    return live_graphs_1, live_graphs_2

def fitness_eval(x, n, k, I1, I2, live_graphs_1, live_graphs_2, graph):
    num_ones = sum(x)
    if num_ones > k:
        return -num_ones  # Penalize solutions that exceed the budget
    
    S1 = {i for i in range(n) if x[i] == 1}
    S2 = {i-n for i in range(n, 2*n) if x[i] == 1}

    a1 = I1.union(S1)
    a2 = I2.union(S2)

    phi = 0.0

    for j in range(len(live_graphs_1)):
        a1_active = bfs(live_graphs_1[j], a1)
        a2_active = bfs(live_graphs_2[j], a2)
        r1 = get_exposured(graph, a1_active)
        r2 = get_exposured(graph, a2_active)

        symmetric_diff = r1 ^ r2
        phi += (n - len(symmetric_diff))

    return phi/len(live_graphs_1)

def generate_intial_population(pop_size, n, k):
    population = []
    for _ in range(pop_size):
        individual = [0] * (2*n)
        ones_count = random.randint(1, k)
        position = random.sample(range(2*n), ones_count)
        for pos in position:
            individual[pos] = 1
        population.append(individual)
    return population

def binary_tournament_selection(population, fitnesses):
    idx1, idx2 = random.sample(range(len(population)), 2)
    if fitnesses[idx1] > fitnesses[idx2]:
        return population[idx1]
    else:
        return population[idx2]
    
def two_point_crossover(parent1, parent2, n):
    p1, p2 = sorted(random.sample(range(2*n), 2))
    child_1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    child_2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
    return child_1, child_2

def mutation(individual, n, k):
    mutation_rate = 1/(2*n)
    mutated = individual.copy()
    for i in range(2*n):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]  # Flip the bit

    if sum(mutated) > k:
        # If mutation exceeds budget, randomly flip some bits back to 0
        ones_indices = [i for i in range(2*n) if mutated[i] == 1]
        random.shuffle(ones_indices)
        for idx in ones_indices[:sum(mutated)-k]:
            mutated[idx] = 0

    elif sum(mutated) < k:
        zero_indices = [i for i in range(2*n) if mutated[i] == 0]
        random.shuffle(zero_indices)
        for idx in zero_indices[:k-sum(mutated)]:
            mutated[idx] = 1
            
    return mutated

def evolutionary_algorithm(n, graph, I1, I2, k, pop_size=50, generations=100):
    population = generate_intial_population(pop_size, n, k)
    
    for gen in range(generations):
        live_graphs_1, live_graphs_2 = precompute_evol_data(graph, n)
        fitnesses = [fitness_eval(ind, n, k, I1, I2, live_graphs_1, live_graphs_2, graph) for ind in population]
        offspring = []
        for _ in range(pop_size // 2):
            parent1 = binary_tournament_selection(population, fitnesses)
            parent2 = binary_tournament_selection(population, fitnesses)
            c1, c2 = two_point_crossover(parent1, parent2, n)
            c1 = mutation(c1, n, k)
            c2 = mutation(c2, n, k)
            offspring.extend([c1, c2])

        offspring_fitnesses = [fitness_eval(ind, n, k, I1, I2, live_graphs_1, live_graphs_2, graph) for ind in offspring]
        combined_population = population + offspring
        combined_fitnesses = fitnesses + offspring_fitnesses

        # Elite selection
        paired = list(zip(combined_population, combined_fitnesses))
        paired.sort(key=lambda x: x[1], reverse=True)
        population = [x[0] for x in paired[:pop_size]]

        best_fitness = paired[0][1]
        print(f"Generation {gen+1}/{generations}: Best Fitness = {best_fitness:.2f}")

    best_individual = population[0]
    S1 = {i for i in range(n) if best_individual[i] == 1}
    S2 = {i-n for i in range(n, 2*n) if best_individual[i] == 1}
    return S1, S2

def main():
    start_time = time.time()

    args = parse_args()
    n, graph = read_network(args.network)
    I1, I2 = read_seeds(args.initial_seeds)
    
    S1, S2 = evolutionary_algorithm(n, graph, I1, I2, args.budget, pop_size=50, generations=25)

    with open(args.balanced_seeds, 'w') as f:
        f.write(f"{len(S1)} {len(S2)}\n")
        for seed in S1:
            f.write(f"{seed}\n")
        for seed in S2:
            f.write(f"{seed}\n")

    end_time = time.time()

    print(f"Evolution time: {end_time - start_time:.2f} s")


if __name__ == '__main__':
    main()
