def bfs(live_graph, seeds, blocked_set=None):
    if blocked_set is None:
        blocked_set = set()
    queue = [s for s in seeds if s not in blocked_set]
    visited = set(queue)
    while queue:
        u = queue.pop()
        for v in live_graph.get(u, []):
            if v not in visited and v not in blocked_set:
                visited.add(v)
                queue.append(v)
    return visited

g = {0: [2], 1: [2], 2: [3], 3: []}
# A1 initially has 0. we consider adding 1.
r1 = bfs(g, {0})
# A1 reaches {0, 2, 3}

# User's code blocked_set = A1
delta_user = bfs(g, {1}, blocked_set={0})
print("User delta:", delta_user) # EXPECT: {1, 2, 3}. BUT 2 and 3 are already in r1!

# Correct code blocked_set = r1
delta_correct = bfs(g, {1}, blocked_set=r1)
print("Correct delta:", delta_correct) # EXPECT: {1}.

