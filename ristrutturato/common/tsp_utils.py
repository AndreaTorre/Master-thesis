# -*- coding: utf-8 -*-

def canon_edge(i, j):
    return (i, j) if i < j else (j, i)

def directed_to_undirected_arcs(arcs):
    return sorted({canon_edge(i, j) for (i, j) in arcs if i != j})

def all_undirected_edges(nodes):
    return sorted({
        canon_edge(nodes[a], nodes[b])
        for a in range(len(nodes))
        for b in range(a + 1, len(nodes))
    })

def base_cost_undirected(base_dist, i, j):
    return 0.5 * (base_dist[i][j] + base_dist[j][i])

def get_edge_value(values, i, j):
    edge = canon_edge(i, j)
    if isinstance(values, dict):
        return values[edge]
    return values

def get_C_value(C, i, j):
    return get_edge_value(C, i, j)

def extract_tour_from_arcs(arcs, start):
    succ = {i: j for (i, j) in arcs}
    tour = [start]
    current = start
    while True:
        if current not in succ:
            raise ValueError("Tour non ricostruibile: manca un successore.")
        nxt = succ[current]
        if nxt == start:
            break
        if nxt in tour:
            raise ValueError("Tour non semplice o sottociclo.")
        tour.append(nxt)
        current = nxt
    return tour

def tour_length_from_arcs(arcs, dist):
    return sum(dist[i][j] for (i, j) in arcs)
