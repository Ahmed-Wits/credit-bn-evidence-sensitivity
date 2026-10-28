"""structure: multi-restart hill-climbing + BIC structure learning.

Wrapped, with no logic changes, from the locked run (old notebook cell 27, recovered from the kernel history):
  - start DAG: random.Random(seed) shuffles a COPY of node_order; each forward-only pair (i<j) gets an edge with prob edge_density
  - HillClimbSearch(data).estimate(scoring_method=BIC(data), start_dag=start, show_progress=False)
  - score = BIC(data).score(learned_dag); best = highest score; ties keep the earliest restart (stable sort)
The result depends on the ORDER of node_order and on PYTHONHASHSEED=0 (set before Python starts).
"""
import os, random, time


def random_start_dag(nodes, rng, edge_prob=0.15):
    from pgmpy.base import DAG
    order = list(nodes)
    rng.shuffle(order)
    dag = DAG()
    dag.add_nodes_from(order)
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if rng.random() < edge_prob:
                dag.add_edge(order[i], order[j])
    return dag


def learn_structure(data, node_order, seeds=range(42, 57), edge_density=0.15, verbose=True):
    """Return dict(best_score, best_dag, best_seed, best_edges, results=[(score, dag, seed), ...], runtime_s)."""
    assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED must be '0' before Python starts"
    from pgmpy.estimators import HillClimbSearch, BIC
    assert list(data.columns) == list(node_order), "data column order must equal node_order"
    t0 = time.time()
    bic = BIC(data)
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        start = random_start_dag(node_order, rng, edge_density)
        learned = HillClimbSearch(data).estimate(scoring_method=bic, start_dag=start, show_progress=False)
        score = bic.score(learned)
        results.append((score, learned, seed))
        if verbose:
            print(f"seed {seed}: BIC = {score:.2f}, edges = {len(learned.edges())}  ({time.time() - t0:.0f}s elapsed)")
    ranked = sorted(results, key=lambda x: x[0], reverse=True)
    best_score, best_dag, best_seed = ranked[0]
    return dict(best_score=best_score, best_dag=best_dag, best_seed=best_seed,
                best_edges=sorted(map(tuple, best_dag.edges())), results=results, runtime_s=time.time() - t0)
