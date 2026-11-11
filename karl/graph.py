"""II-A, Eq.(1), Algorithm 1의 선택적 무방향 가중 그래프."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Graph:
    adjacency: sparse.csr_matrix
    directed_choices: sparse.csr_matrix
    stats: dict


def normalized_cosine(tfidf):
    similarity = cosine_similarity(tfidf)
    np.clip(similarity, 0, 1, out=similarity)
    np.fill_diagonal(similarity, 0)
    maxima = similarity.max(axis=1)
    np.divide(similarity, maxima[:, None], out=similarity, where=maxima[:, None] > 0)
    return similarity


def select_neighbors(normalized, node, knn_k, alpha):
    row = normalized[node]
    candidates = np.flatnonzero(row > 0)
    candidates = candidates[candidates != node]
    # 동일한 유사도에서는 원본 행 번호가 작은 문서를 먼저 선택한다.
    candidates = candidates[np.lexsort((candidates, -row[candidates]))][:knn_k]
    if not len(candidates):
        return [], 0.0, candidates
    # 유효 이웃이 K보다 작으면 마지막 유효 이웃으로 임계값을 정한다.
    threshold = alpha * row[candidates[-1]]
    accepted = [int(candidates[0])]
    for candidate in candidates[1:]:
        # NC[i,candidate]가 아닌 NC[candidate,b]를 비교한다 (line 13).
        if any(normalized[candidate, neighbor] >= threshold for neighbor in accepted):
            accepted.append(int(candidate))
    return accepted, float(threshold), candidates


def graph_from_normalized(normalized, knn_k=50, alpha=0.2):
    normalized = np.asarray(normalized)
    if (normalized.ndim != 2 or normalized.shape[0] != normalized.shape[1]
            or normalized.shape[0] < 2
            or not np.isfinite(normalized).all() or (normalized < 0).any()):
        raise ValueError("유한한 비음수 정사각 유사도 행렬이 필요합니다")
    if knn_k < 1 or not 0 < alpha <= 1:
        raise ValueError("K >= 1, 0 < alpha <= 1이어야 합니다")
    n = normalized.shape[0]
    sources, targets, weights = [], [], []
    merged_edges = {}
    candidate_count = 0
    for node in range(n):
        accepted, _, candidates = select_neighbors(normalized, node, knn_k, alpha)
        candidate_count += len(candidates)
        sources.extend([node] * len(accepted))
        targets.extend(accepted)
        weights.extend(normalized[node, accepted])
        for neighbor in accepted:
            key = (min(node, neighbor), max(node, neighbor))
            # CLARE와 같은 오름차순 노드 순회 및 마지막 선택값 덮어쓰기.
            merged_edges[key] = float(normalized[node, neighbor])
    choices = sparse.csr_matrix((weights, (sources, targets)), shape=(n, n))
    rows, cols, values = [], [], []
    for (source, target), weight in merged_edges.items():
        rows.extend([source, target])
        cols.extend([target, source])
        values.extend([weight, weight])
    adjacency = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    degree = np.diff(adjacency.indptr)
    stats = {
        "num_nodes": n, "undirected_edges": adjacency.nnz // 2,
        "directed_selected": choices.nnz, "topk_candidates": candidate_count,
        "rejected_candidates": candidate_count - choices.nnz,
        "isolated_nodes": int((degree == 0).sum()),
        "components": int(connected_components(adjacency, directed=False)[0]),
        "density": float(adjacency.nnz / max(n * (n - 1), 1)),
        "degree_min": int(degree.min()), "degree_max": int(degree.max()),
        "degree_mean": float(degree.mean()),
        "degree_quantiles": np.quantile(degree, [0, .25, .5, .75, 1]).tolist(),
        "weighted_degree_mean": float(np.asarray(adjacency.sum(axis=1)).mean()),
        "merge_policy": "union_last_selected_direction_in_ascending_source_order_like_CLARE",
        "zero_similarity_rows": int((normalized.max(axis=1) == 0).sum()),
    }
    return Graph(adjacency, choices, stats)


def build_graph(tfidf, knn_k=50, alpha=0.2):
    return graph_from_normalized(normalized_cosine(tfidf), knn_k, alpha)


def normalized_adjacency(adjacency):
    """D=diag(sum_j A_ij), Ahat=D^-1/2(A+I)D^-1/2 (논문 문면 해석)."""
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    # 고립 노드는 가중 degree=0을 1로 두어 자신의 변환만 유지한다.
    safe_degree = np.where(degree > 0, degree, 1.0)
    inv = sparse.diags(1 / np.sqrt(safe_degree))
    return (inv @ (adjacency + sparse.eye(adjacency.shape[0])) @ inv).tocsr()
