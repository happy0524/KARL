"""임베딩 저장 후 수행하는 군집화. 라벨은 이 단계의 지표 계산에만 사용한다."""
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize


class SphericalKMeans:
    """각 assignment는 cosine 최대화, 각 update는 중심을 단위 벡터로 투영한다."""
    def __init__(self, n_clusters, n_init=20, max_iter=300, seed=42):
        self.n_clusters, self.n_init, self.max_iter, self.seed = n_clusters, n_init, max_iter, seed

    def fit_predict(self, embedding):
        points = normalize(np.asarray(embedding, dtype=np.float64))
        n = len(points)
        if not 1 <= self.n_clusters <= n or self.n_init < 1:
            raise ValueError("잘못된 군집 수/초기화 반복 수")
        rng = np.random.default_rng(self.seed)
        best_objective = -np.inf
        for _ in range(self.n_init):
            centers = points[rng.choice(n, self.n_clusters, replace=False)].copy()
            previous = None
            for _ in range(self.max_iter):
                scores = points @ centers.T
                labels = scores.argmax(axis=1)
                if previous is not None and np.array_equal(labels, previous):
                    break
                previous = labels.copy()
                updated = np.zeros_like(centers)
                # 빈 군집은 현재 중심들과 가장 덜 유사한 점부터 재시작한다.
                worst = np.argsort(scores.max(axis=1), kind="stable")
                replacements = iter(worst)
                for cluster in range(self.n_clusters):
                    members = points[labels == cluster]
                    updated[cluster] = members.sum(axis=0) if len(members) else points[next(replacements)]
                centers = normalize(updated)
            # 최종 중심에 대해 다시 할당한다. 이전 iteration의 stale label을 저장하지 않는다.
            scores = points @ centers.T
            labels = scores.argmax(axis=1)
            objective = float(scores[np.arange(n), labels].sum())
            if objective > best_objective:
                best_objective = objective
                self.labels_, self.cluster_centers_ = labels.copy(), centers.copy()
        self.objective_ = best_objective
        return self.labels_


def clustering_metrics(labels, predictions):
    if len(labels) != len(predictions) or not len(labels):
        raise ValueError("평가 라벨/예측 길이가 일치해야 합니다")
    _, true_ids = np.unique(labels, return_inverse=True)
    _, predicted_ids = np.unique(predictions, return_inverse=True)
    # 직사각형 contingency는 알려진 군집 수와 다른 k에도 유효하다.
    counts = np.zeros((predicted_ids.max() + 1, true_ids.max() + 1), dtype=np.int64)
    np.add.at(counts, (predicted_ids, true_ids), 1)
    row, column = linear_sum_assignment(-counts)
    return {"acc": float(counts[row, column].sum() / len(labels)),
            "nmi": float(normalized_mutual_info_score(labels, predictions, average_method="arithmetic")),
            "ari": float(adjusted_rand_score(labels, predictions))}


def cluster_and_evaluate(embedding, labels, n_clusters, seed=42, n_init=20):
    if not 1 <= n_clusters <= len(embedding) or not np.isfinite(embedding).all():
        raise ValueError("군집 수 또는 임베딩 값이 유효하지 않습니다")
    results, assignments = [], {}
    for name in ("kmeans", "spherical_kmeans"):
        start = time.perf_counter()
        if name == "kmeans":
            model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed,
                           init="k-means++", max_iter=300, algorithm="lloyd")
        else:
            model = SphericalKMeans(n_clusters, n_init=n_init, seed=seed)
        predictions = model.fit_predict(embedding)
        duration = time.perf_counter() - start
        assignments[name] = predictions
        results.append({"method": name, "n_clusters": n_clusters,
                        **clustering_metrics(labels, predictions), "seconds": duration,
                        "seed": seed, "n_init": n_init,
                        "occupied_clusters": int(len(np.unique(predictions)))})
    return results, assignments
