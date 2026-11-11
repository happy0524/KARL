"""II-B/III-C: one-hot GCN, 엣지 decoder와 합/N 손실. 라벨을 받지 않는다."""
import csv
import hashlib
import os
import random
import time

import numpy as np
from scipy import sparse
# CUDA의 결정적 GEMM 설정은 CUDA context 생성 전에 지정해야 한다.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

from .graph import normalized_adjacency


def sparse_tensor(matrix, device="cpu"):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        np.stack([coo.row, coo.col]), coo.data.astype(np.float32), coo.shape,
        dtype=torch.float32, device=device).coalesce()


class IdentityGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, embedding_dim=32):
        super().__init__()
        # I @ W0 = W0이므로 N*N identity를 만들 필요가 없다.
        self.weight0 = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.bias0 = nn.Parameter(torch.zeros(hidden_dim))
        self.weight1 = nn.Parameter(torch.empty(hidden_dim, embedding_dim))
        self.bias1 = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, propagation):
        hidden = F.relu(torch.sparse.mm(propagation, self.weight0) + self.bias0)
        return torch.sparse.mm(propagation, hidden @ self.weight1) + self.bias1


def positive_pairs(adjacency):
    upper = sparse.triu(adjacency, k=1).tocoo()
    return np.stack([upper.row, upper.col]).astype(np.int64)


class NegativeSampler:
    """무방향 비연결 쌍에서 중복 없이 균등 추출. 매 epoch RNG를 진행한다."""
    def __init__(self, num_nodes, positives, seed):
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)
        u, v = np.minimum(positives[0], positives[1]), np.maximum(positives[0], positives[1])
        self.positive_codes = np.unique(u * num_nodes + v)
        self.available = num_nodes * (num_nodes - 1) // 2 - len(self.positive_codes)

    def sample(self, count):
        if count < 0 or count > self.available:
            raise ValueError(f"음성 쌍 부족: 요청 {count}, 가능한 비연결 쌍 {self.available}")
        n = self.num_nodes
        selected = np.empty(0, dtype=np.int64)
        if self.available < 2 * count:
            # 밀집된 작은 그래프에서도 무한 rejection 없이 정확한 개수를 반환한다.
            pool = []
            for u in range(n - 1):
                codes = u * n + np.arange(u + 1, n)
                pool.append(codes[~np.isin(codes, self.positive_codes)])
            selected = self.rng.choice(np.concatenate(pool), count, replace=False)
        else:
            while len(selected) < count:
                pairs = self.rng.integers(0, n, size=(2, max(64, int((count - len(selected)) * 1.5))))
                u, v = np.minimum(pairs[0], pairs[1]), np.maximum(pairs[0], pairs[1])
                codes = np.unique((u * n + v)[u != v])
                codes = codes[~np.isin(codes, self.positive_codes)]
                codes = codes[~np.isin(codes, selected)]
                # np.unique의 정렬 순서에 의한 작은 노드 번호 편향을 없앤다.
                codes = self.rng.permutation(codes)[:count - len(selected)]
                selected = np.concatenate([selected, codes])
        return np.stack([selected // n, selected % n])


def reconstruction_loss(embedding, positives, negatives, num_nodes):
    positive_logits = (embedding[positives[0]] * embedding[positives[1]]).sum(dim=1)
    negative_logits = (embedding[negatives[0]] * embedding[negatives[1]]).sum(dim=1)
    # softplus는 log(sigmoid)의 수치적으로 안정한 동일식이다. mean으로 바꾸지 않는다.
    return (F.softplus(-positive_logits).sum() + F.softplus(negative_logits).sum()) / num_nodes


def train_embedding(graph, config, output_dir):
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA를 사용할 수 없습니다. --device cpu를 사용하세요")
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    n = graph.adjacency.shape[0]
    pairs = positive_pairs(graph.adjacency)
    if pairs.shape[1] == 0:
        raise ValueError("학습할 양성 엣지가 없습니다")
    sampler = NegativeSampler(n, pairs, config.seed + 1)
    if sampler.available < pairs.shape[1]:
        raise ValueError("양성 엣지와 같은 수의 고유 음성 쌍을 만들 수 없습니다")
    propagation = sparse_tensor(normalized_adjacency(graph.adjacency), config.device)
    positives = torch.as_tensor(pairs, dtype=torch.long, device=config.device)
    model = IdentityGCN(n, config.hidden_dim, config.embedding_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        negative_array = sampler.sample(pairs.shape[1])
        negatives = torch.as_tensor(negative_array, dtype=torch.long, device=config.device)
        optimizer.zero_grad(set_to_none=True)
        embedding = model(propagation)
        loss = reconstruction_loss(embedding, positives, negatives, n)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"epoch {epoch}: 유한하지 않은 손실")
        loss.backward()
        optimizer.step()
        row = {"epoch": epoch, "loss": float(loss.detach().cpu()),
               "positive_pairs": pairs.shape[1], "negative_pairs": negative_array.shape[1],
               "negative_sha256": hashlib.sha256(negative_array.tobytes()).hexdigest(),
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            print(f"KARL epoch {epoch}/{config.epochs} loss={row['loss']:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        embedding = model(propagation).cpu().numpy()
    seconds = time.perf_counter() - start
    if not np.isfinite(embedding).all():
        raise FloatingPointError("임베딩에 유한하지 않은 값이 있습니다")
    np.save(output_dir / "embeddings.npy", embedding)
    with (output_dir / "loss.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save({"state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "config": config.to_dict(), "num_nodes": n,
                "selected_epoch": config.epochs}, output_dir / "model.pt")
    return embedding, {"training_seconds": seconds, "selected_epoch": config.epochs,
                       "selection_rule": "fixed_final_epoch_without_labels",
                       "optimizer": {"name": "Adam", "learning_rate": config.learning_rate,
                                     "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
                       "initialization": "Xavier uniform weights, zero shared feature biases",
                       "normalization": "D=sum(A), self-loop only in A+I; isolated D=1",
                       "loss": "(sum softplus(-positive_logits)+sum softplus(negative_logits))/N",
                       "pair_convention": "unique unordered pairs u<v; self-loops excluded",
                       "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"],
                       "positive_pairs": pairs.shape[1], "negative_pairs_per_epoch": pairs.shape[1],
                       "negative_pool_size": sampler.available,
                       "unique_negative_epoch_hashes": len({row['negative_sha256'] for row in history})}
