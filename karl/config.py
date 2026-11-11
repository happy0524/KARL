"""논문 III-C 기본값과 로컬 데이터 경로. 군집 수는 학습 설정과 분리한다."""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "bbc": ("bbc", 5, 2225, 37644),
    "20newsgroup": ("20news", 20, 18846, 100259),
    "reuters10k": ("reuters10k", 4, 10000, 28628),
    "reuters8": ("reuter8", 8, 7674, 21088),
    "webkb": ("webKB", 6, 4518, 6887),
}


@dataclass
class Config:
    dataset: str = "bbc"
    seed: int = 42
    knn_k: int = 50
    alpha: float = 0.2
    hidden_dim: int = 64
    embedding_dim: int = 32
    learning_rate: float = 0.01
    epochs: int = 200
    device: str = "cpu"
    threads: int = 4
    max_samples: int | None = None
    n_init: int = 20

    def validate(self):
        if self.dataset not in DATASETS:
            raise ValueError("지원하지 않는 데이터셋")
        if min(self.knn_k, self.hidden_dim, self.embedding_dim, self.epochs,
               self.threads, self.n_init) < 1:
            raise ValueError("차원, epoch, 이웃 수, 반복 수는 양수여야 합니다")
        if not 0 < self.alpha <= 1 or self.learning_rate <= 0:
            raise ValueError("0 < alpha <= 1, learning_rate > 0이어야 합니다")
        if self.max_samples is not None and self.max_samples < 2:
            raise ValueError("max_samples는 2 이상이어야 합니다")

    def to_dict(self):
        return asdict(self)


def paper_config(dataset="bbc"):
    config = Config(dataset=dataset)
    if dataset == "20newsgroup":
        config.knn_k, config.hidden_dim, config.embedding_dim = 20, 128, 64
    config.validate()
    return config
