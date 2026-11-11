"""데이터 → 그래프 → 고정 epoch 학습 → 저장 → 독립적인 군집화 진입점."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

import numpy as np
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits

from .clustering import cluster_and_evaluate
from .config import Config, DATASETS, ROOT, paper_config
from .data import (PREPROCESSING_SETTINGS, TFIDF_SETTINGS, load_dataset,
                   sha256_file, vectorize_documents)
from .graph import build_graph
from .model import train_embedding


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def unique_name():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def environment_info():
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl"):
        versions[package] = importlib.metadata.version(package)
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def source_manifest():
    paths = list((ROOT / "karl").glob("*.py"))
    paths += list((ROOT / "configs" / "karl").glob("*.json"))
    paths += [ROOT / "requirements-karl.txt"]
    paths += list(ROOT.glob("GAE*.pdf"))
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def save_clustering(run_dir, n_clusters=None, seed=42, n_init=20, threads=4):
    """저장된 배열만 읽는다. 그래프 구성과 모델 생성·학습을 호출하지 않는다."""
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "embedding_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if sha256_file(run_dir / name) != expected:
            raise ValueError(f"저장된 데이터 무결성 오류: {name}")
    embedding = np.load(run_dir / "embeddings.npy", allow_pickle=False)
    labels = np.load(run_dir / "labels.npy", allow_pickle=False)
    indices = np.load(run_dir / "source_indices.npy", allow_pickle=False)
    if embedding.ndim != 2 or not len(embedding) == len(labels) == len(indices):
        raise ValueError("임베딩·라벨·원본 행 번호의 차원 또는 행 수 불일치")
    if n_clusters is None:
        n_clusters = manifest["benchmark_n_clusters"]
    if n_init < 1 or threads < 1:
        raise ValueError("n_init과 threads는 양수여야 합니다")
    start = time.perf_counter()
    with threadpool_limits(limits=threads):
        metrics, assignments = cluster_and_evaluate(embedding, labels, n_clusters, seed, n_init)
    output = run_dir / "clustering" / f"k{n_clusters}_{unique_name()}"
    output.mkdir(parents=True, exist_ok=False)
    for name, predictions in assignments.items():
        np.save(output / f"{name}_labels.npy", predictions)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    summary = {"metrics": metrics, "seconds": time.perf_counter() - start,
               "embedding_sha256": manifest["sha256"]["embeddings.npy"],
               "n_clusters": n_clusters, "benchmark_n_clusters": manifest["benchmark_n_clusters"],
               "cluster_count_source": "benchmark_metadata" if n_clusters == manifest["benchmark_n_clusters"]
                                       else "explicit_user_value",
               "seed": seed, "n_init": n_init, "threads": threads,
               "max_iter": 300, "kmeans_init": "k-means++",
               "spherical_init": "random distinct document directions",
               "label_counts_evaluation_only": {str(label): int(count) for label, count
                                                in zip(*np.unique(labels, return_counts=True))},
               "environment": environment_info(), "argv": sys.argv,
               "embedding_retrained": False}
    write_json(output / "summary.json", summary)
    for row in metrics:
        print(f"{row['method']} k={n_clusters} ACC={row['acc']:.6f} "
              f"NMI={row['nmi']:.6f} ARI={row['ari']:.6f}", flush=True)
    print(f"군집화 결과: {output}", flush=True)
    return output, summary


def run_pipeline(config, run_dir=None, data_dir=None, n_clusters=None, evaluate=True):
    config.validate()
    output = (Path(run_dir).resolve() if run_dir else
              ROOT / "results" / "karl" / config.dataset / unique_name())
    output.mkdir(parents=True, exist_ok=False)
    total_start = time.perf_counter()
    summary = {"status": "running", "run_kind": "smoke" if config.max_samples else "full_dataset",
               "paper_performance_reproduced": False,
               "config": config.to_dict(), "timings": {}}
    write_json(output / "config.json", config.to_dict())
    write_json(output / "environment.json", environment_info())
    write_json(output / "source_manifest.json", source_manifest())
    write_json(output / "command.json", {"argv": sys.argv, "cwd": str(Path.cwd())})
    write_json(output / "summary.json", summary)
    try:
        start = time.perf_counter()
        documents, labels, indices, metadata = load_dataset(
            config.dataset, data_dir, config.max_samples, config.seed)
        if n_clusters is not None and not 1 <= n_clusters <= len(documents):
            raise ValueError("n_clusters는 1 이상 문서 수 이하여야 합니다")
        summary["timings"]["load_seconds"] = time.perf_counter() - start
        np.save(output / "labels.npy", labels)
        np.save(output / "source_indices.npy", indices)
        start = time.perf_counter()
        tfidf, _ = vectorize_documents(documents)
        summary["timings"]["tfidf_seconds"] = time.perf_counter() - start
        word_counts = np.array([len(document.split()) for document in documents])
        metadata.update({"tfidf_shape": list(tfidf.shape), "tfidf_nonzeros": tfidf.nnz,
                         "zero_tfidf_documents": int((np.diff(tfidf.indptr) == 0).sum()),
                         "vocabulary_matches_paper": tfidf.shape[1] == metadata["paper_vocabulary"],
                         "raw_words_mean": float(word_counts.mean()),
                         "raw_words_min": int(word_counts.min()), "raw_words_max": int(word_counts.max()),
                         "preprocessing": PREPROCESSING_SETTINGS, "tfidf_settings": TFIDF_SETTINGS})
        write_json(output / "data.json", metadata)
        start = time.perf_counter()
        with threadpool_limits(limits=config.threads):
            graph = build_graph(tfidf, config.knn_k, config.alpha)
        sparse.save_npz(output / "adjacency.npz", graph.adjacency)
        sparse.save_npz(output / "directed_choices.npz", graph.directed_choices)
        write_json(output / "graph.json", graph.stats)
        summary["timings"]["graph_seconds"] = time.perf_counter() - start
        print(f"그래프: N={len(documents)}, 무방향 E={graph.stats['undirected_edges']}, "
              f"선택 거부={graph.stats['rejected_candidates']}", flush=True)
        _, training = train_embedding(graph, config, output)
        write_json(output / "training.json", training)
        summary["timings"]["training_seconds"] = training["training_seconds"]
        manifest = {"benchmark_n_clusters": DATASETS[config.dataset][1],
                    "num_nodes": len(documents), "embedding_dim": config.embedding_dim,
                    "sha256": {name: sha256_file(output / name) for name in
                               ("embeddings.npy", "labels.npy", "source_indices.npy")}}
        write_json(output / "embedding_manifest.json", manifest)
        if evaluate:
            cluster_dir, clustering = save_clustering(output, n_clusters, config.seed,
                                                       config.n_init, config.threads)
            summary["clustering_dir"] = str(cluster_dir.relative_to(output))
            summary["metrics"] = clustering["metrics"]
            summary["timings"]["clustering_seconds"] = clustering["seconds"]
        summary["status"] = "completed"
    except Exception:
        summary["status"] = "failed"
        summary["error"] = traceback.format_exc()
        raise
    finally:
        summary["timings"]["total_seconds"] = time.perf_counter() - total_start
        write_json(output / "summary.json", summary)
        print(f"실행 결과: {output}", flush=True)
    return output


def main():
    # cp949 콘솔에서도 한글 경로와 오류 메시지를 손실 없이 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="KARL 문서 임베딩·군집화")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "train"):
        command = commands.add_parser(name, help="전체 실행" if name == "run" else "임베딩 학습·저장만 실행")
        command.add_argument("--dataset", choices=DATASETS, default=None)
        command.add_argument("--config", type=Path, help="JSON 설정; CLI 값이 우선")
        command.add_argument("--data-dir", type=Path)
        command.add_argument("--run-dir", type=Path, help="새 결과 폴더; 기존 폴더 덮어쓰기 금지")
        for flag in ("seed", "knn-k", "hidden-dim", "embedding-dim", "epochs", "threads", "max-samples", "n-init"):
            command.add_argument("--" + flag, type=int, default=None)
        for flag in ("alpha", "learning-rate"):
            command.add_argument("--" + flag, type=float, default=None)
        command.add_argument("--device", default=None, help="cpu 또는 cuda:0")
        if name == "run":
            command.add_argument("--n-clusters", type=int, default=None,
                                 help="최종 군집화에만 사용; 기본값은 벤치마크 군집 수")
    command = commands.add_parser("cluster", help="저장된 임베딩만 재사용; 학습 없음")
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--n-clusters", type=int, default=None)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--n-init", type=int, default=20)
    command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "cluster":
        save_clustering(args.run_dir, args.n_clusters, args.seed, args.n_init, args.threads)
        return
    supplied = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    dataset = args.dataset or supplied.get("dataset", "bbc")
    values = paper_config(dataset).to_dict()
    values.update(supplied)
    values["dataset"] = dataset
    for name in values:
        if getattr(args, name, None) is not None:
            values[name] = getattr(args, name)
    config = Config(**values)
    run_pipeline(config, args.run_dir, args.data_dir, getattr(args, "n_clusters", None),
                 evaluate=args.command == "run")


if __name__ == "__main__":
    main()
