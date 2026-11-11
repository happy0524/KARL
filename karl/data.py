"""행 순서 보존, 엄격한 정렬 검증, 명시적인 TF-IDF 전처리."""
import hashlib
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .config import DATASETS, ROOT

TFIDF_SETTINGS = {
    "lowercase": True, "strip_accents": None, "stop_words": None,
    "token_pattern": r"(?u)\b\w\w+\b", "ngram_range": (1, 1),
    "min_df": 1, "max_df": 1.0, "max_features": None,
    "norm": "l2", "use_idf": True, "smooth_idf": True, "sublinear_tf": False,
}
PREPROCESSING_SETTINGS = {
    "reference": "src/data_utils.py:preprocess_text (MIT)",
    "lowercase": True, "token_pattern": "[a-z]+", "minimum_token_length": 4,
    "stop_words": "sklearn.feature_extraction.text.ENGLISH_STOP_WORDS",
    "cumulative_tfidf_pruning": False,
}


def preprocess_documents(documents):
    # CLARE의 MIT 코드와 같은 어휘 전처리. KARL의 'meaningful words'는 세부 명세가 없다.
    return [" ".join(word for word in re.findall(r"[a-z]+", document.lower())
                     if len(word) > 3 and word not in ENGLISH_STOP_WORDS)
            for document in documents]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_aligned_files(text_path, label_path, strip_label_prefix=False):
    # 빈 문서를 제거하면 이후 라벨이 밀리므로 빈 행도 하나의 문서로 유지한다.
    # splitlines()는 문서 내부의 form-feed까지 행으로 쪼개므로 실제 파일 행을 읽는다.
    with Path(text_path).open(encoding="utf-8-sig", errors="strict") as stream:
        documents = [line.rstrip("\r\n") for line in stream]
    with Path(label_path).open(encoding="utf-8-sig", errors="strict") as stream:
        labels = [line.rstrip("\r\n") for line in stream]
    labels = [label.strip() for label in labels]
    if len(documents) != len(labels) or not documents:
        raise ValueError(f"문서/라벨 행 수 불일치 또는 빈 데이터: {len(documents)}/{len(labels)}")
    if any(not label for label in labels):
        raise ValueError("빈 라벨을 발견했습니다. 행을 임의로 제거하지 않습니다")
    prefix_count = 0
    if strip_label_prefix:
        cleaned = []
        for document in documents:
            if "\t" in document:
                document = document.split("\t", 1)[1]
                prefix_count += 1
            cleaned.append(document)
        documents = cleaned
    return [document.strip() for document in documents], np.asarray(labels), prefix_count


def load_dataset(dataset, data_dir=None, max_samples=None, seed=42):
    stem, known_clusters, paper_n, paper_vocab = DATASETS[dataset]
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data"
    text_path, label_path = data_dir / f"{stem}.txt", data_dir / f"{stem}_labels.txt"
    documents, labels, prefix_count = load_aligned_files(
        text_path, label_path, strip_label_prefix=(dataset == "reuters8"))
    source_n = len(documents)
    indices = np.arange(source_n)
    if max_samples is not None and max_samples < source_n:
        # 라벨을 보지 않는 무작위 표본. 저장된 원본 행 번호로 재정렬할 수 있다.
        indices = np.sort(np.random.default_rng(seed).choice(source_n, max_samples, replace=False))
    documents = [documents[i] for i in indices]
    labels = labels[indices]
    metadata = {
        "dataset": dataset, "text_path": str(text_path.resolve()),
        "label_path": str(label_path.resolve()), "text_sha256": sha256_file(text_path),
        "label_sha256": sha256_file(label_path), "source_documents": source_n,
        "selected_documents": len(documents), "paper_documents": paper_n,
        "paper_vocabulary": paper_vocab, "benchmark_n_clusters": known_clusters,
        "source_count_matches_paper": source_n == paper_n,
        "empty_documents": sum(not document for document in documents),
        "stripped_label_prefixes_source": prefix_count,
        "order_validation": "동일 행 대응을 가정; 행 수·빈 라벨·저장한 원본 행 번호를 검증",
        "version_identity": "원본 문서 ID/배포 버전 정보 부재: 논문 버전과 동일함을 증명할 수 없음",
    }
    return documents, labels, indices, metadata


def vectorize_documents(documents):
    vectorizer = TfidfVectorizer(**TFIDF_SETTINGS, dtype=np.float64)
    matrix = vectorizer.fit_transform(preprocess_documents(documents))
    # 모두 빈 문서인 경우에는 sklearn이 명확한 empty vocabulary 오류를 낸다.
    return matrix, vectorizer
