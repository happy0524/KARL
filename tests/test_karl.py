"""논문 수식의 수치 검증. unittest만 사용하며 라벨 기반 튜닝은 하지 않는다."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch

from karl.clustering import SphericalKMeans, clustering_metrics
from karl.config import Config, paper_config
from karl.data import load_aligned_files, preprocess_documents
from karl.graph import (graph_from_normalized, normalized_adjacency,
                        normalized_cosine, select_neighbors)
from karl.main import run_pipeline, save_clustering
from karl.model import (IdentityGCN, NegativeSampler, positive_pairs,
                        reconstruction_loss, sparse_tensor)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.nc = np.array([[0, 1, .8, .6, 0], [1, 0, .05, .05, 0],
                            [.8, .1, 0, .3, 0], [.6, .05, .3, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=float)

    def test_cosine_diagonal_row_max_and_zero_row(self):
        x = sparse.csr_matrix([[1, 0], [1, 1], [0, 1], [0, 0]])
        assert_allclose(normalized_cosine(x), [[0, 1, 0, 0], [1, 0, 1, 0],
                                              [0, 1, 0, 0], [0, 0, 0, 0]])
        x = sparse.csr_matrix([[1, 0], [.8, .6], [0, 1]])
        nc = normalized_cosine(x)
        self.assertAlmostEqual(nc[1, 2], .75)
        self.assertAlmostEqual(nc[2, 1], 1)

    def test_algorithm1_prunes_and_rejected_neighbor_cannot_support(self):
        accepted, threshold, candidates = select_neighbors(self.nc, 0, 3, .5)
        self.assertAlmostEqual(threshold, .3)
        assert_array_equal(candidates, [1, 2, 3])
        self.assertEqual(accepted, [1])
        # 3은 거부된 2와 유사해도 아직 선택된 1과 유사하지 않으므로 거부된다.

    def test_candidate_row_index_inclusive_boundary_and_sequential_acceptance(self):
        nc = self.nc.copy()
        nc[2, 1] = .3
        # NC[2,1]을 사용해야 한다. NC[1,2]=.05를 사용하면 이 검증이 실패한다.
        self.assertEqual(select_neighbors(nc, 0, 3, .5)[0], [1, 2, 3])

    def test_expected_entire_graph_and_last_selected_weight(self):
        graph = graph_from_normalized(self.nc, 3, .5)
        # 모든 행을 손으로 순회한 기대값. 양방향 선택의 마지막 원점은 큰 인덱스다.
        expected = np.array([[0, 1, .8, .6, 0], [1, 0, .1, .05, 0],
                             [.8, .1, 0, .3, 0], [.6, .05, .3, 0, 0],
                             [0, 0, 0, 0, 0]])
        assert_allclose(graph.adjacency.toarray(), expected)
        self.assertEqual(graph.stats["isolated_nodes"], 1)
        self.assertEqual(graph.stats["undirected_edges"], 6)
        # 최댓값 병합과 마지막값 병합을 구분하는 고정 예제.
        asymmetric = np.array([[0, 1, 0], [.4, 0, 1], [0, .2, 0]])
        graph = graph_from_normalized(asymmetric, 1, .2)
        self.assertEqual(graph.adjacency[1, 2], .2)
        self.assertEqual(graph.adjacency[0, 1], 1)

    def test_ties_short_candidate_list_zero_row_no_self(self):
        nc = np.array([[0, 1, 1, 0], [1, 0, .2, 0], [1, .2, 0, 0], [0, 0, 0, 0]])
        accepted, threshold, candidates = select_neighbors(nc, 0, 50, .2)
        self.assertEqual(accepted, [1, 2])
        assert_array_equal(candidates, [1, 2])
        self.assertEqual(threshold, .2)
        self.assertEqual(select_neighbors(nc, 3, 50, .2)[0], [])

    def test_weighted_degree_before_loop_normalization(self):
        a = sparse.csr_matrix([[0, .25, 0], [.25, 0, 0], [0, 0, 0]])
        # degree=.25, .25, 0: self-loop 추가 전 가중치 합. 고립 노드 예외는 degree=1.
        expected = np.array([[4, 1, 0], [1, 4, 0], [0, 0, 1]])
        assert_allclose(normalized_adjacency(a).toarray(), expected)
        self.assertEqual(positive_pairs(a).tolist(), [[0], [1]])


class ModelTests(unittest.TestCase):
    def test_sparse_identity_and_edge_decoder_equal_dense_values_and_gradients(self):
        torch.manual_seed(4)
        propagation = normalized_adjacency(sparse.csr_matrix(
            [[0, .5, 0, 0], [.5, 0, .8, 0], [0, .8, 0, 0], [0, 0, 0, 0]]))
        model = IdentityGCN(4, 3, 2).double()
        shat = sparse_tensor(propagation).double()
        ahat = shat.to_dense()
        fast = model(shat)
        dense = ahat @ torch.relu(ahat @ torch.eye(4, dtype=torch.float64) @ model.weight0
                                 + model.bias0) @ model.weight1 + model.bias1
        torch.testing.assert_close(fast, dense, atol=1e-12, rtol=1e-12)
        pos, neg = torch.tensor([[0, 1], [1, 2]]), torch.tensor([[0, 1], [3, 3]])
        edge_loss = reconstruction_loss(fast, pos, neg, 4)
        decoder = torch.sigmoid(dense @ dense.T)
        dense_loss = (-torch.log(decoder[pos[0], pos[1]]).sum()
                      -torch.log1p(-decoder[neg[0], neg[1]]).sum()) / 4
        torch.testing.assert_close(edge_loss, dense_loss, atol=1e-12, rtol=1e-12)
        grad_fast = torch.autograd.grad(edge_loss, tuple(model.parameters()), retain_graph=True)
        grad_dense = torch.autograd.grad(dense_loss, tuple(model.parameters()))
        for left, right in zip(grad_fast, grad_dense):
            torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)

    def test_sum_divided_by_nodes_not_edge_mean(self):
        z = torch.zeros((5, 2), requires_grad=True)
        pos = torch.tensor([[0, 1], [1, 2]])
        neg = torch.tensor([[0, 1], [3, 4]])
        self.assertAlmostEqual(reconstruction_loss(z, pos, neg, 5).item(), 4 * np.log(2) / 5, places=6)

    def test_nonzero_logits_loss_and_no_weighted_targets(self):
        z = torch.tensor([[1., 0], [2., 0], [-1., 0]])
        pos, neg = torch.tensor([[0], [1]]), torch.tensor([[0], [2]])
        expected = (np.logaddexp(0, -2) + np.logaddexp(0, -1)) / 3
        self.assertAlmostEqual(reconstruction_loss(z, pos, neg, 3).item(), expected, places=7)

    def test_final_layer_has_no_relu(self):
        model = IdentityGCN(2, 1, 1)
        with torch.no_grad():
            model.weight0.fill_(1)
            model.weight1.fill_(-2)
        output = model(sparse_tensor(sparse.eye(2)))
        assert_allclose(output.detach().numpy(), [[-2], [-2]])

    def test_negative_samples_unique_exclude_positive_reverse_and_self_resample(self):
        positives = np.array([[0, 1, 2], [1, 2, 3]])
        first = NegativeSampler(8, positives, 7)
        second = NegativeSampler(8, positives, 7)
        a, b = first.sample(10), first.sample(10)
        assert_array_equal(a, second.sample(10))
        self.assertFalse(np.array_equal(a, b))
        codes = a[0] * 8 + a[1]
        self.assertEqual(len(np.unique(codes)), 10)
        self.assertTrue((a[0] < a[1]).all())
        self.assertFalse(np.isin(codes, positives[0] * 8 + positives[1]).any())

    def test_dense_complement_sampling_and_insufficient_negatives(self):
        positives = np.array([[0, 0, 1, 1], [1, 2, 2, 3]])
        sampler = NegativeSampler(4, positives, 2)
        self.assertEqual(set(map(tuple, sampler.sample(2).T)), {(0, 3), (2, 3)})
        with self.assertRaises(ValueError):
            sampler.sample(3)


class DataAndClusteringTests(unittest.TestCase):
    def test_aligned_loading_keeps_empty_documents_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            text, labels = Path(directory) / "문서.txt", Path(directory) / "라벨.txt"
            text.write_text("첫째\n\n셋째\n", encoding="utf-8")
            labels.write_text("a\nb\nc\n", encoding="utf-8")
            documents, y, _ = load_aligned_files(text, labels)
            self.assertEqual(documents, ["첫째", "", "셋째"])
            self.assertEqual(len(y), 3)
            labels.write_text("a\nb\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_aligned_files(text, labels)

    def test_reuters_prefix_and_clare_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            text, label = Path(directory) / "x", Path(directory) / "y"
            text.write_text("earn\tThe Market rose 123!\n", encoding="utf-8")
            label.write_text("earn\n", encoding="utf-8")
            docs, _, stripped = load_aligned_files(text, label, True)
            self.assertEqual(docs, ["The Market rose 123!"])
            self.assertEqual(stripped, 1)
            self.assertEqual(preprocess_documents(docs), ["market rose"])

    def test_control_characters_do_not_create_additional_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            text, label = Path(directory) / "x", Path(directory) / "y"
            text.write_text("alpha\fbravo\vcharlie\nsecond article\n", encoding="utf-8")
            label.write_text("a\nb\n", encoding="utf-8")
            documents, y, _ = load_aligned_files(text, label)
            self.assertEqual(len(documents), 2)
            self.assertEqual(len(y), 2)

    def test_paper_settings_and_cluster_count_absent_from_training_config(self):
        bbc, news = paper_config(), paper_config("20newsgroup")
        self.assertEqual((bbc.knn_k, bbc.hidden_dim, bbc.embedding_dim, bbc.epochs), (50, 64, 32, 200))
        self.assertEqual((news.knn_k, news.hidden_dim, news.embedding_dim), (20, 128, 64))
        self.assertNotIn("n_clusters", bbc.to_dict())

    def test_hungarian_permutation_and_rectangular_contingency(self):
        perfect = clustering_metrics(["a", "a", "b", "b"], [9, 9, 2, 2])
        assert_allclose(list(perfect.values()), [1, 1, 1])
        split = clustering_metrics([0, 0, 1, 1], [0, 0, 2, 3])
        self.assertEqual(split["acc"], .75)

    def test_spherical_update_unit_centers_cosine_assignment_and_objective(self):
        points = np.array([[1., 0], [.9, .1], [.6, .8], [-1, 0], [-.5, -.5]])
        model = SphericalKMeans(2, n_init=1, max_iter=1, seed=7)
        labels = model.fit_predict(points)
        normalized = normalize(points)
        centers = normalized[np.random.default_rng(7).choice(5, 2, replace=False)]
        initial_labels = (normalized @ centers.T).argmax(axis=1)
        expected = normalize(np.array([normalized[initial_labels == k].sum(axis=0) for k in range(2)]))
        assert_allclose(model.cluster_centers_, expected)
        assert_allclose(np.linalg.norm(model.cluster_centers_, axis=1), [1, 1])
        assert_array_equal(labels, (normalized @ expected.T).argmax(axis=1))
        self.assertAlmostEqual(model.objective_, (normalized @ expected.T).max(axis=1).sum())
        # 일반 K-means의 중심은 단위 벡터가 아니다. 매 update의 구면 투영을 검증한다.
        km = KMeans(2, init=centers, n_init=1, max_iter=1).fit(normalized)
        self.assertFalse(np.allclose(np.linalg.norm(km.cluster_centers_, axis=1), 1))


class PipelineTests(unittest.TestCase):
    def test_full_save_reuse_changed_k_no_training_and_label_independence(self):
        with tempfile.TemporaryDirectory(prefix="KARL 한글 공백 ") as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            documents = ["apple orchard harvest fruit", "apple fruit harvest orchard",
                         "banana tropical fruit yellow", "banana yellow tropical fruit",
                         "rocket planet space flight", "space rocket flight planet",
                         "galaxy planet orbit space", "galaxy space orbit planet"]
            (data / "bbc.txt").write_text("\n".join(documents) + "\n", encoding="utf-8")
            label_file = data / "bbc_labels.txt"
            label_file.write_text("\n".join(map(str, [0, 0, 0, 0, 1, 1, 1, 1])) + "\n", encoding="utf-8")
            config = Config(epochs=2, knn_k=1, hidden_dim=4, embedding_dim=2, n_init=2, threads=1)
            with contextlib.redirect_stdout(io.StringIO()):
                run = run_pipeline(config, root / "실행 결과", data, n_clusters=2)
                before = {name: (run / name).read_bytes() for name in
                          ("embeddings.npy", "model.pt", "adjacency.npz", "loss.csv", "config.json")}
                with patch("karl.main.train_embedding", side_effect=AssertionError("재학습 금지")), \
                     patch("karl.main.build_graph", side_effect=AssertionError("재구성 금지")):
                    _, result = save_clustering(run, 3, n_init=2, threads=1)
                self.assertFalse(result["embedding_retrained"])
                for name, contents in before.items():
                    self.assertEqual((run / name).read_bytes(), contents)
                # 라벨만 바꿔도 그래프·임베딩·epoch 손실은 동일해야 한다.
                label_file.write_text("\n".join(map(str, [8, 2, 5, 8, 1, 1, 5, 2])) + "\n", encoding="utf-8")
                other = run_pipeline(config, root / "라벨 교체", data, evaluate=False)
            assert_array_equal(np.load(run / "embeddings.npy"), np.load(other / "embeddings.npy"))
            assert_array_equal(sparse.load_npz(run / "adjacency.npz").toarray(),
                               sparse.load_npz(other / "adjacency.npz").toarray())
            self.assertEqual(json.loads((run / "training.json").read_text())["selected_epoch"], 2)
            # 저장 배열 오염은 군집화 전에 검출한다.
            with (run / "embeddings.npy").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                save_clustering(run, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
