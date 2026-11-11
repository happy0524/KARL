# 논문 대조 및 구현 기록

2026-11-11 명칭 변경: KARL은 로컬 구현·패키지의 이름이다. 원 논문의 제목·저자·DOI와 알고리즘은 그대로이며 새로운 방법으로 주장하지 않는다. 아래 경로·명령·약칭은 현재 이름에 맞춰 변경했다. 과거 실험의 SHA-256은 당시 기록을 보존했으므로, 이름이 변경된 현재 소스 파일의 해시와는 다를 수 있다. 명칭 변경 검증은 `rename_validation.json`을 참조한다.

## 기준과 출처

- Sungwon Jung, Sangmin Ka, **GAE-Based Document Embedding Method for Clustering**, IEEE Access 10, 130089–130096 (2022). DOI: [10.1109/ACCESS.2022.3228548](https://doi.org/10.1109/ACCESS.2022.3228548). [서강대학교 논문 서지](https://scholarworks.sogang.ac.kr/item/c4e7dc0f-d744-48d8-ae12-bbe23fc075f4)에서도 저자·연도·DOI를 확인했다.
- 로컬 `GAE-Based_Document_Embedding_Method_for_Clustering(가상민).pdf` 8쪽 전체의 관련 부분을 읽었고, PDF 3·4·5·6쪽의 수식·Algorithm 1·설정·표를 이미지로 확인했다. 논문 첫 쪽은 **CC BY 4.0**을 명시한다. 원 저자 코드는 제공되지 않았고 검색에서도 확인하지 못했다. 원 구현과의 비트 단위 일치를 주장하지 않는다.
- CLARE 소스의 사용 조건은 아래 재사용 코드 고지의 **MIT License, Copyright (c) 2025 CLARE authors**다. 이 고지를 보존한다. README와 동봉 논문은 익명화된 출처이므로 실제 저자·원 저장소 URL을 확인할 수 없다. 논문 라이선스와 코드 라이선스는 서로 별개다.
- 데이터별 배포 버전·이용 조건 문서는 로컬에 없다. 논문 라이선스나 CLARE MIT를 데이터 라이선스로 간주하지 않는다. 데이터 재배포·외부 업로드·공개 배포는 수행하지 않았다.

## 논문 → 구현 → 검증

쪽 번호는 로컬 PDF 기준이다. Eq.(1) 외 수식에는 논문 자체의 번호가 없어 위치로 식별한다.

| 논문 위치·규칙 | 구현 파일·함수 | 검증 |
|---|---|---|
| II-A, PDF 2–3쪽, Eq.(1): TF-IDF cosine, 대각 0, 행 최대값 나누기 | `karl/data.py:vectorize_documents`, `karl/graph.py:normalized_cosine` | `test_cosine_diagonal_row_max_and_zero_row`: 비대칭성과 0행 수치 |
| Definition 2.1 / Algorithm 1 lines 4–8: 내림차순 Top-K, t=alpha·K번째, 최상위부터 시작 | `karl/graph.py:select_neighbors` | 고정 후보 순서·임계값·동점·짧은 목록 검사 |
| Algorithm 1 lines 10–15: **NC[candidate,b] >= t**, b는 이미 선택한 이웃 | 같은 함수 | `test_algorithm1_prunes_and_rejected_neighbor_cannot_support`, `test_candidate_row_index_inclusive_boundary_and_sequential_acceptance` |
| II-A 무방향 가중 그래프 | `graph_from_normalized` | 기대 전체 인접행렬, 비대칭 중복 병합 수치 검사 |
| II-B, PDF 3쪽: Ahat=D^-1/2(A+I)D^-1/2 | `normalized_adjacency`, `karl/model.py:sparse_tensor` | 가중치 .25, 고립 노드 예제의 정규화 기대값 검사 |
| II-B one-hot X=I; III-C, PDF 5쪽: 2-layer, 중간 ReLU, 최종 선형 출력, bias | `karl/model.py:IdentityGCN` | dense identity 식과 출력·모든 parameter gradient 일치; 음수 최종 출력 유지 |
| II-B, PDF 4쪽: decoder=sigmoid(z_u·z_v), 목표 M=1[A>0] | `reconstruction_loss`, `positive_pairs` | dense 전체 decoder의 선택된 항과 값·gradient 일치; 가중치와 독립인 이진 target |
| II-B 손실: 양·음성 각각 sum/N, 매 epoch 같은 수의 새 비연결 쌍 | `NegativeSampler`, `reconstruction_loss`, `train_embedding` | 0 logits에서 4log(2)/5, 비영 logits 수계산; 양성·역방향·자기 자신·중복 제외 및 재추출 |
| III-C 설정: alpha=.2, Adam .01, 200 epoch; 20News 128→64/K20, 나머지 64→32/K50 | `karl/config.py:paper_config`, `configs/karl/*.json` | 기본값 검증, 전체 BBC 200 epoch 실행 |
| II-B / III-D: 임베딩 한 번 학습, 군집 수 변경 시 재사용 | `karl/main.py:run_pipeline`, `save_clustering` | 학습·그래프 호출을 강제 실패로 대체해도 k 변경 성공; 모델·임베딩·손실 파일 불변 |
| III-B 평가: NMI/ARI/ACC | `karl/clustering.py:clustering_metrics` | 라벨 permutation 완전 일치, 직사각형 contingency ACC=.75 |
| II-B/III-C: K-means와 SK-means | `karl/clustering.py:SphericalKMeans`, `cluster_and_evaluate` | 매 중심 update를 단위 구면에 투영, cosine 할당·목적함수·최종 중심 재할당 검증 |
| 데이터/라벨은 동일 문서 순서; 라벨은 평가용 | `karl/data.py:load_aligned_files`, `load_dataset` | 빈 행 유지·행 수 불일치 중단·Reuters prefix 제거·한글 경로 검사; 라벨 교체 시 그래프·임베딩 동일 |

## 논문 명시와 구현상 가정

사용자는 추가 원 저자 자료가 없으며, 모호한 부분은 CLARE의 방향을 참고하되 KARL에 명시된 규칙을 우선하라고 답했다. 다음 선택은 결과를 보기 전에 고정했으며, 지표에 맞춰 탐색하지 않았다.

1. **가중치 병합은 가정이다.** 행 정규화 NC는 비대칭이고, 논문은 이를 무방향으로 합치는 방법을 정의하지 않는다. CLARE의 `construct_graph_selective_filtering`과 동일하게 원점 노드를 0…N−1 순회하고, 같은 무방향 쌍이 재선택되면 마지막 선택 방향의 NC를 사용한다. 결과는 대칭인 단일 A다. 한 방향만 선택되면 그 가중치를 사용한다. 합/평균/최댓값은 사용하지 않는다. 이는 원본 행 순서에 의존하며, 원 저자와 일치하는지 확인 불가다. 병합 전 선택 방향도 저장한다.
2. **degree는 일부 해석이 필요하다.** 논문은 G의 degree라고만 쓰고 가중 합인지 이웃 개수인지 엄밀히 정의하지 않는다. 여기서는 가중 그래프의 degree를 `sum_j A[i,j]`로 해석한다. 수식은 D를 원래 G에서 만들고 A+I를 따로 쓰므로 **self-loop 추가 전** A로 D를 계산한다. 일반적인 PyG 기본 GCN의 `sum(A+I)`와 다르다. 수동 sparse 연산만 수행하므로 자동 정규화·self-loop 중복이 없다. 이 선택이 저자 PyG 코드와 같은지는 확인 불가다.
3. **양성 쌍은 무방향 E를 한 번 집계한다.** `(u,v), u<v`만 사용한다. encoder에는 A 양방향 항과 self-loop가 있지만, loss에는 self-loop가 없다. 음성도 자기 자신과 양성의 양 방향을 제외한 unordered 쌍에서 균등하게 중복 없이 `|E|`개를 매 epoch 추출한다. 논문의 E가 무방향 집합인 점을 따른다. PyG의 양방향 edge_index를 그대로 두 번 세는 방식과는 손실 크기가 다르다.
4. **손실은 edge 평균이 아니다.** `sum(softplus(-pos_logits))+sum(softplus(neg_logits))`를 **노드 수 N**으로 나눈다. 안정적인 log-sigmoid 동치식이며 A의 실수 가중치는 encoder에서만 사용한다. 음성 쌍이 부족하면 목적함수를 바꾸거나 중복 허용하지 않고 오류로 중단한다.
5. **일반식과 실험식의 차이:** II-B 일반 layer-wise 식은 매 층 ReLU를 쓰지만 III-C 실제 2-layer 식에는 마지막 ReLU가 없다. 실험식을 우선했다. Bias는 CLARE/PyG처럼 모든 노드가 공유하는 feature별 벡터, 초기값은 0으로 해석했다. 가중치는 Xavier uniform이다. 초기화·bias 공유 범위는 논문에 자세히 없다.
6. **빈 행·동점·짧은 Top-K:** cosine이 양수인 자기 이외 후보만 사용한다. 동점이면 원본 행 번호가 작은 문서 우선이다. 양수 후보가 K개 미만이면 그 수만 사용하고 마지막 유효 이웃으로 t를 정한다. 모든 유사도가 0이면 이웃을 만들지 않는다. 논문은 이 경우를 설명하지 않는다. 이웃이 없으면 encoder의 D만 1로 대체해 자기 변환을 유지한다. 가짜 엣지를 추가하지 않는다. 빈 문서도 원본 행 번호와 라벨 위치를 유지한다. 모든 문서가 무어휘이면 명확한 오류로 중단한다.
7. **전처리는 CLARE 참조 가정이다.** 논문은 'meaningful words'라고만 쓰고 tokenizer·불용어·stemming·TF/IDF 정의를 제시하지 않는다. CLARE의 소문자 `[a-z]+`, 영문 불용어 제거, 길이 4 이상 규칙을 옮겼다. KARL에는 CLARE의 누적 TF-IDF pruning을 적용하지 않는다. TF-IDF는 sklearn의 unigram/count TF/smoothed IDF/L2 norm 기본 정의를 명시적으로 설정하고 제한 없는 어휘를 사용한다. stopword·어휘 변경 실험이나 라벨 기반 튜닝을 수행하지 않았다.
8. **최종 epoch 고정:** 200 epoch 완료 후 마지막 모델의 임베딩을 한 번 저장한다. early stopping, best ACC/NMI epoch 선택, validation edge split은 없다. 개별 라벨은 학습 함수의 인수가 아니며 그래프·임베딩·epoch 선택에 쓰지 않는다. 벤치마크의 알려진 군집 수만 최종 clustering에 제공한다.
9. **군집화 세부값은 가정이다.** 논문에 n_init/seed/tolerance가 없다. K-means는 raw 임베딩, k-means++/n_init20/max_iter300, sklearn 기본 tol=1e-4다. SK-means는 문서 벡터와 매 update의 중심을 정규화하고 cosine 최대 할당을 한다. 무작위 서로 다른 문서 중심/n_init20/max_iter300, 할당 수렴 시 종료, cosine 합이 가장 큰 초기화를 선택한다. 빈 군집은 현재 가장 설명되지 않는 문서 방향으로 재초기화한다. 영벡터는 0으로 유지하며 cosine 동점은 작은 cluster index 우선이다. 일반 K-means를 입력 정규화만 하여 대체하지 않았다.
10. **동등한 메모리 절약:** `I@W0=W0`를 사용하여 N×N identity를 만들지 않는다. decoder는 loss에 필요한 쌍의 내적만 계산한다. 고정 예제에서 dense 식과 값·gradient를 검증했다. 그래프용 cosine NC는 정확한 전체 N×N 행렬이며 근사 이웃 탐색을 사용하지 않는다. 20News에서는 float64 NC만 약 2.84GB이므로 해당 데이터셋의 메모리 실행은 이번에 검증하지 않았다.

## CLARE 검토와 변경 구분

| 검토 대상 | 판단 및 최종 처리 |
|---|---|
| `src/data_utils.py` | 전처리 규칙을 `karl/data.py`로 옮김(MIT). 빈 행 제거·errors=ignore·길이 잘라 맞추기는 재사용하지 않고 엄격한 순서 보존 loader 작성 |
| `src/embeddings.py` | TF-IDF 도구만 참고. 누적 가중치 pruning, BGE/ResNet, hybrid similarity, cache는 사용하지 않음 |
| `src/graph_utils.py` | 선택적 엣지 조건의 candidate/b 인덱스를 PDF와 대조. 무방향 마지막값 병합 규칙을 따름. pruning 대체 방식·2-hop edge·PyG negative sampling은 재사용하지 않음 |
| `models/proposed_gae.py` | GCN·one-hot 개념만 참고. 기존 StandardGCN에도 dropout이 있어 KARL encoder를 별도로 사용. AdamW·scheduler·GELU·dense+binary·2-hop loss·edge mean·early stopping 배제 |
| `src/clustering_eval.py` | NMI/ARI와 Hungarian ACC 정의를 확인. 별도 KARL 코드에서 같은 정의 사용. 불필요한 L2 K-means 비교는 제외 |
| `models/spherical_kmeans.py` | 구면 중심 update를 확인. 기존 구현은 마지막 update 전 labels를 저장할 수 있어, KARL에서는 최종 중심으로 다시 할당하는 독립 구현 사용 |
| `src/config.py`, `src/main.py` | 기존 경로·기본값 그대로 보존. 별도 `karl` CLI, config, 결과 디렉터리 추가 |
| `experiments/` | CLARE 민감도·제거·특징 실험과 과거 지표를 확인. KARL 학습에 연결하거나 자동 순회하지 않음 |

작업 시작 시 `karl/{config,data,graph,model,clustering,__init__}.py`가 이미 있었다. 초안 원본은 `tmp/karl_initial_20261108/karl/`, 최초 전체 파일 해시는 같은 폴더의 `files_sha256.json`에 보존했다. 기존 `docs/karl/original_files_sha256.json`도 보존했다. config/data/graph/model 초안을 수정했고 clustering/__init__은 그대로 사용했다. 새로 작성한 것은 실행 진입점, JSON config, 최소 의존성 파일, 검증 테스트, 재구현 문서와 결과다. 기존 CLARE 또는 저자의 KARL 코드를 새 알고리즘으로 주장하지 않는다.

`preservation.json`에 기존 65개 파일의 비교 결과가 있다. 변경은 위 KARL 초안 4개에 한정되며 CLARE 소스·데이터·기존 결과·README·LICENSE는 모두 원래 해시와 같다. Git 메타데이터가 없어 commit/remote/기존 Git diff는 확인 불가다.


## 재사용 코드 고지

CLARE에서 참고·옮긴 전처리 코드의 원 고지를 보존합니다. 아래 고지는 원 코드에 대한 것이며 프로젝트 전체의 별도 라이선스 지정은 아닙니다.

MIT License

Copyright (c) 2025 CLARE authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
