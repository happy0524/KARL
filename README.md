# 그래프 기반 문서 임베딩과 군집화

KARL은 이 작업 폴더의 구현 이름입니다. Jung·Ka (2022)의 **GAE-Based Document Embedding Method for Clustering** 재구현이며, 명칭 변경으로 새로운 방법을 제안하는 것은 아닙니다.

요약 : 용량이 커서 복잡도가 큰 문서들 사이의 유사도 관계를 Graph로 구성하고, Graph Autoencoder(GAE)를 활용하여 문서의 임베딩을 학습합니다.

학습된 임베딩은 K-means와 Spherical K-means를 적용하여 문서를 Clustering하고, NMI·ACC·ARI등의 평가 지표로 성능을 평가합니다. **임베딩 학습에 군집 수를 지정하지 않는다는 장점이 있으며, 학습이 끝난 뒤 같은 임베딩으로 군집 수만 바꾸어 재사용할 수 있어 효율이 좋습니다.**

## 처리 과정

1. 문서를 전처리하고 TF-IDF로 Feature를 표현합니다.
2. 문서 간 유사도를 바탕으로 그래프를 구성합니다.
3. GAE를 학습하여 문서 임베딩을 생성합니다.
4. 임베딩을 군집화하고 결과를 평가합니다.

## 실행 방법

Python 3.10 기준, 명령어는 **프로젝트 루트의 PowerShell**에서 실행하기

### 환경 설정

기존 `.venv`에 의존성이 설치되어 있다면 이 단계는 생략.

```powershell
py -3.10 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-karl.txt
```

### BBC 데이터셋 실행

설정 파일 : `configs/karl/bbc.json`

```powershell
# 전체 학습과 군집화 평가
& ".\.venv\Scripts\python.exe" -X utf8 -m karl run --config configs/karl/bbc.json

# 소규모로 먼저 동작 확인하기(문서 128개, 3 epoch)
& ".\.venv\Scripts\python.exe" -X utf8 -m karl run --config configs/karl/bbc.json --max-samples 128 --knn-k 5 --epochs 3 --n-init 3
```

기본 실행 장치는 CPU입니다. GPU를 사용하려면 명령 끝에 `--device cuda:0`을 추가하면 됨.

### 저장된 임베딩 재사용

`--run-dir`에는 기존 학습 결과 폴더를 지정합니다. 아래 예시는 재학습 없이 군집 수를 6으로 변경합니다.

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m karl cluster --run-dir "results/karl/<기존 학습 결과 폴더>" --n-clusters 6
```

## 실행 결과

결과는 기본적으로 `results/karl/<dataset>/<실행별 폴더>/`에 저장됨.

| 주요 파일 | 내용 |
|---|---|
| `embeddings.npy`, `model.pt` | 학습된 문서 임베딩과 모델 |
| `adjacency.npz` | 문서 관계 그래프 |
| `loss.csv` | 학습 손실 기록 |
| `clustering/` | 군집화 평가 지표와 예측 결과 |
| `summary.json` | 실행 상태와 결과 요약 |

실험 설정과 실행 환경도 함께 기록합니다. 구현 검증 및 실험 결과는 [VALIDATION.md](docs/karl/VALIDATION.md), 논문과 구현의 대응 관계 및 가정은 [REIMPLEMENTATION.md](docs/karl/REIMPLEMENTATION.md)에서 확인할 수 있습니다.


## 데이터 준비

데이터·논문 PDF·학습 결과는 이 저장소에 포함하지 않습니다. BBC 실행에는 프로젝트 루트의 data/bbc.txt와 data/bbc_labels.txt가 필요합니다. 문서와 라벨은 한 줄씩 같은 순서로 대응해야 합니다. 다른 위치는 --data-dir로 지정할 수 있습니다.

## 검증

python -m unittest discover -s tests -p test_karl.py -v
