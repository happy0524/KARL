# 실제 검증 결과

2026-11-11 명칭 변경: 아래 실험 수치는 2026-11-08에 측정한 과거 결과이며, 문서의 명령·경로는 현재 패키지 이름 `karl`로 치환했다. 과거 source manifest의 SHA-256은 다시 계산하지 않고 보존했다. 명칭 변경 후의 실행·배열 보존 검증은 별도 `rename_validation.json`에 기록한다.

검증일: 2026-11-08 (한국 시간). 결과 폴더명 안의 자동 시각은 UTC다. **논문을 참고한 구현과 실행 검증은 완료했으며, 논문 성능의 완전 재현은 달성하지 않았다.**

## 실행한 검증

- `tests/test_karl.py`의 **19개 테스트 통과**. Algorithm 1의 기대 엣지·비대칭 인덱스·선택 순서·임계값 경계·동점·0행·고립 노드, 가중 degree 정규화, 음성 샘플 제약, sum/N 손실 수계산, dense와 sparse encoder/decoder의 출력 및 모든 parameter gradient 동등성을 검증했다.
- 한글·공백 임시 경로에서 전체 파이프라인, 파일 저장, k 변경 시 재학습 금지, 배열 오염 검출을 확인했다. 같은 문서에 라벨만 교체하고 재학습해 그래프와 임베딩이 정확히 같은 것도 확인했다.
- CPU BBC 128개/3 epoch smoke test: 전체 저장·두 군집화·평가 완료. 학습 손실 4.382268 → 4.218040. 이 결과는 성능 재현 결과가 아니다.
- GTX 1060 6GB의 `cuda:0` BBC 128개/2 epoch 학습·저장 완료. GPU 전체 데이터 성능은 측정하지 않았다.
- CPU 전체 BBC 2,225개/200 epoch 완료. 학습 17.48초, 전체 파이프라인 20.41초(프로세스 import 시작 시간 제외). 그래프 무방향 엣지 77,805개, 고립 노드 0개, 연결 성분 1개. Top-K 후보 111,250개 중 949개 거부. 매 epoch 양성·음성 각각 77,805쌍, 200개 음성 표본 해시가 모두 달랐다. 손실 48.476059 → 30.488750.
- 같은 전체 BBC를 한글·공백 결과 경로에서 다시 실행해 최종 임베딩, 가중 인접행렬, 200개 손실과 200개 음성 표본 해시가 **정확히 일치**했다. 데이터 loader의 물리 행 처리 보완 후에도 BBC 결과가 동일함을 확인하는 재현성 실행이다. 반복 실행 학습 17.44초, 전체 20.57초.
- 저장된 전체 임베딩의 `n_clusters=6` 군집화만 실행했다. 저장 배열의 해시가 그대로이며, 단위 테스트에서는 학습·그래프 함수를 강제 오류로 대체해 재호출이 없음을 확인했다.
- 기존 CLARE 소스·데이터·결과 해시는 모두 보존되었다. 공통 코드를 바꾸지 않았으므로 CLARE 학습은 추가 수행하지 않았다. 패키지 설치·업그레이드, 외부 업로드·공개 배포는 없었다.

테스트 로그: `results/karl/tests_stdout_20261108.txt`. 기계 판독 결과는 `docs/karl/validation.json`, 기존 파일 보존 비교는 `preservation.json`에 있다. 초기 PowerShell의 stderr 병합이 테스트 성공 출력을 NativeCommandError로 표시했지만, stdout runner로 다시 실행한 19개 테스트의 실제 종료 코드는 0이다.

## BBC 전체 결과와 논문 비교

동일 임베딩 한 개, seed=42, 최종 군집 수 5, 각 군집화 n_init=20이다. 논문 Table 2는 PDF 6쪽에서 직접 확인했다. 지표는 0–1 척도다.

| 방법 | 구분 | ACC | NMI | ARI |
|---|---|---:|---:|---:|
| K-means | 이번 실행 | 0.949213 | 0.845635 | 0.878759 |
| K-means | 논문 Table 2 | 0.962 | 0.880 | 0.911 |
| Spherical K-means | 이번 실행 | 0.947416 | 0.840030 | 0.874811 |
| Spherical K-means | 논문 Table 2 | 0.960 | 0.878 | 0.907 |

현재 구현은 논문보다 ACC 약 1.3%p, NMI 약 3.4–3.8%p, ARI 약 3.2%p 낮다. 이는 **한 seed에서의 결과**이며 논문은 seed/반복 횟수/분산을 보고하지 않아 통계적 동등성을 평가할 수 없다.

관찰한 차이 및 한계:

1. BBC 문서/라벨은 2,225/2,225행, 5종, 빈 문서 0개로 Table 1과 표본 수가 같다. 하지만 원문 어휘 집합·전처리·배포 버전이 같다는 증거는 없다. 현재 전처리 후 어휘는 **26,428개**, 논문은 **37,644개**다. 원시 파일 그대로 TF-IDF를 했을 때도 29,421개였으므로, 단순히 CLARE 전처리를 해제하는 것만으로 논문 데이터 어휘를 복원할 근거가 없다.
2. 현재 원시 문서의 공백 기준 단어 수 평균/최소/최대는 **390.30/90/4,492**, 논문 Table 1은 **372/86/4,342**다. 세는 규칙과 원본 데이터 버전이 모두 미확인이다. 이 차이는 측정된 사실이고 성능 차이의 단일 원인이라고 단정하지 않는다.
3. NC 무방향 병합·degree·self-loop·bias·양방향 중복 집계·초기화·군집화 반복 세부사항은 저자 구현으로 확인하지 못했다. 채택한 해석을 `REIMPLEMENTATION.md`에 명확히 분리했다. 논문 본문 수식의 D와 일반 PyG 기본값이 다른 점도 알려진 재현 한계다.
4. alpha=.2, K=50, 64→32, Adam .01, 200 epoch는 논문 설정이다. 라벨을 사용한 그래프 조정, 어휘 수 맞추기, best epoch 선택, seed 탐색, 파라미터 탐색은 수행하지 않았다. 반복 실행은 동일 설정의 수치 재현 확인이다.

## 실제 실행한 PowerShell 명령

프로젝트 루트에서 실행했다. 아래 `--run-dir` 폴더는 이미 존재하므로 그대로 재실행하면 덮어쓰기 방지를 위해 중단한다. 다시 학습하려면 `--run-dir`를 생략하거나 새 이름으로 바꾼다.

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m karl --help
& ".\.venv\Scripts\python.exe" -X utf8 -m unittest discover -s tests -p test_karl.py -v

& ".\.venv\Scripts\python.exe" -X utf8 -m karl run --config configs/karl/bbc.json --max-samples 128 --knn-k 5 --epochs 3 --n-init 3 --run-dir results/karl/bbc_smoke_cpu_20261108
& ".\.venv\Scripts\python.exe" -X utf8 -m karl train --config configs/karl/bbc.json --max-samples 128 --knn-k 5 --epochs 2 --device cuda:0 --run-dir results/karl/bbc_smoke_cuda_20261108
& ".\.venv\Scripts\python.exe" -X utf8 -m karl run --config configs/karl/bbc.json --run-dir results/karl/bbc_full_cpu_seed42_20261108
& ".\.venv\Scripts\python.exe" -X utf8 -m karl run --config configs/karl/bbc.json --run-dir "results/karl/BBC 재현성 확인 20261108"
& ".\.venv\Scripts\python.exe" -X utf8 -m karl cluster --run-dir results/karl/bbc_full_cpu_seed42_20261108 --n-clusters 6

# stderr 병합을 피하고 성공한 테스트 로그를 남긴 실제 명령
& ".\.venv\Scripts\python.exe" -X utf8 -c "import sys,unittest; result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_karl.py')); sys.exit(not result.wasSuccessful())" | Tee-Object -FilePath results/karl/tests_stdout_20261108.txt
```

전체 BBC 주요 결과: `results/karl/bbc_full_cpu_seed42_20261108/summary.json`, `embeddings.npy`, `loss.csv`. k=5 최초 군집화 결과의 정확한 하위 폴더 경로는 루트 `summary.json`의 `clustering_dir`에 있다. k=6 재사용 검증 결과는 K-means ACC/NMI/ARI 0.884494/0.809570/0.810547, spherical 0.861573/0.798112/0.793965이며 논문 성능 비교에 사용하지 않았다.

## 미실행 데이터셋과 구체적인 이유

데이터 정렬 사전 감사 결과는 `data_audit.json`에 원본 해시와 함께 저장했다. 아래 명령은 **미실행 예시**이며 이번에 전체 데이터셋 순회는 하지 않았다.

| 데이터셋 | 상태·미실행 이유 | 실행 명령 |
|---|---|---|
| 20Newsgroups | 18,846개/20종으로 개수는 일치. 이번 범위는 첫 데이터셋 BBC 전체 실행까지이며 추가 학습은 생략. 전체 NC 약 2.84GB의 메모리 실행은 미검증 | `python -X utf8 -m karl run --config configs/karl/20newsgroup.json` |
| Reuters-8 | 7,674개/8종. 모든 행의 category-TAB prefix를 제거하는 loader 검증 완료; 전체 학습은 범위상 생략 | `python -X utf8 -m karl run --dataset reuters8` |
| Reuters10k | **원본 45,438행 vs 라벨 10,000행 불일치**. loader가 오류로 중단함을 실제 확인. 문서 경계/ID를 확인할 자료가 없어 잘라 맞추거나 라벨 반복하지 않음 | `python -X utf8 -m karl run --dataset reuters10k` (정렬된 원본 없이는 실행 불가) |
| WebKB | 로컬 4,168개/4종, 논문 4,518개/6종으로 **서로 다른 데이터**. 기본 알려진 군집 수는 논문 기준 6이며 로컬 라벨 수에서 추론하지 않음. 논문 성능 재현용 실행은 생략 | `python -X utf8 -m karl run --dataset webkb` (로컬 데이터 동작만 확인할 수 있음) |
| Opinosis | 로컬 데이터 없음. 임의 다른 데이터로 대체하지 않음 | 지원 등록 없음; 올바른 문서·라벨 자료가 필요 |

추가로 필요한 자료는 저자 KARL 코드, 정확한 데이터 버전/전처리, Reuters10k 문서 경계 대응 정보다. BBC 실행과 코드 전달을 막는 미해결 항목은 없지만, 이 자료 없이 **원 구현과 동일한 성능을 재현했다고 선언할 수 없다.**
