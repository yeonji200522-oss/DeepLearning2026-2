# DeepLearning2026-2

식물 잎 이미지의 작물 3종과 건강 상태 2종을 분류합니다. 목표는 평균 Macro-F1을 더 높이는 것입니다.
최신 실험은 **기존 best_model.pt 업로드 없이 처음부터 학습**합니다. 같은 버전의 미완료 학습이 있으면 자동 재개합니다.
기본 실행은 한 모델의 단일 학습이며
교차검증과 여러 recipe 검색을 실행하지 않습니다. 외부 사전학습 가중치를 사용하지 않습니다.

사용자가 제공한 직전 실험(`fc94a63`)의 public은 **0.9766**(작물 0.9705, 건강 0.9826),
valid는 **0.9853**(작물 0.9791, 건강 0.9916)입니다. 이전 public 0.9739보다 0.0027 높습니다.
valid Apple F1은 0.9688이며 Apple 360장 중 18장을 놓쳤고 그중 15장을 Tomato로 예측했습니다.
Grape 429장 중 9장도 Tomato로 예측했습니다. 다음 실험은 작물의 공간 특징과 소수 작물 학습을 보강합니다.
최신 실험의 public **0.9900**은 목표이며, 도달 여부와 전체 GPU 성능은 아직 확인하지 않았습니다.
이후 다른 모델에서 나온 valid 0.9822/public 0.9723을 이 새 모델의 성능으로 기재하지 않습니다.

## Colab에서 세 파일 교체 후 실행

현재 학습을 종료한 상태에서 다음 셀을 실행합니다. 추가 Python 파일이나 Drive 연결은 필요 없습니다.
기존 실험의 model.py와 best_model.pt는 같은 한 쌍으로 보존하세요.

```python
%cd /content/Class2026_DL/project1
from pathlib import Path
from urllib.request import urlopen
base = "https://raw.githubusercontent.com/yeonji200522-oss/DeepLearning2026-2/main/project1/"
names = ["model.py", "dataset.py", "train.py"]
sources = {name: urlopen(base + name).read() for name in names}
for name, source in sources.items():
    Path(name).write_bytes(source)
    print("교체 완료:", name)
```

GPU 런타임인지 확인한 뒤 기존처럼 실행합니다.

```python
import torch
assert torch.cuda.is_available(), "런타임을 GPU로 변경하세요."
!python train.py
```

기본값은 **고정 epoch 상한 없이 valid 수렴까지 학습**, batch 64, learning rate 0.001입니다. 저장 폴더를 지정하지 않으면
동일 코드·설정의 가장 최근 미완료 run을 재개합니다. 호환 run이 없으면 `runs/single_seed42_<날짜와 시간>`을
자동 생성합니다. 이미 완료된 run은 보존하며, 실행 중인 같은 run이 있으면 중복 GPU 학습을 차단합니다.
시작 로그의 `run:`에 경로가 표시됩니다. 기존 best_model.pt가 프로젝트 폴더에 있어도 자동으로 불러오지 않습니다.
매 epoch 첫 배치·25배치마다·마지막 배치의 진행 상황과 epoch 종료 후 valid F1을 출력합니다.

사용자 요청에 따라 **성능을 우선**합니다. 시간 절약을 위해 뺐던 SAM·SWA와 매 epoch raw/EMA 평가를 다시 기본 설정에
포함했습니다. 이번에는 특징 추출 CNN 자체를 17개 MBConv 블록으로 교체합니다.
학습 설정은 직전 adaptive 버전을 유지해 구조 변경을 비교합니다. 각 기법의 개별 성능 이득을 측정한 것은 아닙니다.
고정 120/200 epoch를 자동 종료 기준으로 삼지 않고 학습률 감소와 정밀 학습을 거쳐 valid의 정체를 판단합니다.
학습 시간이 길어질 수 있으므로 같은 런타임의 중단/재개를 지원합니다. Colab의 할당량이나 파일 삭제는 코드로 해제하지 않습니다.

## 성능 중심의 적응 학습 단계

1. **기본 학습:** 5 epoch warmup 후 SAM(반경 0.03), MixUp·smoothing·기존 증강과 규제를 적용합니다.
2. **학습률 조절:** valid 평균 F1이 10 epoch 동안 0.0001를 넘는 개선을 보이지 않으면 LR을 절반으로 낮춥니다.
   개선되면 정체 카운터를 초기화합니다. 최저 LR은 0.00001입니다.
3. **정밀 학습:** 최저 LR에서도 정체되면 저장한 최고 모델로 되돌아가고 optimizer의 moment를 초기화합니다.
   full-pixel 증강, hard label, Dropout 0.1, SAM 반경 0.01을 사용합니다. MixUp·smoothing·stochastic depth는 끕니다.
4. **종료:** 정밀 학습을 최소 20 epoch 진행한 뒤에도 12 epoch 동안 의미 있는 valid 개선이 없으면 종료합니다.
   이 단계의 checkpoint와 가중치 평균·optimizer/RNG 상태도 저장하므로 전환 직후 중단해도 재개할 수 있습니다.

로그는 `[01/adaptive]`처럼 표시됩니다. 총 epoch가 미리 고정돼 있지 않다는 의미입니다.
`--epochs` 기본값 0은 적응 종료를 뜻합니다. 명시적으로 양수(예: `--epochs 160`)를 지정하면 이전 cosine 예산 방식으로
실행합니다. 과제의 사전학습 금지·2M·10MB·라이브러리 제한은 성능 우선 실험에서도 지킵니다.

## 이번 모델과 학습 변경

[model.py](project1/model.py)는 **17개 MBConv 블록**을 처음부터 학습하는 CNN입니다.
직전 모델의 일반 3×3 합성곱 두 개를 반복하는 7개 블록 대신,
1×1 채널 확장 → depthwise 공간 합성곱 → SE → 1×1 압축을 사용합니다.
단계별 출력 채널은 16→24→40→64→96→160, 반복 수는 1→2→3→5→4→2입니다.
마지막 1×1 합성곱으로 192채널을 만들며, 블록 내부에서는 최대 960채널로 특징을 처리합니다.
이는 torchvision 사전학습 모델을 가져온 것이 아니며, 모든 가중치를 무작위로 초기화합니다.
이번 변경에서 train.py와 dataset.py의 학습·평가 설정은 유지합니다.

| 방법 | 최신 실험에 적용한 내용과 목적 |
|---|---|
| 특징 추출 구조 | 17개 MBConv, 3×3/5×5 depthwise convolution, 4/6배 채널 확장과 잔차 연결; 작은 파라미터 수로 더 많은 처리 단계 확보 |
| 활성화 | ReLU를 SiLU로 변경; 음수 영역에도 부드러운 기울기를 제공 |
| 채널 attention | SE 형태의 attention으로 채널별 중요도를 학습 |
| 다중 스케일 | 16×16 해상도에서 9개 블록으로 세부 특징을 처리; 96채널 중간 특징의 평균·최대 pooling을 최종 8×8 특징과 결합 |
| 작물 공간 경로 | 16×16 특징을 2×2 영역으로 pooling하고 384→64→320으로 변환해 작물 표현에 추가; 위치 정보를 보강 |
| 분류 헤드 | 작물 헤드는 256→320으로 확장; 건강 헤드는 256 유지; 각각 LayerNorm·Dropout·독립 출력층 사용 |
| 잎 보존 증강 | 큰 crop 없이 수평/수직 뒤집기, 90도 회전, 작은 affine 변화·색상 변화; 기본 leaf 증강의 blur 제거 |
| 최적화 | SAM·AdamW; 기본 반경 0.03, 적응 정밀 단계 0.01 |
| BatchNorm 처리 | SAM 두 번째 pass의 running 통계 중복 갱신 방지 |
| 불균형 대응 | train 빈도의 역수 거듭제곱을 정규화한 손실 가중치; 작물 α를 0.5→0.6, 건강 α=0.35 유지 |
| 과제 비중 | 작물 손실 비중을 1.25→1.5, 건강 1.0 유지; 공유 특징의 작물 학습을 보강 |
| 규제 | AdamW weight decay 0.0003, Dropout 0.2, stochastic depth 최대 0.03, label smoothing 0.02 |
| MixUp | α=0.1, 확률 0.2로 작물·건강 라벨을 같은 순열과 비율로 혼합 |
| 학습률 | 기본 5 epoch warmup 후 valid plateau에 따라 감소; 명시적인 유한 epoch는 cosine 방식 |
| 정밀 학습 | 최저 LR에서도 정체되면 최고 모델에서 full-pixel flip/90도 회전, hard label, Dropout 0.1로 학습 |
| 가중치 평균 | EMA 및 정밀 단계 3번째 epoch부터 SWA; train 기반 BN 재추정 후보를 비교 |
| 종료 기준 | 정밀 학습 최소 20 epoch 이후 12 epoch 동안 0.0001를 넘는 valid 개선이 없으면 종료 |

이 방법들은 개선을 위한 설계입니다. 전체 학습이나 동일 조건의 제거 실험을 하지 않은 상태에서
각 기법의 향상 폭이나 private 데이터 성능을 주장하지 않습니다. AdamW의 가중치 감쇠는 적응형 optimizer의
손실에 L2 항을 추가하는 방식과 동일하다고 표현하지 않습니다.

추가 공간 경로는 작물 출력에만 연결하며 건강 출력에 직접 더하지 않습니다. 공통 CNN은 함께 학습하므로
건강 F1도 확인해야 합니다. 최신 실험은 다중 스케일·작물 공간 경로를 새 MBConv 특징에 연결하고,
기존 손실 비중과 성능 중심의 적응 학습 절차를 유지하는 후보입니다.
각 변경의 독립적인 효과를 확인한 실험은 아닙니다. 먼저 기존 valid 평균 0.9853과 작물 0.9791 대비 개선 여부를 검토하세요.

EMA와 raw를 매 epoch 공식 valid로 평가해 최고 모델을 저장합니다. 종료 후 best/last/EMA/SWA의 train 기반
BN 재추정 후보를 비교하고 valid가 더 좋은 경우에만 저장합니다. 최종 제출은 **한 CNN의 state_dict**이며
여러 모델의 예측을 동시에 실행하는 ensemble은 아닙니다. 이 선택 절차가 public 개선을 보장하지는 않습니다.

## 평가와 제출 제한

[dataset.py](project1/dataset.py)의 평가 전처리는 공식 **ToTensor + Normalize(mean=[0.5]*3, std=[0.5]*3)**를
그대로 유지합니다. 평가 시 crop·회전·blur·TTA를 추가하지 않습니다.
공식 [check_submission.py](project1/check_submission.py)는 변경하지 않았습니다.

- 파라미터 **1,927,385개** + 실수형 버퍼 **26,896개** = **1,954,281개**로 2M 이내
- FP32 plain state_dict 직렬화 확인: 약 **7.59 MiB**, 10MB 이내
- 출력 (B,3), (B,2); 제출 모델은 torch와 표준 라이브러리만 사용
- torchvision 및 그 의존성 Pillow로 학습 이미지를 읽음
- 기존 첫 모델과 새 모델은 구조가 다르므로 당시 소스와 가중치를 같은 폴더의 한 쌍으로 사용

학습 시작 로그의 실제 run 경로를 아래에 넣어 확인하세요.

```python
!python check_submission.py --model runs/실제_저장폴더/model.py \
    --weights runs/실제_저장폴더/best_model.pt --data_dir student_data_128
```

필요하면 이전 모델과 같은 valid에서 클래스별 F1, confusion matrix, 오분류와 paired bootstrap 구간을 비교합니다.
각 run에 저장된 당시 model.py로 평가하며 가중치를 변경하지 않습니다. 구간은 valid 표본의 불확실성만 반영합니다.

```python
!python train.py --compare_runs runs/이전_저장폴더 runs/새_저장폴더 \
    --run_dir runs/validation_audit
```

코드만 최신으로 교체한 후 과거 가중치를 그대로 새 모델과 제출하지 마세요. 리더보드에 자동 제출하지 않습니다.

## 재개·다운로드·보고서

같은 런타임에서 중단 후 같은 `!python train.py` 명령을 다시 실행하면 호환 미완료 run을 자동 재개합니다.
소스와 학습 설정이 일치하는 경우에만 재개하며, 이전 버전의 run은 자동 재개하지 않습니다.
특정 폴더를 직접 재개하려면 다음 명령을 사용합니다.

```python
!python train.py --run_dir runs/실제_저장폴더 --resume
```

같은 코드와 설정으로 재개하며 이전 버전에서 시작한 run은 재개하지 않습니다. 원래 명령에 추가 옵션이 있었다면
동일하게 전달하세요. `--run_dir`을 직접 지정한 새 학습은 폴더의 기존 기록을 덮어쓰지 않습니다.
`/content`는 런타임 삭제 시 사라지므로 제출 파일과 기록을 다운로드해 보관합니다.

- 제출: 해당 run의 **model.py + best_model.pt**
- 보고서·재현: **model.py + train.py + dataset.py + config.json + best_metrics.json + history.json/CSV**
- 재개: **last_checkpoint.pt**; 제출 파일이 아님

처음부터 새 실험을 강제하려면 `!python train.py --new_run`을 사용합니다. `completed.json`은 정상 종료 여부를 기록하며
완료한 실험을 자동 재개하지 않도록 합니다. `.run.lock`의 OS 잠금은 프로세스 종료 시 자동 해제되어 재개를 막지 않습니다.
Colab 노트북은 고정 RUN_NAME으로 중단된 실험을 명시적으로 재개하며 기본 GPU smoke 학습을 생략합니다.

`--sam_rho 0`, `--swa_start 1`, `--bn_candidates best`는 비교 실험에서 비용이 큰 기능을 끄는 옵션입니다.
기본 적응 학습은 종료 판단을 위해 양수 patience를 요구합니다. 명시적 cosine 예산에서는 `--patience 0`으로 조기 종료를 끕니다.
`--cv`, `--search`, `--fixed_epochs`는 기존 선택 기능으로
남아 있으나 기본 실행에는 적용되지 않습니다. 사용자 요청의 새 실험은 `!python train.py` 한 번으로 진행합니다.
`--init_weights`는 구조가 일치하는 자체 학습 가중치에만 허용되며 이번 실험에는 지정하지 않습니다.

## 검증과 출처

```bash
PLANT_DATA_DIR=/path/to/student_data_128 python -m unittest discover -s tests -v
python scripts/build_colab.py
```

25개 검사 항목으로 제출 규격·동일 평가 텐서·전체 블록 기울기·SAM perturb/restore·loss scaling·BN 통계·실제 이미지 학습·중단/재개,
자동 재개·중복 차단·LR 감소/정밀 학습 전환·전환 직후 재개의 가중치 동일성을 확인했습니다.
적응 단계의 전환 테스트는 synthetic plateau 점수를 사용하므로 전체 정확도의 증거로 사용하지 않습니다.
CPU의 작은 학습 점수는 전체 모델 성능이 아닙니다. CUDA AMP와 전체 GPU 학습은 Colab에서 확인해야 합니다.
[자동 준비 Colab](https://colab.research.google.com/github/yeonji200522-oss/DeepLearning2026-2/blob/main/colab_train.ipynb)는
동일한 scratch 단일 학습을 기본값으로 사용합니다. Drive와 기존 가중치 업로드는 필요 없습니다.

과제·데이터 출처는 [Class2026_DL/project1](https://github.com/maygodwithu/Class2026_DL/tree/main/project1)이며,
자동 준비는 공식 커밋 `c0c7737fa2355cf57bf78792b376bf770aa950dd`를 사용합니다.
첫 제출 보고서와 복구 코드의 기준은 과거 커밋 `9f23d1389feb6e224e0ba3bd86aba40ff99e4c3a`입니다.
이번 새 실험은 별도 버전이며 첫 제출의 0.9739를 새 모델의 측정 결과로 표시하지 않습니다.
