# DeepLearning2026-2

식물 잎 이미지의 작물 3종과 건강 상태 2종을 분류합니다. 외부 사전학습 가중치 없이 제공 데이터만 사용합니다.
사용자가 보고한 최근 **리더보드** 점수는 평균 Macro-F1 **0.9739**, 작물 **0.9662**, 건강 **0.9817**입니다.
이전 검증 점수와는 평가 데이터가 다르므로 직접 비교하지 않습니다. 아래 개선안의 전체 GPU 성능은 아직 측정하지 않았습니다.

## 기존 Colab에서 실행

학습을 종료한 상태에서 기존 세 파일만 교체합니다. 추가 Python 파일이나 Drive 연결은 필요 없습니다.
기존 `runs/deep_seed42`의 모델·가중치는 유지하세요.

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

다음 셀에서 본인이 제공 데이터로 학습한 deep 최고 모델을 출발점으로 3개 설정을 각각 120 epoch 학습합니다.
`--init_weights`에는 외부 사전학습 모델을 넣지 마세요. 이전 작은 standard 모델은 구조가 달라 사용할 수 없습니다.

```python
!python train.py --search --run_dir runs/max_refine \
    --init_weights runs/deep_seed42/best_model.pt
```

새 경로 `runs/max_refine`에 결과를 저장합니다. 각 설정의 최고 raw/EMA 모델, 학습 후 BN 통계 재계산 후보,
SWA 후보, 시작 모델을 검증하고 최종 한 모델을 저장합니다. 같은 출발점에서 학습한 가중치를 평균하는 후보도
검증 점수가 높아질 때만 선택합니다. 선택한 모델의 검증 점수는 포함한 시작 모델보다 낮아지지 않으며,
**리더보드 점수의 개선이나 0.9800 도달은 보장하지 않습니다.** test 라벨을 학습이나 선택에 사용하지 않습니다.

완료 후 공식 검사기를 실행합니다.

```python
!python check_submission.py --model runs/max_refine/model.py \
    --weights runs/max_refine/best_model.pt --data_dir student_data_128
```

제출 파일은 **같은 폴더의 `model.py`와 `best_model.pt`**입니다. 새 모델 파일과 과거 가중치를 직접 짝지어 제출하지 마세요.

```python
import zipfile
from google.colab import files
with zipfile.ZipFile('/content/submission.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ['model.py', 'best_model.pt']:
        archive.write('runs/max_refine/' + name, arcname=name)
files.download('/content/submission.zip')
```

학습이 중단되면 같은 런타임에서 같은 명령에 `--resume`을 추가합니다.
이미 완료한 child 실험은 추가 epoch를 학습하지 않고 재사용합니다. 설정·소스·출발 가중치가 바뀌면 재개를 거부합니다.
런타임이 삭제되면 `/content`의 파일도 사라지므로 제출 파일과 필요한 체크포인트를 다운로드하세요.

```python
!python train.py --search --run_dir runs/max_refine \
    --init_weights runs/deep_seed42/best_model.pt --resume
```

이전 최고 가중치가 없는 새 런타임에서는 `--init_weights`를 생략할 수 있습니다.
이때 각 실험을 처음부터 학습하고, 채널이 정렬되지 않은 서로 다른 초기화의 가중치는 평균하지 않습니다.
단일 실험은 `!python train.py`로 120 epoch 학습합니다. 기본 경로는 `runs/refined_seed42`입니다.
같은 run 이름을 덮어쓰지 않으므로 새 실험에는 다른 `--run_dir`을 사용하세요.

## 개선 내용과 제한

- [model.py](project1/model.py): 7개 Residual 블록, BatchNorm, SE attention, 평균/최대 pooling,
  두 LayerNorm/Dropout 분류 헤드. 16×16 중간 특징과 8×8 최종 특징을 결합하는 경로를 추가했습니다.
  새 경로의 0 초기화 덕분에 과거 deep 가중치를 학습용으로 가져오면 이전 예측을 정확히 재현합니다.
  추론 함수는 입력 이미지 한 장으로 예측하며 평가 시 추가 이미지 증강을 하지 않습니다.
- [dataset.py](project1/dataset.py): gentle/standard 증강 외에 전체 잎·병변 픽셀을 보존하는 detail/finish 증강.
  학습 후반 20%는 crop·색상 변화를 줄이고 MixUp과 label smoothing을 해제합니다.
  평가 전처리는 공식 **ToTensor + Normalize(mean=0.5, std=0.5)** 그대로입니다.
- [train.py](project1/train.py): AdamW weight decay, warmup/cosine, train 라벨 기반 클래스 가중치,
  CUDA AMP, gradient clipping, MixUp, 학습 전용 stochastic depth, EMA와 SWA.
  BN 통계 재계산에는 **증강 없는 train 이미지**만 사용하고 dropout은 끕니다.
  `--search`는 세 설정·seed를 비교하며 최종 가중치를 평균하더라도 모델은 한 개입니다.
- [check_submission.py](project1/check_submission.py): 공식 검사기를 변경 없이 보존했습니다.

현재 모델은 파라미터 **1,874,597개** + 실수형 버퍼 **3,840개** = **1,878,437개**로 2M 이내입니다.
FP32 `state_dict`는 **약 7.22 MiB(7.57 MB)**로 10 MB 이내입니다. 제출 `model.py`는 torch와 표준 라이브러리만 사용합니다.
이미지 로딩에 사용하는 Pillow는 torchvision의 의존성입니다. 데이터와 가중치는 GitHub에 올리지 않습니다.

`--search`의 설정은 gentle + 약한 MixUp, detail + MixUp 없음, standard + MixUp의 세 후보입니다.
각각 dropout·stochastic depth·학습률·규제를 달리하고 검증 Macro-F1의 평균으로 선택합니다.
독립적으로 초기화된 모델을 임의로 평균하거나 여러 모델을 제출 파일 안에 숨기지 않습니다.

## 결과와 재현

최종 run에는 `model.py`, `best_model.pt`, `best_metrics.json`, `evaluation.json`, `search_results.json`이 남습니다.
`trial1_seed42`, `trial2_seed43`, `trial3_seed44`에는 각각 소스, 설정/해시, epoch 기록, 재개용 `last_checkpoint.pt`를 보관합니다.
단일 실험은 같은 구조의 기록을 해당 run에 바로 저장합니다. `last_checkpoint.pt`는 제출 파일이 아니며 10 MB 제한 대상이 아닙니다.

`--eval_only`, `--no_augment`, `--mixup_alpha 0`, `--swa_start 1`, `--ema_decay 0`,
`--no_bn_recalibration`으로 단일 실험에서 각 효과를 확인할 수 있습니다.
`--class_weight_power`와 과제별 `--crop_class_weight_power`, `--health_class_weight_power`도 설정할 수 있습니다.
과제별 설정이 전체 설정보다 우선합니다. 재개에는 같은 학습 옵션과 총 epoch 수를 유지하세요.

## 자동 준비 노트북과 검증

[Colab에서 열기](https://colab.research.google.com/github/yeonji200522-oss/DeepLearning2026-2/blob/main/colab_train.ipynb)를 사용하면
공식 코드·데이터와 학습 파일을 자동으로 준비합니다. GPU 런타임을 선택하고 셀을 순서대로 실행하세요.
노트북은 현재 런타임의 이전 deep 가중치를 자동으로 찾고, 있으면 추가 학습하고 없으면 처음부터 학습합니다.
최종 후보와 남아 있는 이전 실험을 공식 CPU 검사기로 비교한 뒤 제출 ZIP을 다운로드합니다.
Drive는 사용하지 않으며 리더보드 자동 제출도 하지 않습니다.

```bash
python -m unittest discover -s tests -v
PLANT_DATA_DIR=/path/to/student_data_128 python -m unittest discover -s tests -v
python scripts/build_colab.py
```

테스트는 출력 규격·크기·가중치 호환성·평가 전처리·MixUp 라벨 대응·BN 재계산을 확인합니다.
실제 데이터 경로를 지정하면 학습 중단/재개, SWA 저장, 세 파일만으로 학습, search와 단일 모델 내보내기도 검사합니다.
CPU 소규모 학습 점수는 전체 학습 성능이 아닙니다. CUDA AMP와 전체 성능은 Colab GPU에서 확인해야 합니다.

데이터·과제 규격 출처는 [Class2026_DL/project1](https://github.com/maygodwithu/Class2026_DL/tree/main/project1)입니다.
자동 준비 노트북은 공식 커밋 `c0c7737fa2355cf57bf78792b376bf770aa950dd`를 고정해 사용합니다.
기존 `model_py.ipynb`는 보존했습니다.
