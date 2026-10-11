# DeepLearning2026-2

식물 잎 이미지의 작물 3종과 건강 상태 2종을 분류합니다. 외부 사전학습 가중치 없이 제공 데이터만 사용합니다.
사용자가 보고한 최근 결과는 **valid 평균 Macro-F1 0.9822**(작물 0.9736, 건강 0.9908), **public 0.9723**입니다.
이전 public 최고는 0.9739입니다. valid와 public은 다른 데이터이며, 이 차이만으로 과적합이나 분포 변화를 단정하지 않습니다.
목표 public 0.9820 도달은 보장할 수 없습니다. 아래 절차의 전체 GPU 성능은 아직 확인하지 않았습니다.

## 먼저 기존 모델을 같은 valid에서 비교

기존 Colab에서 학습을 종료한 상태로 세 파일만 교체합니다. 추가 Python 파일과 Drive 연결은 필요 없습니다.
이전 run 폴더의 `model.py`와 `best_model.pt`는 보존하세요.

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

아래 명령은 두 run에 보관된 당시 모델 소스로 동일한 공식 valid를 평가합니다. 가중치는 수정하지 않습니다.
경로가 다르면 실제 학습 폴더 이름으로 바꾸세요. 이전 가중치가 없으면 한 폴더만 지정해 오류 보고서를 얻을 수 있습니다.

```python
!python train.py --compare_runs runs/deep_seed42 runs/refined_seed42 \
    --run_dir runs/validation_audit
```

`comparison.json`에는 클래스별 F1/confusion, 작물·건강 조합별 정확 개수, NLL과 오답의 평균 확신도를 저장합니다.
모델별 `errors_*.csv`에는 오분류 이미지 ID와 라벨/예측을 저장합니다.
첫 모델 대비 paired bootstrap 차이와 95% 구간도 계산합니다. 구간이 0을 포함하면 작은 valid 차이를 확실한 개선으로
취급하지 마세요. 이 구간은 **valid 표본의 불확실성**이며 public/private 성능에 대한 보장이 아닙니다.

## train 내부 교차 검증 후 최종 모델 고정

동일한 official valid에서 매 epoch와 여러 후보를 계속 고르는 방식의 선택 편향을 줄이기 위한 새 기본 권장 절차입니다.

```python
!python train.py --cv --run_dir runs/generalize_cv
```

이 명령은 다음을 수행합니다.

1. **train만** 작물/건강 조합으로 층화해 3-fold를 만듭니다. RGB 픽셀이 완전히 같은 이미지들은 같은 fold에 묶습니다.
2. 이전 deep 구조, 촬영 변화에 대응하는 증강, 현재 다중 스케일 구조의 **3개 사전 지정 설정 × 3-fold**를 비교합니다.
   각 fold는 **처음부터 100 epoch** 학습합니다. 전체 train을 본 과거 가중치는 사용하지 않습니다.
3. fold 평균과 표준편차를 보고, 최고 평균에서 1 standard error 이내라면 선언 순서에서 더 단순한 설정을 우선합니다.
   각 fold도 마지막 EMA 또는 미리 지정한 SWA를 고정한 뒤 평가하며, held-out fold에서 최고 epoch를 계속 찾지 않습니다.
4. 선택한 설정과 같은 epoch 예산(기본 100)으로 **전체 train을 처음부터 다시 학습**합니다.
5. 최종 EMA/SWA 가중치를 먼저 저장·해시로 고정하고 **공식 valid는 마지막 평가에만** 사용합니다.
   그 점수로 가중치·설정·epoch를 바꾸지 않습니다.

평가 전처리는 공식 **ToTensor + Normalize(mean=0.5, std=0.5)** 그대로입니다. 평가 시 추가 이미지 증강을 하지 않습니다.
`--cv_epochs`는 실험 전에 정하는 fold 학습 기간이고 `--epochs`는 최종 기간의 상한입니다.
예를 들어 더 긴 예산을 시험하려면 시작 전에 `--cv_epochs 120 --epochs 120`을 함께 지정하고 다른 run 이름을 쓰세요.
public 점수에 맞춰 예산을 반복 조정하지 않습니다.

CSV의 촬영 개체/환경 정보는 제공되지 않으므로 이 분할이 새 촬영 환경에 대한 성능을 보증하지는 않습니다.
전체 제공 데이터의 exact-pixel 검사에서는 중복 그룹 3개, train과 valid를 가로지르는 동일 이미지 0개를 확인했습니다.
이는 근접 중복이나 촬영 개체의 중복을 전부 배제했다는 의미가 아닙니다.

## 검사와 제출 전 검토

```python
!python check_submission.py --model runs/generalize_cv/model.py \
    --weights runs/generalize_cv/best_model.pt --data_dir student_data_128
```

기존 최고 모델과의 공식 valid 비교도 확인합니다.

```python
!python train.py --compare_runs runs/deep_seed42 runs/generalize_cv \
    --run_dir runs/final_validation_audit
```

`cv_results.json`의 fold 평균/분산, 클래스별 오분류, 기존 모델과의 valid 차이를 함께 보고 제출 여부를 결정하세요.
명확한 개선 근거가 없으면 기존 모델을 보존하고 제출을 보류할 수 있습니다. 이 프로젝트는 리더보드에 자동 제출하지 않습니다.
제출할 때는 **같은 run 폴더의 `model.py`와 `best_model.pt`**를 함께 사용합니다.

```python
import zipfile
from google.colab import files
with zipfile.ZipFile('/content/submission.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ['model.py', 'best_model.pt']:
        archive.write('runs/generalize_cv/' + name, arcname=name)
files.download('/content/submission.zip')
```

## 재개와 저장 결과

```python
!python train.py --cv --run_dir runs/generalize_cv --resume
```

완료한 fold는 재학습하지 않고 재사용합니다. 소스·설정·fold 분할·train CSV가 바뀌면 재개를 거부합니다.
같은 런타임에서 같은 설정으로 재개하세요. `/content`는 런타임이 삭제되면 사라지므로 체크포인트와 제출 파일을
필요에 맞게 다운로드해야 합니다. `last_checkpoint.pt`는 재개용이며 제출 파일이 아닙니다.

- `folds.json`: train 내부 이미지 ID·라벨·pixel hash·fold 지정
- `cv_results.json`, `cv_selection.json`: 내부 검증 결과와 공식 valid 전에 고정한 선택
- `<recipe>/fold0` 등: 각 fold의 소스·설정·history·재개용 체크포인트
- `final_fit`: 전체 train으로 다시 학습한 기록과 가중치
- 루트 `model.py`, `best_model.pt`, `evaluation.json`: 고정한 최종 모델과 공식 valid 보고
- `frozen_selection.json`: 평가 전 weights SHA-256, 고정 기간/종류, valid의 선택 미사용 기록

`--init_weights`는 일반 추가 학습에만 사용할 수 있습니다. **CV/fold/fixed refit에서는 금지**합니다.
전체 train으로 학습한 가중치는 내부 held-out 이미지도 이미 보았기 때문입니다.
과거 `--search`는 동일 valid에서 여러 후보를 비교하는 방식으로 남아 있지만 새로운 기본 권장은 `--cv`입니다.
`!python train.py`만 실행하면 종전 단일 학습이므로 새 절차에는 반드시 **`--cv`**를 붙이세요.

## 모델·라이브러리 제한과 검증

[model.py](project1/model.py)는 7개 Residual 블록, BatchNorm, SE attention, 평균/최대 pooling,
두 LayerNorm/Dropout 헤드와 16×16 중간 특징 경로를 포함합니다. CV의 이전 deep 설정은 새 경로를 0으로 고정해
같은 추론 함수를 사용합니다. 최종 제출은 한 모델이며 파라미터를 중복해서 숨기거나 여러 모델을 동시에 실행하지 않습니다.

파라미터 **1,874,597개** + 실수형 버퍼 **3,840개** = **1,878,437개**, FP32 state_dict **약 7.22 MiB**로 제한 이내입니다.
제출 모델은 torch와 표준 라이브러리만 사용합니다. 학습 데이터 로딩의 Pillow는 torchvision의 의존성입니다.
데이터/가중치는 GitHub에 올리지 않습니다. 공식 [check_submission.py](project1/check_submission.py)는 변경하지 않았습니다.

```bash
python -m unittest discover -s tests -v
PLANT_DATA_DIR=/path/to/student_data_128 python -m unittest discover -s tests -v
python scripts/build_colab.py
```

테스트는 제출 규격·평가 전처리·과거 가중치 호환·MixUp/BN·실제 이미지 중단/재개 외에 다음도 검사합니다.
동일 이미지의 fold 분리 금지, 공식 valid 파일 없는 내부 학습, 공식 valid 라벨 변경 시 고정 가중치의 완전 동일성,
CV 선택/최종 단일 모델 내보내기, read-only 비교와 paired bootstrap입니다. 작은 CPU 학습 점수는 전체 성능이 아닙니다.
CUDA AMP와 전체 성능은 Colab에서 확인해야 합니다.

[자동 준비 Colab](https://colab.research.google.com/github/yeonji200522-oss/DeepLearning2026-2/blob/main/colab_train.ipynb)는
새 CV 절차를 기본값으로 사용합니다. GPU 런타임에서 셀을 순서대로 실행하고 제출 전에 보고서를 검토하세요.
Drive를 사용하지 않습니다. 데이터·과제 규격은 [Class2026_DL/project1](https://github.com/maygodwithu/Class2026_DL/tree/main/project1)에서 왔으며,
자동 준비는 공식 커밋 `c0c7737fa2355cf57bf78792b376bf770aa950dd`를 고정합니다. 기존 `model_py.ipynb`는 보존했습니다.
