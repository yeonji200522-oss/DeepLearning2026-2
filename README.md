# DeepLearning2026-2

식물 잎 이미지의 작물 종류(3종)와 건강 상태(2종)를 분류하는 CNN 프로젝트입니다.
사전학습 가중치 없이 학습하며, 두 과제 Macro-F1의 평균으로 최적 모델을 선택합니다.

## Colab에서 시작

1. **[Colab에서 바로 열기](https://colab.research.google.com/github/yeonji200522-oss/DeepLearning2026-2/blob/main/colab_train.ipynb)**를 누릅니다. 파일 다운로드나 업로드는 필요 없습니다.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
3. 셀을 위에서부터 실행합니다. Google Drive 연결은 필요 없습니다.

노트북 안에 개선 모델·학습 코드가 포함되어 있어 `train.py` 등의 개별 업로드나 개인 GitHub push가
필요하지 않습니다. 공식 코드와 학습/검증 데이터를 자동으로 내려받습니다.
기본 실행은 공식 baseline 10 epoch와 개선 모델 40 epoch를 학습하고, 공식 CPU 검사기로 두 결과를
비교한 뒤 검증 점수가 높은 모델의 제출 파일을 묶습니다. 실행 시간은 첫 epoch 로그로 확인하세요.

첫 실험은 `RUN_NAME='improved_seed42'`로 시작합니다. 새 실험은 RUN_NAME을 바꾸고,
중단 후 재개는 같은 RUN_NAME·학습 설정·총 epoch 수를 사용합니다.
완료한 실험은 덮어쓰지 않습니다. GPU 메모리가 부족하면 새 실험 이름에서 batch size를 32로 줄이세요.

## 구현

- [모델](project1/model.py): Residual CNN, BatchNorm, SE channel attention, 평균/최대 pooling,
  두 분류 헤드의 LayerNorm과 Dropout(0.2).
- [데이터](project1/dataset.py): 완만한 crop·회전·색상 변화·뒤집기. 검증은 공식 Normalize(0.5, 0.5) 유지.
- [학습](project1/train.py): AdamW(행렬 가중치에 weight decay 적용), warmup/cosine, label smoothing,
  train 라벨 기반 클래스 가중치, CUDA AMP, gradient clipping, raw/EMA 검증 비교.
- [공식 검사기](project1/check_submission.py): 과제 저장소의 코드를 변경 없이 사용.

현재 기본 모델은 파라미터 1,301,837개 + 실수형 버퍼 3,072개 = **1,304,909개**,
FP32 state_dict는 **약 5.01 MiB**입니다. 생성과 추론에 `torch` 외 추가 패키지는 필요하지 않습니다.
학습 이미지 로딩에는 torchvision의 의존성인 Pillow를 사용합니다.
기법의 조합이 성능 향상을 보장하지 않으며, 전체 학습 후 baseline 대비 개선 여부를 확인해야 합니다.

## 저장 결과

Colab의 `/content/leaf-cnn/runs/<RUN_NAME>/`에 다음이 남습니다.
같은 런타임에서는 학습을 재개할 수 있습니다. 런타임이 삭제되면 파일도 사라지므로
종료 전에 제출 ZIP과 필요한 실험 기록을 다운로드하세요.

- `best_model.pt`, `model.py`: 서로 대응하는 제출용 두 파일
- `last_checkpoint.pt`: 마지막 완료 epoch부터 학습 재개하기 위한 파일. 제출하지 않습니다.
- `dataset.py`, `train.py`, `config.json`: 당시 코드·설정·소스 해시
- `best_metrics.json`, `history.csv`, `history.json`, `evaluation.json`, `console.log`: 학습 및 클래스별 검증 결과
- `submission_check.log`, `comparison.json`, `submission.zip`: 공식 검사 결과, baseline 비교, 제출용 묶음

`submission.zip`에는 검사에 통과한 후보 중 검증 점수가 높은 모델의 `model.py`와 `best_model.pt`가
들어 있습니다. zip을 풀어 두 파일을 리더보드에 직접 제출하세요. 이 프로젝트는 자동 제출하지 않습니다.
baseline과 개선 모델의 기본 학습 기간이 다르므로 기본 비교는 전체 학습 방법의 비교입니다.
구조만의 효과를 판단하려면 동일한 학습 조건으로 추가 비교하세요.

## 로컬 실행

Python과 호환되는 torch/torchvision을 설치하고 공식 저장소의 `student_data_128`을 준비합니다.
Colab에는 두 라이브러리가 기본 설치되어 있으므로 먼저 버전을 확인하고 임의로 재설치하지 마세요.

```bash
cd project1
python train.py --data_dir /path/to/student_data_128 --run_dir runs/improved_seed42
python train.py --data_dir /path/to/student_data_128 --run_dir runs/improved_seed42 --resume
python train.py --data_dir /path/to/student_data_128 --run_dir runs/improved_seed42 --eval_only
python check_submission.py --model runs/improved_seed42/model.py \
  --weights runs/improved_seed42/best_model.pt --data_dir /path/to/student_data_128
```

재개 시 기존 실험과 동일한 학습 옵션을 전달해야 합니다. 코드가 달라지면 재개를 거부하므로
저장한 run 폴더의 코드를 쓰거나 새 실험을 시작하세요. CUDA AMP는 Colab GPU에서
확인해야 합니다. CPU smoke 학습의 점수를 전체 검증 성능으로 해석하지 마세요.

증강 효과는 `--no_augment`, 불균형 대응은 `--class_weight_power 0` 또는 `0.5`,
EMA는 `--ema_decay 0`으로 별도 실험할 수 있습니다. batch size 변경 시에도 새 run을 사용하세요.
최종 후보는 seed 42/43/44로 재학습해 성능 변동을 확인하는 것을 권장합니다.

## 검증과 노트북 갱신

```bash
python -m unittest discover -s tests -v
PLANT_DATA_DIR=/path/to/student_data_128 python -m unittest discover -s tests -v
python scripts/build_colab.py
```

실제 데이터 경로를 지정하면 epoch checkpoint에서 끊긴 학습의 재개 결과와 연속 학습 결과가
동일한지도 검사합니다. 모델·학습 파일을 편집한 뒤에는 `build_colab.py`로 노트북에 소스를 다시 넣으세요.
코드·데이터의 출처는 [Class2026_DL/project1](https://github.com/maygodwithu/Class2026_DL/tree/main/project1)이며,
노트북은 확인한 공식 커밋 `c0c7737fa2355cf57bf78792b376bf770aa950dd`를 고정해서 사용합니다.
기존 `model_py.ipynb`는 보존했습니다.
