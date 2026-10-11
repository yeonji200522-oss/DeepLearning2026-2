"""Build a self-contained Colab notebook using only Python's standard library."""
import json
from pathlib import Path
import pprint
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip()})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(text).strip(),
                  "execution_count": None, "outputs": []})


markdown("""
# 잎 사진 CNN: baseline 비교와 개선 모델 학습

**런타임 → 런타임 유형 변경 → GPU**를 선택한 뒤 위에서부터 실행하세요.
이 노트북 안에 model.py / train.py / dataset.py가 포함되어 있으므로 파일을 따로 업로드하거나
개인 GitHub에 push할 필요가 없습니다. 공식 저장소에서 데이터와 baseline을 자동으로 받습니다.

Residual CNN + BatchNorm + SE attention + 두 출력 헤드의 Dropout,
완만한 데이터 증강, AdamW weight decay, warmup/cosine 스케줄,
train 라벨에만 근거한 클래스 가중치와 raw/EMA 검증 비교를 사용합니다.
사전학습 가중치는 사용하지 않습니다. 높은 점수는 전체 학습 후 검증으로 확인해야 합니다.

학습은 Colab GPU에서 실행하며 가중치·코드·로그는 `/content/leaf-cnn/runs`에 저장합니다.
Drive 연결은 필요 없습니다. 같은 런타임에서는 같은 설정과 RUN_NAME으로 재개할 수 있습니다.
런타임이 삭제되면 파일도 사라지므로 종료 전에 제출 ZIP과 필요한 실험 기록을 다운로드하세요.
GPU는 T4로 시작할 수 있으며 전체 실행 시간은 첫 epoch 측정값을 참고하세요.
""")
code("""
import torch
import torchvision
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
assert torch.cuda.is_available(), 'Colab 런타임을 GPU로 변경하세요.'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
""")
markdown("""
## 실행 설정
처음에는 아래 기본값으로 시작하세요. 비교 실험은 RUN_NAME을 새 이름으로 바꾸고 진행합니다.
재개할 때는 학습 설정과 총 EPOCHS를 그대로 유지해야 합니다.
RUN_BASELINE=False는 공식 baseline 비교를 생략합니다. 검증 성능을 비교하려면 True를 사용하세요.
""")
code("""
from pathlib import Path
RUN_NAME = 'improved_seed42'
SEED = 42
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 3e-4
CLASS_WEIGHT_POWER = 0.35  # 0이면 클래스 가중치 해제
CROP_CLASS_WEIGHT_POWER = None  # 작물만 조정하려면 예: 0.5
CROP_LOSS_WEIGHT = 1.0          # 작물 손실 비중을 높이려면 예: 1.25
AUGMENTATION = 'standard'       # 'gentle'은 잎 형태를 더 보존하는 약한 증강
INIT_WEIGHTS = None            # 본인이 제공 데이터로 학습한 best_model.pt 경로만 사용
RUN_BASELINE = True
BASELINE_EPOCHS = 10

RUN_ROOT = Path('/content/leaf-cnn/runs')
RUN_DIR = RUN_ROOT / RUN_NAME
WORK = Path('/content/leaf-cnn/project1')
WORK.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
""")
markdown("""
## 공식 코드·데이터 자동 준비
재현성을 위해 확인한 공식 저장소 커밋을 고정합니다. 이미 있는 checkout을 자동으로 갱신하지 않습니다.
""")
code("""
import subprocess
import sys

OFFICIAL_COMMIT = 'c0c7737fa2355cf57bf78792b376bf770aa950dd'
OFFICIAL = Path('/content/Class2026_DL-' + OFFICIAL_COMMIT[:8])
if not OFFICIAL.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--single-branch', '--branch', 'main',
                    'https://github.com/maygodwithu/Class2026_DL.git', str(OFFICIAL)], check=True)
    head = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    if head != OFFICIAL_COMMIT:
        subprocess.run(['git', '-C', str(OFFICIAL), 'fetch', '--depth', '1', 'origin', OFFICIAL_COMMIT], check=True)
        subprocess.run(['git', '-C', str(OFFICIAL), 'checkout', '--detach', OFFICIAL_COMMIT], check=True)
head = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
assert head == OFFICIAL_COMMIT, '공식 저장소 커밋이 다릅니다. 기존 checkout을 확인하세요.'
assert not subprocess.check_output(['git', '-C', str(OFFICIAL), 'status', '--porcelain'], text=True).strip(), '공식 코드에 변경 사항이 있습니다.'
OFFICIAL_PROJECT = OFFICIAL / 'project1'
DATA_DIR = OFFICIAL_PROJECT / 'student_data_128'
for name in ['train.csv', 'valid.csv', 'class_mapping.json']:
    assert (DATA_DIR / name).is_file(), name
print('데이터 경로:', DATA_DIR)
""")
markdown("""
## 개선 학습 파일 자동 생성
아래 셀은 노트북에 포함된 소스를 Colab 작업 폴더에 씁니다. 수동 파일 업로드는 필요 없습니다.
이미 수정한 파일이 있으면 덮어쓰지 않고 중단합니다. 파일을 편집한 뒤에는 기존 작업 폴더에서 실행하세요.
""")
sources = {name: (ROOT / 'project1' / name).read_text(encoding='utf-8')
           for name in ('model.py', 'dataset.py', 'train.py', 'diagnose.py', 'check_submission.py')}
code('SOURCES = ' + pprint.pformat(sources, width=100, sort_dicts=False) + '\n\n' + textwrap.dedent("""
for name, source in SOURCES.items():
    path = WORK / name
    if path.exists() and path.read_text(encoding='utf-8') != source:
        raise RuntimeError(f'{path}에 다른 코드가 있습니다. 기존 파일을 확인하고 작업 폴더를 선택하세요.')
    path.write_text(source, encoding='utf-8')
print('준비 완료:', ', '.join(SOURCES))
""").strip())
markdown("""
## 데이터·평가 전처리·제출 규격 확인
모든 이미지의 경로와 RGB 크기를 확인합니다. 학습 라벨만으로 클래스 분포를 확인합니다.
개선 모델의 검증 전처리가 공식 검사기와 같은 텐서를 만드는지도 검사합니다.
""")
code("""
import csv
import importlib.util
from collections import Counter
from PIL import Image

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

data_module = load_module('leaf_dataset', WORK / 'dataset.py')
model_module = load_module('leaf_model', WORK / 'model.py')
checker_module = load_module('leaf_checker', WORK / 'check_submission.py')
assert not checker_module.check_imports(str(WORK / 'model.py'))
for split, expected in [('train', 8000), ('valid', 2284)]:
    dataset = data_module.PlantDataset(DATA_DIR, split, data_module.get_transforms(False))
    assert len(dataset) == expected, (split, len(dataset))
    for row in dataset.rows:
        with Image.open(dataset.img_dir / row['id']) as image:
            assert image.size == (128, 128) and image.mode == 'RGB', row['id']
    print(split, len(dataset), '장 확인')
    if split == 'train':
        print('학습 라벨 분포:', Counter((int(r['crop_label']), int(r['health_label'])) for r in dataset.rows))
sample = data_module.PlantDataset(DATA_DIR, 'valid', data_module.get_transforms(False))
with Image.open(sample.img_dir / sample.rows[0]['id']) as image:
    image = image.convert('RGB')
    assert torch.equal(data_module.get_transforms(False)(image), checker_module.EVAL_TRANSFORM(image))
model = model_module.MultiTaskCNN().eval()
params = sum(p.numel() for p in model.parameters())
buffers = sum(b.numel() for b in model.buffers() if b.is_floating_point())
assert params + buffers <= 2_000_000
with torch.no_grad():
    a, b = model(torch.randn(2, 3, 128, 128))
assert a.shape == (2, 3) and b.shape == (2, 2)
assert torch.isfinite(a).all() and torch.isfinite(b).all()
print(f'파라미터 {params:,} + 실수형 버퍼 {buffers:,} = {params + buffers:,}')
del model
""")
markdown("""
## 실제 이미지로 학습 smoke check
소수 데이터로 1 epoch 학습하여 역전파·저장·평가가 연결되는지 확인합니다.
이 점수는 전체 검증 성능이 아니며 제출 파일로 사용하지 않습니다.
""")
code("""
import json
import tempfile

def run_logged(command, cwd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
            status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)

smoke_dir = Path(tempfile.mkdtemp(prefix='leaf-smoke-', dir='/content'))
run_logged([sys.executable, '-u', 'train.py', '--data_dir', str(DATA_DIR),
            '--run_dir', str(smoke_dir), '--epochs', '1', '--batch_size', '8',
            '--train_limit', '32', '--valid_limit', '32', '--device', 'cuda'],
           WORK, smoke_dir / 'console.log')
assert json.loads((smoke_dir / 'best_metrics.json').read_text())['smoke_run']
""")
markdown("""
## 공식 baseline 학습 (기본 설정 10 epoch)
완료한 baseline은 다시 학습하지 않습니다. 중단된 공식 baseline은 재개 기능이 없어
같은 폴더에 처음부터 다시 학습합니다. 개선 모델은 다음 셀에서 epoch 단위 재개를 지원합니다.
""")
code("""
import shutil

BASELINE_DIR = RUN_ROOT / f'baseline_official_{BASELINE_EPOCHS}epochs'
if RUN_BASELINE:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ['model.py', 'dataset.py', 'train.py']:
        source = OFFICIAL_PROJECT / name
        target = BASELINE_DIR / name
        if target.exists():
            assert target.read_bytes() == source.read_bytes(), f'Baseline 코드 변경: {target}'
        else:
            shutil.copy2(source, target)
    marker = BASELINE_DIR / 'completed.json'
    if not marker.exists():
        run_logged([sys.executable, '-u', str(OFFICIAL_PROJECT / 'train.py'),
                    '--data_dir', str(DATA_DIR), '--device', 'cuda',
                    '--epochs', str(BASELINE_EPOCHS), '--batch_size', '64', '--lr', '0.001',
                    '--save_path', str(BASELINE_DIR / 'best_model.pt')],
                   OFFICIAL_PROJECT, BASELINE_DIR / 'console.log')
        marker.write_text(json.dumps({'official_commit': OFFICIAL_COMMIT,
                                      'epochs': BASELINE_EPOCHS}), encoding='utf-8')
    else:
        assert json.loads(marker.read_text()) == {'official_commit': OFFICIAL_COMMIT, 'epochs': BASELINE_EPOCHS}
        assert (BASELINE_DIR / 'best_model.pt').is_file()
        print('완료한 baseline 재사용:', BASELINE_DIR)
""")
markdown("""
## 개선 모델 전체 학습
각 epoch에서 raw와 EMA의 평균 Macro-F1을 비교해 최고 가중치를 저장합니다.
best_model.pt는 제출용 state_dict이고, last_checkpoint.pt는 optimizer 등을 포함한 재개 전용 파일입니다.
마지막 epoch까지 무조건 쓰지 않고 전체 학습 중 가장 좋은 검증 결과를 사용합니다.
""")
code("""
command = [sys.executable, '-u', 'train.py', '--data_dir', str(DATA_DIR),
           '--run_dir', str(RUN_DIR), '--epochs', str(EPOCHS), '--batch_size', str(BATCH_SIZE),
           '--lr', str(LEARNING_RATE), '--weight_decay', str(WEIGHT_DECAY),
           '--class_weight_power', str(CLASS_WEIGHT_POWER), '--crop_loss_weight', str(CROP_LOSS_WEIGHT),
           '--augmentation', AUGMENTATION, '--seed', str(SEED), '--device', 'cuda']
if CROP_CLASS_WEIGHT_POWER is not None:
    command.extend(['--crop_class_weight_power', str(CROP_CLASS_WEIGHT_POWER)])
if INIT_WEIGHTS is not None:
    command.extend(['--init_weights', str(INIT_WEIGHTS)])
if (RUN_DIR / 'last_checkpoint.pt').exists():
    command.append('--resume')
run_logged(command, WORK, RUN_DIR / 'console.log')
metrics = json.loads((RUN_DIR / 'best_metrics.json').read_text())
assert not metrics['smoke_run']
print('최적 모델:', json.dumps(metrics, ensure_ascii=False, indent=2))
history = json.loads((RUN_DIR / 'history.json').read_text())
print('epoch | loss | crop F1 | health F1 | 평균 F1 | 초')
for row in history:
    print(f"{row['epoch']:5d} | {row['train_loss']:.4f} | {row['crop_f1']:.4f} | "
          f"{row['health_f1']:.4f} | {row['score']:.4f} | {row['seconds']:.1f}")
""")
markdown("""
## 공식 CPU 검사와 baseline 비교
전체 valid를 공식 전처리로 검사합니다. 두 학습 기간이 다르므로 이 비교는 전체 학습 방법의 비교입니다.
구조만의 효과를 확인하려면 두 모델을 동일 epoch·seed·학습 설정으로 따로 비교해야 합니다.
검사에 실패하면 제출 묶음을 만들지 않습니다.
""")
code("""
import re

candidates = [('improved', RUN_DIR)]
if RUN_BASELINE:
    candidates.append(('baseline', BASELINE_DIR))
results = []
for name, directory in candidates:
    checked = subprocess.run(
        [sys.executable, str(WORK / 'check_submission.py'), '--model', str(directory / 'model.py'),
         '--weights', str(directory / 'best_model.pt'), '--data_dir', str(DATA_DIR)],
        cwd=str(WORK), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(checked.stdout)
    (directory / 'submission_check.log').write_text(checked.stdout, encoding='utf-8')
    checked.check_returncode()
    match = re.search(r'평균 Macro-F1\\s*:\\s*([0-9.]+)', checked.stdout)
    assert match, '공식 점수를 찾지 못했습니다.'
    results.append({'name': name, 'score': float(match.group(1)), 'run_dir': str(directory)})
print('공식 검증 비교:', results)
selected = max(results, key=lambda row: row['score'])
print('검증 점수 기준 선택:', selected)
if RUN_BASELINE:
    lookup = {row['name']: row['score'] for row in results}
    print('개선 모델 - baseline:', round(lookup['improved'] - lookup['baseline'], 4))
(RUN_DIR / 'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
""")
markdown("""
## 제출용 두 파일 묶기·다운로드
선택한 실험의 model.py와 best_model.pt만 묶습니다. checkpoint나 데이터는 제출하지 않습니다.
리더보드에는 zip을 풀어서 두 파일을 올리세요. 제출은 이 노트북이 자동으로 수행하지 않습니다.
다른 실험은 RUN_NAME을 바꾸고 학습·검사 셀을 실행하세요. 예: 가중치 해제(power=0),
가중치 강화(power=0.5), seed=43/44. 실제 검증 결과를 보고 설정을 선택하세요.
""")
code("""
import hashlib
import zipfile
from google.colab import files

selected_dir = Path(selected['run_dir'])
archive = RUN_DIR / 'submission.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for name in ['model.py', 'best_model.pt']:
        path = selected_dir / name
        if name == 'best_model.pt':
            assert path.stat().st_size <= 10 * 1024 * 1024
        bundle.write(path, arcname=name)
        print(name, path.stat().st_size, 'bytes', 'sha256:', hashlib.sha256(path.read_bytes()).hexdigest())
print('Colab 저장 완료:', archive)
files.download(str(archive))
""")

notebook = {
    'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
    'metadata': {'colab': {'provenance': []}, 'accelerator': 'GPU',
                 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                 'language_info': {'name': 'python'}},
}
for index, cell in enumerate(cells):
    cell['id'] = f'leaf-{index:02d}'
    cell['source'] = cell['source'].splitlines(keepends=True)
target = ROOT / 'colab_train.ipynb'
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(target)
