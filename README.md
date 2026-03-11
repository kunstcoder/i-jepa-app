# I-JEPA Downstream Tasks

I-JEPA(Image-based Joint-Embedding Predictive Architecture) 사전학습 모델의 encoder를 활용한 downstream task(분류, 세그멘테이션) 학습 프레임워크입니다.

사전학습된 ViT encoder의 가중치를 freeze한 채로, 경량 head/decoder만 학습하여 빠르게 downstream 성능을 평가할 수 있습니다.

---

## 목차

- [지원 모델](#지원-모델)
- [프로젝트 구조](#프로젝트-구조)
- [설치](#설치)
- [사전학습 체크포인트 준비](#사전학습-체크포인트-준비)
- [사용법](#사용법)
  - [Image Classification](#1-image-classification)
  - [Semantic Segmentation](#2-semantic-segmentation)
  - [Evaluation](#3-evaluation)
- [Config 설정 가이드](#config-설정-가이드)
- [아키텍처 상세](#아키텍처-상세)
  - [IJEPABackbone](#ijepabackbone)
  - [IJEPAClassifier](#ijepaclassifier)
  - [IJEPASegmentor](#ijepasegmentor)
- [API 레퍼런스](#api-레퍼런스)

---

## 지원 모델

| 모델 이름 | embed_dim | depth | num_heads | 파라미터 수 (약) |
|-----------|-----------|-------|-----------|-----------------|
| `vit_tiny` | 192 | 12 | 3 | 5.7M |
| `vit_small` | 384 | 12 | 6 | 22M |
| `vit_base` | 768 | 12 | 12 | 86M |
| `vit_large` | 1024 | 24 | 16 | 304M |
| `vit_huge` | 1280 | 32 | 16 | 632M |
| `vit_giant` | 1408 | 40 | 16 | 1.0B |

config의 `backbone.model_name` 항목에서 위 모델 이름을 지정하여 사용합니다.

---

## 프로젝트 구조

```
i-jepa-app/
├── configs/
│   ├── base.yaml                    # 공통 설정 (backbone + training)
│   ├── classification.yaml          # 분류 태스크 설정
│   └── segmentation.yaml           # 세그멘테이션 태스크 설정
├── src/
│   ├── models/
│   │   ├── vision_transformer.py    # I-JEPA ViT 정의 (원본 기반, 자체 포함)
│   │   ├── ijepa_backbone.py        # 사전학습 encoder wrapper
│   │   ├── classifier.py            # 분류 헤드 (linear / attentive)
│   │   └── segmentor.py             # 세그멘테이션 디코더 (linear / progressive)
│   ├── datasets/
│   │   ├── classification_dataset.py  # ImageFolder 기반 분류 데이터셋
│   │   └── segmentation_dataset.py    # 이미지+마스크 세그멘테이션 데이터셋
│   ├── utils/
│   │   └── checkpoint.py            # I-JEPA 체크포인트 로딩
│   └── train/
│       ├── train_classifier.py      # 분류 학습/평가 루프
│       └── train_segmentor.py       # 세그멘테이션 학습/평가 루프
├── train_classification.py          # 분류 학습 진입점
├── train_segmentation.py            # 세그멘테이션 학습 진입점
├── evaluate.py                      # 통합 평가 스크립트
└── requirements.txt
```

---

## 설치

### 요구사항

- Python >= 3.10
- CUDA 지원 GPU (권장)

### 설치 단계

```bash
# 1. 저장소 클론
git clone <repo-url> i-jepa-app
cd i-jepa-app

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 의존성 설치
pip install -r requirements.txt
```

`requirements.txt` 내용:
```
torch>=2.0
torchvision>=0.15
pyyaml>=6.0
tensorboard>=2.14
```

> PyTorch는 CUDA 버전에 맞춰 설치해야 합니다. 공식 사이트에서 확인:
> https://pytorch.org/get-started/locally/

---

## 사전학습 체크포인트 준비

I-JEPA 공식 체크포인트를 다운로드합니다. Meta 공식 레포에서 제공하는 체크포인트 파일(`.pth`)을 사용합니다.

| 체크포인트 | model_name | patch_size | 비고 |
|-----------|------------|------------|------|
| ViT-H/14 (IN1K, 300ep) | `vit_huge` | 14 | 가장 널리 사용 |
| ViT-H/16 (IN1K, 300ep) | `vit_huge` | 16 | |
| ViT-g/16 (IN1K, 300ep) | `vit_giant` | 16 | |

> **주의**: `model_name`과 `patch_size`는 체크포인트와 반드시 일치해야 합니다.
> 예를 들어 ViT-H/14 체크포인트를 사용하면 `model_name: vit_huge`, `patch_size: 14`로 설정합니다.

체크포인트 파일 안에는 `encoder`, `predictor`, `target_encoder` 등의 키가 있으며, 이 프레임워크는 **`encoder` 키만 자동 추출**하여 로딩합니다.

---

## 사용법

### 1. Image Classification

#### 데이터 준비

ImageFolder 형식으로 정리합니다:

```
/path/to/dataset/
├── train/
│   ├── class_a/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── class_b/
│       └── ...
└── val/
    ├── class_a/
    └── class_b/
```

#### config 수정

`configs/classification.yaml`을 편집합니다:

```yaml
backbone:
  model_name: vit_base        # vit_tiny ~ vit_giant 중 선택
  checkpoint_path: /path/to/ijepa_checkpoint.pth
  img_size: 224
  patch_size: 16              # 체크포인트에 맞춰 설정
  freeze: true

task:
  type: classification
  num_classes: 1000           # 데이터셋 클래스 수
  head_mode: linear           # linear 또는 attentive

data:
  train_dir: /path/to/train
  val_dir: /path/to/val
```

#### 학습 실행

```bash
# config 파일 기반 실행
python train_classification.py --config configs/classification.yaml

# CLI에서 파라미터 오버라이드
python train_classification.py \
  --config configs/classification.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --model-name vit_base \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.0005 \
  --device cuda
```

#### CLI 오버라이드 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--config` | YAML config 경로 | `configs/classification.yaml` |
| `--checkpoint` | I-JEPA 체크포인트 경로 | config 값 사용 |
| `--model-name` | ViT 모델 이름 | config 값 사용 |
| `--epochs` | 학습 에폭 수 | config 값 사용 |
| `--batch-size` | 배치 크기 | config 값 사용 |
| `--lr` | 학습률 | config 값 사용 |
| `--device` | 디바이스 (`cuda` / `cpu`) | config 값 사용 |

---

### 2. Semantic Segmentation

#### 데이터 준비

이미지와 마스크 파일을 쌍으로 준비합니다. 마스크는 단일 채널 PNG이며 픽셀 값이 클래스 인덱스(0~N-1)입니다. 무시할 영역은 255로 설정합니다.

```
/path/to/dataset/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── masks/
│       ├── img_001.png    # 각 픽셀 = 클래스 인덱스
│       └── img_002.png
└── val/
    ├── images/
    └── masks/
```

> **이미지와 마스크의 파일명이 일치해야 합니다** (확장자 제외).
> 예: `img_001.jpg` ↔ `img_001.png`

#### config 수정

`configs/segmentation.yaml`을 편집합니다:

```yaml
backbone:
  model_name: vit_huge
  checkpoint_path: /path/to/ijepa_checkpoint.pth
  img_size: 224
  patch_size: 14
  freeze: true

task:
  type: segmentation
  num_classes: 21             # VOC=21, ADE20K=150, Cityscapes=19
  decoder_type: progressive   # linear 또는 progressive
  ignore_index: 255           # 무시할 마스크 값

data:
  train_images_dir: /path/to/train/images
  train_masks_dir: /path/to/train/masks
  val_images_dir: /path/to/val/images
  val_masks_dir: /path/to/val/masks
```

#### 학습 실행

```bash
python train_segmentation.py --config configs/segmentation.yaml

# CLI 오버라이드 예시
python train_segmentation.py \
  --config configs/segmentation.yaml \
  --model-name vit_large \
  --batch-size 8 \
  --epochs 120
```

---

### 3. Evaluation

학습 완료 후 저장된 모델을 평가합니다:

```bash
# 분류 모델 평가
python evaluate.py \
  --config configs/classification.yaml \
  --weights checkpoints/classification/best.pth \
  --device cuda

# 세그멘테이션 모델 평가
python evaluate.py \
  --config configs/segmentation.yaml \
  --weights checkpoints/segmentation/best.pth \
  --device cuda
```

#### 출력 메트릭

| 태스크 | 메트릭 |
|--------|--------|
| Classification | Top-1 Accuracy, Top-5 Accuracy, Loss |
| Segmentation | mIoU, Pixel Accuracy, Loss |

---

## Config 설정 가이드

### backbone 섹션

```yaml
backbone:
  model_name: vit_huge      # 사용할 ViT 모델
  checkpoint_path: null     # null이면 랜덤 초기화
  img_size: 224             # 입력 이미지 크기
  patch_size: 14            # 패치 크기 (체크포인트에 맞춰야 함)
  freeze: true              # true: encoder 가중치 고정
```

### training 섹션

```yaml
training:
  epochs: 50
  batch_size: 64
  lr: 0.001
  weight_decay: 0.05
  scheduler: cosine          # cosine 또는 step
  num_workers: 8             # DataLoader 워커 수
  device: cuda
  mixed_precision: true      # AMP 사용 여부
  log_interval: 50           # N 배치마다 로그 출력
  save_dir: ./checkpoints    # 체크포인트 저장 경로
```

### 모델별 권장 설정

| 모델 | patch_size | batch_size | lr | 용도 |
|------|-----------|------------|-----|------|
| `vit_base` | 16 | 128 | 0.001 | 빠른 실험, 제한된 GPU |
| `vit_large` | 16 | 64 | 0.0005 | 중간 규모 실험 |
| `vit_huge` | 14 | 32~64 | 0.001 | I-JEPA 기본 모델 |
| `vit_giant` | 16 | 16~32 | 0.0005 | 최대 성능 |

---

## 아키텍처 상세

### IJEPABackbone

I-JEPA 사전학습 encoder를 감싸는 wrapper입니다.

```
Input Image [B, 3, 224, 224]
    ↓
PatchEmbed (patch_size=14)
    ↓
[B, 256, 1280]  (256 = 16×16 patches, 1280 = embed_dim for vit_huge)
    ↓
Transformer Blocks (×32 for vit_huge)
    ↓
LayerNorm
    ↓
Output: [B, 256, 1280]   ← forward()
    or: [B, 1280, 16, 16] ← get_spatial_features()
```

- **freeze 모드** (기본): encoder 파라미터 고정, `eval()` 모드 유지
- **unfreeze 모드**: `backbone.unfreeze()` 호출로 fine-tuning 가능

### IJEPAClassifier

```
IJEPABackbone → [B, N, D]
    ↓
[linear 모드]         [attentive 모드]
Global Avg Pool       Attention Pooling (learned query)
    ↓                     ↓
[B, D]                [B, D]
    ↓                     ↓
Linear(D, C)          MLP(D → D → C)
    ↓                     ↓
[B, num_classes]      [B, num_classes]
```

- **linear**: 빠른 linear probing 평가에 적합
- **attentive**: 학습 가능한 query로 attention pooling 후 MLP head. 더 높은 성능

### IJEPASegmentor

```
IJEPABackbone → [B, D, H, W]   (spatial features)
    ↓
[linear 디코더]                      [progressive 디코더]
1×1 Conv(D → C)                     Conv+BN+ReLU → Upsample ×3
    ↓                                    ↓
Bilinear Upsample                    Conv 1×1
    ↓                                    ↓
[B, C, 224, 224]                     [B, C, 224, 224]
```

Progressive 디코더 상세 (vit_huge, patch_size=14 기준):

```
[B, 1280, 16, 16]
  → Conv3×3 + BN + ReLU → [B, 512, 16, 16]
  → Upsample 2×          → [B, 512, 32, 32]
  → Conv3×3 + BN + ReLU → [B, 256, 32, 32]
  → Upsample 2×          → [B, 256, 64, 64]
  → Conv3×3 + BN + ReLU → [B, 128, 64, 64]
  → Upsample 2×          → [B, 128, 128, 128]
  → Conv3×3 + BN + ReLU → [B, 64, 128, 128]
  → Conv1×1              → [B, num_classes, 128, 128]
  → Bilinear Upsample    → [B, num_classes, 224, 224]
```

---

## API 레퍼런스

### 코드에서 직접 사용하기

```python
import torch
from src.models.ijepa_backbone import IJEPABackbone
from src.models.classifier import IJEPAClassifier
from src.models.segmentor import IJEPASegmentor

# 1. Backbone 생성
backbone = IJEPABackbone(
    model_name="vit_base",                        # 모델 선택
    checkpoint_path="path/to/checkpoint.pth",      # 체크포인트 (None이면 랜덤 초기화)
    img_size=224,
    patch_size=16,
    freeze=True,
)

# 2. Feature 추출
x = torch.randn(4, 3, 224, 224)
features = backbone(x)                   # [4, 196, 768]
spatial = backbone.get_spatial_features(x)  # [4, 768, 14, 14]

# 3. Classification
classifier = IJEPAClassifier(
    backbone=backbone,
    num_classes=1000,
    head_mode="linear",   # 또는 "attentive"
)
logits = classifier(x)   # [4, 1000]

# 4. Segmentation
segmentor = IJEPASegmentor(
    backbone=backbone,
    num_classes=21,
    decoder_type="progressive",   # 또는 "linear"
)
masks = segmentor(x)   # [4, 21, 224, 224]

# 5. Fine-tuning이 필요한 경우
backbone.unfreeze()
# 이후 optimizer에 backbone 파라미터도 포함
```

### 사용 가능한 모델 목록 확인

```python
from src.models.vision_transformer import VIT_REGISTRY
print(list(VIT_REGISTRY.keys()))
# ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge', 'vit_giant']
```
