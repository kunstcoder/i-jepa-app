# I-JEPA Downstream Tasks 구현 계획

## 1. 배경

**I-JEPA**(Image-based Joint-Embedding Predictive Architecture)는 Meta에서 개발한 자기지도 학습 모델이다.
ViT(Vision Transformer) 백본을 사용하며, 사전학습 후 context encoder가 이미지의 풍부한 시맨틱 표현을 생성한다.

### 핵심 아키텍처 정보

| 항목 | 값 |
|------|-----|
| 백본 | Vision Transformer (ViT) |
| 출력 형태 | `[batch_size, num_patches, embed_dim]` |
| 사용 가능한 모델 | ViT-H/14 (embed_dim=1280), ViT-H/16, ViT-g/16 (embed_dim=1408) |
| 체크포인트 키 | `encoder`, `predictor`, `target_encoder`, `opt`, `scaler`, `epoch` |
| downstream에 필요한 것 | **`encoder` 가중치만 사용** |

---

## 2. 프로젝트 구조

```
i-jepa-app/
├── configs/
│   ├── base.yaml                # 공통 설정
│   ├── classification.yaml      # 분류 태스크 설정
│   └── segmentation.yaml        # 세그멘테이션 태스크 설정
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vision_transformer.py    # I-JEPA ViT 정의 (원본 코드 기반)
│   │   ├── ijepa_backbone.py        # 사전학습 encoder 로딩 + feature 추출
│   │   ├── classifier.py            # 분류 헤드
│   │   └── segmentor.py             # 세그멘테이션 디코더
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── classification_dataset.py
│   │   └── segmentation_dataset.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── checkpoint.py            # 체크포인트 로딩 유틸리티
│   └── train/
│       ├── __init__.py
│       ├── train_classifier.py      # 분류 학습 루프
│       └── train_segmentor.py       # 세그멘테이션 학습 루프
├── train_classification.py          # 분류 학습 진입점
├── train_segmentation.py            # 세그멘테이션 학습 진입점
├── evaluate.py                      # 평가 스크립트
├── requirements.txt
└── PLAN.md
```

---

## 3. 구현 단계

### 3.1 단계 1: ViT 백본 및 체크포인트 로딩

**파일**: `src/models/vision_transformer.py`, `src/utils/checkpoint.py`, `src/models/ijepa_backbone.py`

I-JEPA 원본 레포의 `VisionTransformer` 클래스를 가져와 사용한다.
체크포인트에서 `encoder` 키만 추출하여 ViT에 로딩한다.

```python
# checkpoint.py 핵심 로직
def load_ijepa_encoder(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder_state = checkpoint['encoder']
    model.load_state_dict(encoder_state)
    model.eval()
    return model
```

```python
# ijepa_backbone.py 핵심 로직
class IJEPABackbone(nn.Module):
    def __init__(self, model_name, checkpoint_path, img_size=224, patch_size=14):
        super().__init__()
        self.encoder = VisionTransformer(img_size=img_size, patch_size=patch_size, ...)
        load_ijepa_encoder(checkpoint_path, self.encoder)
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size  # e.g. 224/14 = 16

    def forward(self, x):
        # returns [B, N, D] where N = num_patches
        return self.encoder(x)

    def get_spatial_features(self, x):
        # [B, N, D] -> [B, D, H, W] for segmentation
        features = self.forward(x)
        B, N, D = features.shape
        H = W = self.num_patches_per_side
        return features.transpose(1, 2).reshape(B, D, H, W)
```

**backbone은 freeze 상태로 사용한다.** 필요시 fine-tuning 옵션도 제공한다.

---

### 3.2 단계 2: Classification 모델

**파일**: `src/models/classifier.py`

ViT 출력 `[B, N, D]`에서 패치 토큰을 평균 풀링하여 `[B, D]` 벡터를 만든 뒤, linear head로 분류한다.

```
Input Image → IJEPABackbone → [B, N, D] → Global Average Pooling → [B, D] → Linear Head → [B, num_classes]
```

**두 가지 모드 지원:**
- **Linear Probing**: backbone freeze + linear layer 1개 (빠른 평가용)
- **Fine-tuning (Attentive Probing)**: backbone freeze + 작은 attention pooler + MLP head (성능 극대화)

```python
class IJEPAClassifier(nn.Module):
    def __init__(self, backbone, num_classes, mode='linear'):
        super().__init__()
        self.backbone = backbone
        if mode == 'linear':
            self.head = nn.Linear(backbone.embed_dim, num_classes)
        elif mode == 'attentive':
            self.head = AttentiveProbe(backbone.embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():  # backbone frozen
            features = self.backbone(x)  # [B, N, D]
        pooled = features.mean(dim=1)    # [B, D]
        return self.head(pooled)
```

---

### 3.3 단계 3: Segmentation 모델

**파일**: `src/models/segmentor.py`

ViT 패치 토큰을 2D 공간으로 재배치한 뒤, 경량 디코더로 픽셀 단위 예측을 수행한다.

```
Input Image → IJEPABackbone → [B, N, D] → Reshape → [B, D, H, W] → Decoder → [B, num_classes, H_orig, W_orig]
```

**디코더 옵션:**
- **Linear Decoder** (단순): 1x1 conv + bilinear upsample
- **Progressive Upsampling Decoder** (권장): conv + upsample 반복으로 점진적 해상도 복원

```python
class IJEPASegmentor(nn.Module):
    def __init__(self, backbone, num_classes, decoder_type='progressive'):
        super().__init__()
        self.backbone = backbone
        if decoder_type == 'linear':
            self.decoder = LinearDecoder(backbone.embed_dim, num_classes)
        elif decoder_type == 'progressive':
            self.decoder = ProgressiveDecoder(backbone.embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            spatial = self.backbone.get_spatial_features(x)  # [B, D, H, W]
        return self.decoder(spatial)  # [B, num_classes, H_orig, W_orig]
```

**Progressive Decoder 구조 (patch_size=14, img_size=224 기준):**
```
[B, 1280, 16, 16]  →  Conv+BN+ReLU  →  [B, 512, 16, 16]
                    →  Upsample 2x   →  [B, 512, 32, 32]
                    →  Conv+BN+ReLU  →  [B, 256, 32, 32]
                    →  Upsample 2x   →  [B, 256, 64, 64]
                    →  Conv+BN+ReLU  →  [B, 128, 64, 64]
                    →  Upsample 2x   →  [B, 128, 128, 128]
                    →  Conv+BN+ReLU  →  [B, 64, 128, 128]
                    →  Upsample      →  [B, 64, 224, 224]
                    →  1x1 Conv      →  [B, num_classes, 224, 224]
```

---

### 3.4 단계 4: 데이터셋

**파일**: `src/datasets/classification_dataset.py`, `src/datasets/segmentation_dataset.py`

- **Classification**: ImageFolder 형식 지원 (root/class_name/image.jpg)
- **Segmentation**: 이미지 + 마스크 쌍 지원 (VOC/ADE20K 등의 형식)
- 공통 transform: Resize → CenterCrop → ToTensor → Normalize (ImageNet 통계 사용)

---

### 3.5 단계 5: 학습 스크립트

**파일**: `src/train/train_classifier.py`, `src/train/train_segmentor.py`

| 항목 | Classification | Segmentation |
|------|---------------|-------------|
| Loss | CrossEntropyLoss | CrossEntropyLoss (pixel-wise) |
| Optimizer | AdamW | AdamW |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Metrics | Accuracy, Top-5 Accuracy | mIoU, Pixel Accuracy |
| Backbone | Frozen (default) | Frozen (default) |

---

### 3.6 단계 6: Config 시스템

**파일**: `configs/*.yaml`

YAML 기반 설정으로 모델, 데이터, 학습 파라미터를 관리한다.

```yaml
# classification.yaml 예시
backbone:
  model_name: vit_huge
  checkpoint_path: /path/to/ijepa_checkpoint.pth
  img_size: 224
  patch_size: 14
  freeze: true

task:
  type: classification
  num_classes: 1000
  head_mode: linear  # 'linear' or 'attentive'

data:
  train_dir: /path/to/train
  val_dir: /path/to/val
  batch_size: 64
  num_workers: 8

training:
  epochs: 50
  lr: 0.001
  weight_decay: 0.05
  scheduler: cosine
```

---

## 4. 구현 순서

| 순서 | 작업 | 의존성 |
|------|------|--------|
| 1 | `vision_transformer.py` - ViT 모델 정의 | 없음 |
| 2 | `checkpoint.py` - 체크포인트 로딩 유틸리티 | 1 |
| 3 | `ijepa_backbone.py` - backbone wrapper | 1, 2 |
| 4 | `classifier.py` - 분류 헤드 | 3 |
| 5 | `segmentor.py` - 세그멘테이션 디코더 | 3 |
| 6 | 데이터셋 클래스 | 없음 |
| 7 | Config 파일 | 없음 |
| 8 | 학습/평가 스크립트 | 4, 5, 6, 7 |

---

## 5. 기술 스택

- **PyTorch** >= 2.0
- **torchvision** - 데이터 transform 및 데이터셋
- **PyYAML** - config 관리
- **tensorboard** 또는 **wandb** - 학습 로깅 (선택)

---

## 6. 핵심 설계 원칙

1. **Backbone 분리**: I-JEPA encoder를 독립적인 모듈로 감싸서, downstream head와 완전히 분리한다
2. **Frozen by default**: backbone은 기본적으로 freeze 상태로 사용하되, config로 fine-tuning 전환 가능
3. **단순함 우선**: 불필요한 추상화 없이, 각 downstream task를 독립적으로 실행 가능하게 구성
4. **원본 호환**: I-JEPA 원본 레포의 `VisionTransformer` 코드를 최소 수정으로 사용하여 체크포인트 호환성 보장
