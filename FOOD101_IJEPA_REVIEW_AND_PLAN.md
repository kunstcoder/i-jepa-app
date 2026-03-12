# Food-101용 I-JEPA 분류기 상세 코드/설정 검토 및 실험 계획

## 1) 핵심 결론
- Food-101에서는 **encoder는 freeze**하고, 분류 head를 중심으로 실험하는 전략이 현재 코드 구조와 가장 잘 맞습니다.
- 기본 베이스라인은 `linear` head로 두고, 성능 향상은 `multi_layer_mlp` → `global_local_fusion` 순서가 합리적입니다.
- PyTorch AMP 관련 코드는 `torch.cuda.amp`에서 `torch.amp`로 마이그레이션하여 deprecated 스타일을 제거했습니다.

---

## 2) 코드 상세 리뷰

### 2-1. Backbone / 토큰 추출 경로
- `IJEPABackbone`은 `freeze=true` 시 encoder 파라미터의 `requires_grad=False`로 고정하고, frozen일 때만 `no_grad` 경로를 사용합니다.
- `get_last_n_layer_tokens()`가 있어 last-n 레이어 pooling 기반 head 실험에 바로 사용 가능합니다.
- `unfreeze_last_n_blocks()`가 이미 구현되어 있어 Stage D(점진 unfreeze)로 자연스럽게 확장 가능합니다.

**리뷰 판단**
- Food-101 frozen 실험에 필요한 기능(최종 토큰, last-n 토큰, 이후 unfreeze 확장)이 모두 준비된 상태입니다.

### 2-2. Classifier head 구조
- `linear`: mean pooling + linear.
- `multi_layer_mlp`: 마지막 n개 레이어 평균풀링 concat + MLP.
- `global_local_fusion`: global branch + depthwise conv local adapter branch 결합.

**리뷰 판단**
- Food-101처럼 질감/지역적 단서가 중요한 데이터셋에는 `global_local_fusion`가 특히 유효할 가능성이 큽니다.
- 단, 반드시 `linear` 기준선을 먼저 고정해 비교해야 개선폭 해석이 가능합니다.

### 2-3. Optimizer / 스케줄 / freeze 연동
- `build_optimizer()`는 head/backbone LR 분리 및 optional layer-wise decay를 지원합니다.
- freeze 기반 실험에서는 backbone이 학습 대상이 아니므로,
  - `head_lr_mult` 효과는 사실상 제한적,
  - `use_layer_decay=false`, `unfreeze_schedule=[]`로 두는 것이 깔끔합니다.

**리뷰 판단**
- 현재 optimizer 구현은 frozen 및 staged unfreeze 모두 대응 가능.
- 실험 설계에서 frozen 단계와 unfreeze 단계를 config로 명확히 분리하는 것이 중요합니다.

---

## 3) PyTorch deprecated 스타일 점검 결과

### 점검 결과 (수정 완료)
기존 코드에서 아래 구문이 deprecated 경고 대상이었습니다.
- `from torch.cuda.amp import autocast`
- `from torch.cuda.amp import GradScaler`
- `with autocast():`

이를 다음과 같이 수정했습니다.
- `from torch.amp import autocast`
- `from torch.amp import GradScaler`
- `with autocast(device_type=device.type):`
- `GradScaler("cuda") if mixed_precision and device.type == "cuda" else None`

### 영향
- 최신 PyTorch 권장 스타일과 정합성 확보.
- CPU 실행 시 불필요한 GradScaler 생성 방지.
- 기존 mixed precision 학습 동작 의도는 유지.

---

## 4) Food-101 (frozen backbone) 권장 하이퍼파라미터

### 공통 전제
- backbone: `vit_huge`, `patch_size=14`, `freeze=true`
- `num_classes=101`
- 입력 해상도: 224
- 기본 transform: 현재 코드의 ImageNet normalize + RandomResizedCrop/Flip 유지

### Stage A (Baseline)
- `head_mode: linear`
- `epochs: 60`
- `lr: 1.5e-3`
- `batch_size: 128`
- `weight_decay: 0.05`
- `label_smoothing: 0.1`
- `mixed_precision: true`

### Stage B (저비용 향상)
- `head_mode: multi_layer_mlp`
- `pool_last_n`: 2, 4 ablation
- `head_hidden_dim: 2048` (vit_huge 기준 안정적)
- 나머지 학습 파라미터는 Stage A 동일

### Stage C (권장 본실험)
- `head_mode: global_local_fusion`
- `local_ratio`: 0.25 / 0.5 / 0.75 ablation
- `head_hidden_dim: 2048`
- 나머지 학습 파라미터는 Stage A 동일

### Stage D (선택: 부분 unfreeze)
- Stage A/B/C best head 고정 후 아래만 변경
- `freeze: false`
- `head_lr_mult: 5.0`
- `use_layer_decay: true`, `layer_decay: 0.75`
- `unfreeze_schedule:`
  - `{epoch: 10, unfreeze_last_n: 4}`
  - `{epoch: 20, unfreeze_last_n: 8}`

---

## 5) 실험용 config 반영
- frozen backbone 전용 설정 파일을 추가:
  - `configs/food101_classification_frozen.yaml`
- 이 파일은 Food-101 분류의 Stage A~C 실험 시작점으로 바로 사용 가능하도록 구성했습니다.

---

## 6) 실행 예시
```bash
python train_classification.py \
  --config configs/food101_classification_frozen.yaml \
  --checkpoint /path/to/ijepa_vith14.pth \
  --device cuda
```

필요하면 다음 단계로 `food101_classification_unfreeze.yaml`도 분리해 Stage D를 재현 가능한 형태로 추가하는 것을 권장합니다.
