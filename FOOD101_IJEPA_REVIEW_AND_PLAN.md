# Food-101용 I-JEPA 분류기 검토 및 반영 계획

## 요약 결론
- **decoder 없이 encoder + classifier head** 전략이 맞음.
- 1차 기준선은 **mean pooling + linear**.
- 이후 저비용 고효율 실험 순서는
  1) **last-n layer pooled concat + MLP**
  2) **global + tiny local conv adapter fusion**.

## 현재 코드 반영 사항
1. `head_mode=linear`를 기본 baseline으로 유지.
2. `head_mode=multi_layer_mlp` 추가:
   - 마지막 `n`개 레이어 토큰을 평균풀링 후 concat하여 MLP 분류.
3. `head_mode=global_local_fusion` 추가:
   - global mean branch + depthwise conv 기반 local adapter branch 융합.
4. 백본에 `get_last_n_layer_tokens()` 추가:
   - I-JEPA [cls] 없는 토큰 표현을 레이어별로 추출해 head에서 활용.

## 실험 우선순위 (권장)

### Stage A — Baseline (고정 백본)
- `head_mode: linear`
- backbone freeze
- 목적: representation sanity check

### Stage B — 저비용 업그레이드
- `head_mode: multi_layer_mlp`
- `pool_last_n: 2 -> 4` ablation
- 목적: 마지막 단일 레이어 대비 성능 차이 검증

### Stage C — 본추천
- `head_mode: global_local_fusion`
- `local_ratio: 0.25 / 0.5 / 0.75` ablation
- 목적: Food-101의 local texture 민감도 반영

### Stage D — 파인튜닝
- A/B/C 중 best head 고정
- backbone 마지막 블록부터 점진 unfreeze
- head LR > backbone LR (예: 5x~10x)

## 학습/데이터 운영 체크리스트
- Food-101 클래스 수 101 확인 (`num_classes` 자동 추론 권장)
- `freeze=true`일 때 pretrained checkpoint 필수
- 공식 cleaned split은 최종 평가로 쓰고,
  noisy train에서 holdout validation 분리 권장
- 기본 augmentation은 mild하게 시작

## 이번 변경에서 반영된 항목
- layer-wise LR decay optimizer 그룹 자동 구성 (`training.use_layer_decay`, `training.layer_decay`)
- staged unfreeze 스케줄(에폭 기반) 자동화 (`training.unfreeze_schedule`)
- head/backbone LR 분리 (`training.head_lr_mult`)

## 이번 변경에서 미반영(추가 예정)
- 224->336 해상도 curriculum 학습 스크립트
