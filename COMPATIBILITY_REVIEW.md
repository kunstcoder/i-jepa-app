# I-JEPA Backbone Compatibility Review (Meta official repository 기준)

## 검토 범위
- `src/models/vision_transformer.py`
- `src/models/ijepa_backbone.py`
- `src/utils/checkpoint.py`

## 결론
- 백본 아키텍처(모델 depth, heads, embed dim, patch embedding, sin-cos positional embedding)는 I-JEPA ViT 계열과 **구조적으로 호환**됩니다.
- 체크포인트 로더는 기존에는 key 포맷 차이(`module.`, `encoder.` 등)에서 실패/저품질 로딩 위험이 있었으나, 현재는 prefix 정규화와 포맷 분기(`encoder`, `target_encoder`, `state_dict`, `model`)를 추가하여 **실사용 호환성**을 강화했습니다.
- 기존 `IJEPABackbone.forward()`가 항상 `@torch.no_grad()`로 고정되어 있어 unfreeze 학습이 불가능한 문제가 있었고, 이를 frozen 상태에서만 no_grad를 쓰도록 수정해 **공식 파인튜닝 시나리오와 동작 일치성**을 개선했습니다.

## 확인 포인트
1. `model_name`과 checkpoint의 모델 스케일 일치(vit_huge/vit_giant 등)
2. `patch_size` 일치(14, 16 등)
3. 입력 해상도 차이가 있을 경우 pos-embed interpolation 경로 동작 확인
4. 체크포인트 저장 포맷(`encoder`, `state_dict`, `model`) 확인

## 남은 주의사항
- 공식 repo의 commit/branch별로 checkpoint dict schema가 달라질 수 있으므로, 새로운 포맷이 발견되면 `_pick_encoder_state` 분기를 확장해야 합니다.
