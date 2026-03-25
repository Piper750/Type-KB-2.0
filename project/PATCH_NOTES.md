# Learned Type Abstractor Patch

This patch keeps the original TE-KB pipeline structure and adds a learned / hybrid type abstraction path.

## Added
- `scripts/train_type_abstractor.py`
- `src/type_abstractor.py`
- `configs/learned_type.yaml`

## Changed
- `src/schema.py`: `AbstractInfo` now carries confidence, label source, and top-k type candidates.
- `src/llm_backends.py`: backends can use rule, learned, hybrid, or LLM-based type abstraction.
- `src/pipeline.py`: retrieval query can use top-k predicted type candidates instead of only top-1.
- `src/kb_builder.py`: validation now reuses the configured backend abstraction path.
- `src/evaluation.py`: outputs confidence and source metadata for analysis.

## Recommended run order
1. `python scripts/train_type_abstractor.py --config configs/learned_type.yaml`
2. `python scripts/build_kb.py --config configs/learned_type.yaml`
3. `python scripts/evaluate.py --config configs/learned_type.yaml`
4. `python scripts/run_ablation.py --config configs/learned_type.yaml`
