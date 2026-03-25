from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import load_records
from src.io_utils import ensure_dir, write_json, write_jsonl
from src.type_abstractor import LABEL_METADATA, RuleTypeAbstractor


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/learned_type.yaml")
    parser.add_argument("--artifact-dir", type=str, default="")
    args = parser.parse_args()

    try:
        import joblib
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "train_type_abstractor.py requires sentence-transformers, scikit-learn and joblib. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    config_path = resolve_path(PROJECT_ROOT, args.config)
    cfg = load_config(config_path)
    dataset_root = resolve_path(PROJECT_ROOT, cfg["paths"]["dataset_root"])
    train_records, _ = load_records(dataset_root)

    type_cfg = cfg.get("type_abstractor", {})
    train_cfg = type_cfg.get("train", {})
    encoder_name = str(train_cfg.get("encoder_name", "sentence-transformers/all-mpnet-base-v2"))
    normalize_embeddings = bool(train_cfg.get("normalize_embeddings", True))
    min_samples_per_fine_type = int(train_cfg.get("min_samples_per_fine_type", 1))

    if args.artifact_dir:
        artifact_dir = resolve_path(PROJECT_ROOT, args.artifact_dir)
    else:
        artifact_dir = resolve_path(PROJECT_ROOT, str(type_cfg.get("artifact_dir", "./outputs/type_abstractor")))
    ensure_dir(artifact_dir)

    labeler = RuleTypeAbstractor()
    labeled_rows: List[Dict[str, str]] = []
    for record in train_records:
        info = labeler.abstract_problem(record.question)
        labeled_rows.append(
            {
                "id": record.id,
                "question": record.question,
                "coarse_type": info.coarse_type,
                "fine_type": info.fine_type,
                "subject": record.subject,
                "difficulty": record.difficulty,
                "label_source": "rule",
            }
        )

    fine_counts = Counter(row["fine_type"] for row in labeled_rows)
    filtered_rows = [
        row for row in labeled_rows if fine_counts[row["fine_type"]] >= min_samples_per_fine_type
    ]
    if not filtered_rows:
        raise RuntimeError("No training rows available after min_samples_per_fine_type filtering.")

    encoder = SentenceTransformer(encoder_name)
    questions = [row["question"] for row in filtered_rows]
    X = encoder.encode(
        questions,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )
    X = np.asarray(X, dtype=float)

    coarse_y = [row["coarse_type"] for row in filtered_rows]
    coarse_clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="multinomial",
        random_state=42,
    )
    coarse_clf.fit(X, coarse_y)

    fine_clf_map: Dict[str, object] = {}
    rows_by_coarse: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(filtered_rows):
        rows_by_coarse[row["coarse_type"]].append(idx)

    for coarse_type, indices in rows_by_coarse.items():
        X_sub = X[indices]
        fine_labels = [filtered_rows[idx]["fine_type"] for idx in indices]
        unique_labels = sorted(set(fine_labels))
        if len(unique_labels) == 1:
            fine_clf_map[coarse_type] = {"mode": "constant", "label": unique_labels[0]}
            continue
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=42,
        )
        clf.fit(X_sub, fine_labels)
        fine_clf_map[coarse_type] = clf

    encoder.save(str(artifact_dir / "encoder"))
    joblib.dump(coarse_clf, artifact_dir / "coarse_clf.joblib")
    joblib.dump(fine_clf_map, artifact_dir / "fine_clf_map.joblib")

    label_metadata = {
        fine_type: LABEL_METADATA[fine_type]
        for fine_type in sorted({row["fine_type"] for row in filtered_rows})
    }
    write_json(artifact_dir / "label_metadata.json", label_metadata)
    write_json(
        artifact_dir / "manifest.json",
        {
            "encoder_name": encoder_name,
            "normalize_embeddings": normalize_embeddings,
            "num_train_records": len(train_records),
            "num_labeled_rows": len(labeled_rows),
            "num_filtered_rows": len(filtered_rows),
            "coarse_type_counts": dict(Counter(coarse_y)),
            "fine_type_counts": dict(Counter(row["fine_type"] for row in filtered_rows)),
        },
    )
    write_jsonl(artifact_dir / "weak_labels.jsonl", filtered_rows)

    print(f"[train_type_abstractor] saved artifacts to: {artifact_dir}")
    print(f"[train_type_abstractor] rows={len(filtered_rows)} coarse_types={len(set(coarse_y))}")


if __name__ == "__main__":
    main()
