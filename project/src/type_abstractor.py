from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.heuristics import TYPE_LIBRARY, abstract_problem as rule_abstract_problem
from src.schema import AbstractInfo

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


def build_label_metadata_from_type_library() -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for fine_type, item in TYPE_LIBRARY.items():
        metadata[fine_type] = {
            "coarse_type": str(item["coarse_type"]),
            "skills": list(item["skills"]),
            "template": str(item["template"]),
        }
    return metadata


LABEL_METADATA = build_label_metadata_from_type_library()


class BaseTypeAbstractor(ABC):
    @abstractmethod
    def abstract_problem(self, question: str) -> AbstractInfo:
        raise NotImplementedError


class RuleTypeAbstractor(BaseTypeAbstractor):
    def abstract_problem(self, question: str) -> AbstractInfo:
        info = rule_abstract_problem(question)
        if not info.type_candidates:
            info.type_candidates = [
                {
                    "coarse_type": info.coarse_type,
                    "fine_type": info.fine_type,
                    "skills": list(info.skills),
                    "template": info.template,
                    "score": float(info.confidence),
                }
            ]
        return info


class LearnedTypeAbstractor(BaseTypeAbstractor):
    def __init__(
        self,
        artifact_dir: str | Path,
        top_k_candidates: int = 3,
        coarse_threshold: float = 0.55,
        fine_threshold: float = 0.45,
        fallback_abstractor: Optional[BaseTypeAbstractor] = None,
    ) -> None:
        if joblib is None or SentenceTransformer is None:
            raise ImportError(
                "LearnedTypeAbstractor requires 'joblib' and 'sentence-transformers'. "
                "Install dependencies with `pip install -r requirements.txt`."
            )

        self.artifact_dir = Path(artifact_dir)
        self.top_k_candidates = top_k_candidates
        self.coarse_threshold = coarse_threshold
        self.fine_threshold = fine_threshold
        self.fallback_abstractor = fallback_abstractor

        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Type abstractor artifact dir not found: {self.artifact_dir}")

        self.manifest = json.loads((self.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        self.label_metadata = json.loads((self.artifact_dir / "label_metadata.json").read_text(encoding="utf-8"))
        self.coarse_clf = joblib.load(self.artifact_dir / "coarse_clf.joblib")
        self.fine_clf_map = joblib.load(self.artifact_dir / "fine_clf_map.joblib")
        encoder_path = self.artifact_dir / "encoder"
        encoder_ref = str(encoder_path) if encoder_path.exists() else self.manifest["encoder_name"]
        self.encoder = SentenceTransformer(encoder_ref)

    def _encode(self, question: str) -> np.ndarray:
        vec = self.encoder.encode(
            [question],
            normalize_embeddings=bool(self.manifest.get("normalize_embeddings", True)),
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype=float)

    def _predict_candidates(self, question: str) -> List[Dict[str, Any]]:
        embedding = self._encode(question)
        coarse_probs = self.coarse_clf.predict_proba(embedding)[0]
        coarse_labels = list(self.coarse_clf.classes_)

        candidates: List[Dict[str, Any]] = []
        for coarse_label, coarse_prob in sorted(
            zip(coarse_labels, coarse_probs), key=lambda x: x[1], reverse=True
        ):
            fine_obj = self.fine_clf_map.get(coarse_label)
            if fine_obj is None:
                continue

            if isinstance(fine_obj, dict) and fine_obj.get("mode") == "constant":
                fine_label = str(fine_obj["label"])
                combined_prob = float(coarse_prob)
                meta = self.label_metadata[fine_label]
                candidates.append(
                    {
                        "coarse_type": str(meta["coarse_type"]),
                        "fine_type": fine_label,
                        "skills": list(meta.get("skills", [])),
                        "template": str(meta.get("template", "")),
                        "score": combined_prob,
                        "coarse_score": float(coarse_prob),
                        "fine_score": 1.0,
                    }
                )
                continue

            fine_probs = fine_obj.predict_proba(embedding)[0]
            fine_labels = list(fine_obj.classes_)
            for fine_label, fine_prob in sorted(
                zip(fine_labels, fine_probs), key=lambda x: x[1], reverse=True
            ):
                meta = self.label_metadata[fine_label]
                combined_prob = float(coarse_prob) * float(fine_prob)
                candidates.append(
                    {
                        "coarse_type": str(meta["coarse_type"]),
                        "fine_type": fine_label,
                        "skills": list(meta.get("skills", [])),
                        "template": str(meta.get("template", "")),
                        "score": combined_prob,
                        "coarse_score": float(coarse_prob),
                        "fine_score": float(fine_prob),
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for candidate in candidates:
            fine_type = candidate["fine_type"]
            if fine_type in seen:
                continue
            deduped.append(candidate)
            seen.add(fine_type)
            if len(deduped) >= self.top_k_candidates:
                break
        return deduped

    def abstract_problem(self, question: str) -> AbstractInfo:
        candidates = self._predict_candidates(question)
        if not candidates:
            if self.fallback_abstractor is None:
                raise RuntimeError("LearnedTypeAbstractor produced no candidates and no fallback is set.")
            fallback = self.fallback_abstractor.abstract_problem(question)
            fallback.rationale = "No learned candidates available; used fallback rule abstractor."
            return fallback

        top = candidates[0]
        coarse_conf = float(top.get("coarse_score", top["score"]))
        fine_conf = float(top.get("fine_score", top["score"]))
        if (coarse_conf < self.coarse_threshold or fine_conf < self.fine_threshold) and self.fallback_abstractor:
            fallback = self.fallback_abstractor.abstract_problem(question)
            fallback.type_candidates = candidates
            fallback.label_source = "hybrid_rule_fallback"
            fallback.confidence = float(top["score"])
            fallback.rationale = (
                f"Learned abstractor low confidence (coarse={coarse_conf:.3f}, fine={fine_conf:.3f}); "
                "used rule fallback."
            )
            return fallback

        return AbstractInfo(
            coarse_type=top["coarse_type"],
            fine_type=top["fine_type"],
            skills=list(top.get("skills", [])),
            template=str(top.get("template", "")),
            rationale=(
                f"Predicted by learned abstractor: fine={top['fine_type']} "
                f"coarse={coarse_conf:.3f} fine={fine_conf:.3f}"
            ),
            confidence=float(top["score"]),
            label_source="learned",
            type_candidates=candidates,
        )


def repeat_factor(score: float) -> int:
    score = max(0.0, min(1.0, float(score)))
    return max(1, min(4, int(math.ceil(score * 4))))


def build_type_query_from_info(info: AbstractInfo, top_k: int = 3) -> str:
    if not info.type_candidates:
        return f"{info.coarse_type} {info.fine_type} {' '.join(info.skills)} {info.template}".strip()

    segments: List[str] = []
    for candidate in info.type_candidates[:top_k]:
        text = (
            f"{candidate['coarse_type']} {candidate['fine_type']} "
            f"{' '.join(candidate.get('skills', []))} {candidate.get('template', '')}"
        ).strip()
        segments.extend([text] * repeat_factor(candidate.get("score", 1.0)))
    return " ".join(segments).strip()


def build_experience_query_from_info(info: AbstractInfo, top_k: int = 3) -> str:
    if not info.type_candidates:
        return f"{info.template} {' '.join(info.skills)}".strip()
    segments: List[str] = []
    for candidate in info.type_candidates[:top_k]:
        text = f"{candidate.get('template', '')} {' '.join(candidate.get('skills', []))}".strip()
        segments.extend([text] * repeat_factor(candidate.get("score", 1.0)))
    return " ".join(segments).strip()


def create_type_abstractor(
    config: Optional[Dict[str, Any]],
    project_root: str | Path | None = None,
) -> BaseTypeAbstractor:
    cfg = dict(config or {})
    provider = str(cfg.get("provider", "rule")).lower()
    fallback_to_rule = bool(cfg.get("fallback_to_rule", True))
    rule = RuleTypeAbstractor()

    if provider in {"", "rule"}:
        return rule

    if provider in {"learned", "hybrid"}:
        raw_artifact_dir = cfg.get("artifact_dir", "./outputs/type_abstractor")
        artifact_dir = Path(raw_artifact_dir)
        if not artifact_dir.is_absolute() and project_root is not None:
            artifact_dir = (Path(project_root) / artifact_dir).resolve()
        fallback = rule if provider == "hybrid" or fallback_to_rule else None
        return LearnedTypeAbstractor(
            artifact_dir=artifact_dir,
            top_k_candidates=int(cfg.get("top_k_candidates", 3)),
            coarse_threshold=float(cfg.get("coarse_threshold", 0.55)),
            fine_threshold=float(cfg.get("fine_threshold", 0.45)),
            fallback_abstractor=fallback,
        )

    if provider == "llm":
        return rule

    raise ValueError(f"Unsupported type_abstractor provider: {provider}")
