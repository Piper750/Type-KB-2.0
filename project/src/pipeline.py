from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from src.retriever import HybridRetriever
from src.schema import RetrievedItem
from src.type_abstractor import build_experience_query_from_info, build_type_query_from_info


class MathKBPipeline:
    def __init__(
        self,
        backend,
        retriever: HybridRetriever,
        top_k: int = 4,
        use_quality: bool = True,
        refine: bool = True,
        type_query_top_k: int = 3,
    ) -> None:
        self.backend = backend
        self.retriever = retriever
        self.top_k = top_k
        self.use_quality = use_quality
        self.refine = refine
        self.type_query_top_k = type_query_top_k

    def _build_context(self, query_info, retrieved_items: List[RetrievedItem], mode: str) -> str:
        if mode == "zero_shot":
            return ""

        blocks: List[str] = [
            (
                "[Current Type]\n"
                f"coarse_type: {query_info.coarse_type}\n"
                f"fine_type: {query_info.fine_type}\n"
                f"skills: {', '.join(query_info.skills)}\n"
                f"template: {query_info.template}\n"
                f"confidence: {getattr(query_info, 'confidence', 1.0):.4f}\n"
                f"label_source: {getattr(query_info, 'label_source', 'rule')}"
            )
        ]

        if getattr(query_info, "type_candidates", None):
            for rank, candidate in enumerate(query_info.type_candidates, start=1):
                blocks.append(
                    (
                        f"[Type Candidate {rank}]\n"
                        f"coarse_type: {candidate['coarse_type']}\n"
                        f"fine_type: {candidate['fine_type']}\n"
                        f"score: {float(candidate.get('score', 0.0)):.4f}\n"
                        f"skills: {', '.join(candidate.get('skills', []))}\n"
                        f"template: {candidate.get('template', '')}"
                    )
                )

        if mode == "type_only":
            for rank, item in enumerate(retrieved_items, start=1):
                blocks.append(
                    (
                        f"[Retrieved Type {rank}]\n"
                        f"fine_type: {item.entry.abstract_info.fine_type}\n"
                        f"coarse_type: {item.entry.abstract_info.coarse_type}\n"
                        f"template: {item.entry.abstract_info.template}"
                    )
                )
            return "\n\n".join(blocks)

        if mode in {"experience_only", "full"}:
            for rank, item in enumerate(retrieved_items, start=1):
                exp = item.entry.experience_info
                pieces = [
                    f"[Retrieved Experience {rank}]\nscore: {item.score:.4f}",
                    f"type: {item.entry.abstract_info.fine_type}",
                    f"summary: {exp.summary}",
                    f"steps: {' | '.join(exp.strategy_steps)}",
                    f"pitfalls: {' | '.join(exp.pitfalls)}",
                    f"principles: {' | '.join(exp.key_principles)}",
                    f"formulas: {' | '.join(exp.formulas)}",
                ]
                blocks.append("\n".join(pieces))

        return "\n\n".join(blocks)

    def predict(self, question: str, mode: str = "full") -> Dict[str, object]:
        query_info = self.backend.abstract_problem(question)
        query_type_text = build_type_query_from_info(query_info, top_k=self.type_query_top_k)
        query_experience_text = build_experience_query_from_info(query_info, top_k=self.type_query_top_k)

        retrieved_items: List[RetrievedItem] = []
        if mode != "zero_shot":
            retrieved_items = self.retriever.retrieve(
                query_type_text=query_type_text,
                query_question_text=question,
                query_experience_text=query_experience_text,
                top_k=self.top_k,
                use_quality=self.use_quality,
                refine=self.refine,
            )

        context = self._build_context(query_info, retrieved_items, mode)
        answer = self.backend.solve(question, context=context, mode=mode)
        return {
            "answer": answer,
            "query_info": asdict(query_info),
            "context": context,
            "retrieved_items": [item.to_dict() for item in retrieved_items],
        }
