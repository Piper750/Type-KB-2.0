from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from src.heuristics import generate_experience, heuristic_solve_math
from src.schema import AbstractInfo, ExperienceInfo, ProblemRecord
from src.type_abstractor import BaseTypeAbstractor, RuleTypeAbstractor, create_type_abstractor


class BaseBackend:
    def abstract_problem(self, question: str) -> AbstractInfo:
        raise NotImplementedError

    def generate_experience(self, record: ProblemRecord, abstract_info: AbstractInfo) -> ExperienceInfo:
        raise NotImplementedError

    def solve(self, question: str, context: str = "", mode: str = "full") -> str:
        raise NotImplementedError


class MockBackend(BaseBackend):
    """Purely local backend for smoke tests and code validation."""

    def __init__(self, type_abstractor: Optional[BaseTypeAbstractor] = None) -> None:
        self.type_abstractor = type_abstractor or RuleTypeAbstractor()

    def abstract_problem(self, question: str) -> AbstractInfo:
        return self.type_abstractor.abstract_problem(question)

    def generate_experience(self, record: ProblemRecord, abstract_info: AbstractInfo) -> ExperienceInfo:
        return generate_experience(abstract_info, record.solution)

    def solve(self, question: str, context: str = "", mode: str = "full") -> str:
        answer = heuristic_solve_math(question)
        if answer is not None:
            return answer
        numbers = re.findall(r"\b-?\d+(?:\.\d+)?\b", context)
        return numbers[0] if numbers else "UNKNOWN"


class OpenAICompatibleBackend(BaseBackend):
    """Optional backend for real experiments with any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
        api_base: Optional[str] = None,
        temperature: float = 0.2,
        type_abstractor: Optional[BaseTypeAbstractor] = None,
        use_llm_for_type_abstraction: bool = False,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for OpenAICompatibleBackend. "
                "Please install dependencies with `pip install -r requirements.txt`."
            ) from exc

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Environment variable {api_key_env} is not set.")

        self.client = OpenAI(api_key=api_key, base_url=api_base or None)
        self.model_name = model_name
        self.temperature = temperature
        self.type_abstractor = type_abstractor or RuleTypeAbstractor()
        self.use_llm_for_type_abstraction = use_llm_for_type_abstraction

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def abstract_problem(self, question: str) -> AbstractInfo:
        if not self.use_llm_for_type_abstraction:
            return self.type_abstractor.abstract_problem(question)

        content = self._chat(
            system_prompt=(
                "You are a mathematical problem abstraction engine. "
                "Return strict JSON with keys: coarse_type, fine_type, skills, template, rationale."
            ),
            user_prompt=f"Question:\n{question}\n\nReturn JSON only.",
        )
        parsed = json.loads(content)
        return AbstractInfo(
            coarse_type=parsed.get("coarse_type", "general_math"),
            fine_type=parsed.get("fine_type", "general_math"),
            skills=list(parsed.get("skills", [])),
            template=parsed.get("template", ""),
            rationale=parsed.get("rationale", ""),
            confidence=float(parsed.get("confidence", 1.0)),
            label_source="llm",
            type_candidates=list(parsed.get("type_candidates", [])),
        )

    def generate_experience(self, record: ProblemRecord, abstract_info: AbstractInfo) -> ExperienceInfo:
        content = self._chat(
            system_prompt=(
                "You extract structured mathematical solving experience from a question and reference solution. "
                "Return strict JSON with keys: strategy_steps, key_principles, formulas, pitfalls, summary."
            ),
            user_prompt=(
                f"Question:\n{record.question}\n\n"
                f"Reference answer:\n{record.answer}\n\n"
                f"Reference solution:\n{record.solution}\n\n"
                f"Abstract type:\n{json.dumps(asdict(abstract_info), ensure_ascii=False)}\n\n"
                "Return JSON only."
            ),
        )
        parsed = json.loads(content)
        return ExperienceInfo(
            strategy_steps=list(parsed.get("strategy_steps", [])),
            key_principles=list(parsed.get("key_principles", [])),
            formulas=list(parsed.get("formulas", [])),
            pitfalls=list(parsed.get("pitfalls", [])),
            summary=parsed.get("summary", ""),
        )

    def solve(self, question: str, context: str = "", mode: str = "full") -> str:
        content = self._chat(
            system_prompt=(
                "You are a careful mathematical reasoner. "
                "Use the provided context when helpful, but do not copy unsupported content. "
                "At the end, output `FINAL_ANSWER: ` on a separate line."
            ),
            user_prompt=(
                f"Mode: {mode}\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n"
            ),
        )
        for line in content.splitlines()[::-1]:
            if line.strip().startswith("FINAL_ANSWER:"):
                return line.split("FINAL_ANSWER:", 1)[1].strip()
        return content.strip()


def build_backend(
    llm_config: Dict[str, Any],
    type_abstractor_config: Optional[Dict[str, Any]] = None,
    project_root: str | Path | None = None,
) -> BaseBackend:
    backend_name = str(llm_config.get("backend", "mock")).lower()
    type_abstractor = create_type_abstractor(type_abstractor_config, project_root=project_root)

    if backend_name == "mock":
        return MockBackend(type_abstractor=type_abstractor)

    if backend_name in {"openai", "openai_compatible"}:
        use_llm_for_type_abstraction = str((type_abstractor_config or {}).get("provider", "rule")).lower() == "llm"
        return OpenAICompatibleBackend(
            model_name=str(llm_config.get("model_name", "gpt-4o-mini")),
            api_key_env=str(llm_config.get("api_key_env", "OPENAI_API_KEY")),
            api_base=llm_config.get("api_base"),
            temperature=float(llm_config.get("temperature", 0.2)),
            type_abstractor=type_abstractor,
            use_llm_for_type_abstraction=use_llm_for_type_abstraction,
        )

    raise ValueError(f"Unsupported backend: {backend_name}")
