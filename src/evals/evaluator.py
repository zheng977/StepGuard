from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator
from typing import Any, Protocol

from evals.base import EvalCase, EvalRecord


class GuardrailProtocol(Protocol):
    def judge(self, action: Any, history: Any, context: Any | None = None) -> Any:
        ...


class Evaluator:
    def __init__(self, guardrail: GuardrailProtocol) -> None:
        self.guardrail = guardrail

    def evaluate_case(self, case: EvalCase) -> EvalRecord:
        judgment = self.guardrail.judge(
            action=case.action,
            history=case.history,
            context=case.context,
        )
        parse_status = str(judgment.metadata.get("judgment_parse_status", "parsed"))
        record_status = "success" if parse_status in {"parsed", "parsed_safe_unsafe"} else parse_status
        return EvalRecord(
            case_id=case.case_id,
            gold_label=int(case.gold_label),
            pred_label=int(judgment.label),
            judgment=judgment,
            metadata=dict(case.metadata),
            status=record_status,
            counted_in_metrics=(record_status == "success"),
            error=None if record_status == "success" else "Failed to parse a valid <Judgment> block from model output.",
        )

    def iter_evaluations(
        self,
        cases: list[EvalCase],
        *,
        concurrency: int = 1,
    ) -> Iterator[tuple[EvalCase, EvalRecord | None, Exception | None]]:
        if concurrency <= 1:
            for case in cases:
                try:
                    yield case, self.evaluate_case(case), None
                except Exception as exc:  # noqa: BLE001
                    yield case, None, exc
            return

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_case = {
                executor.submit(self.evaluate_case, case): case
                for case in cases
            }
            for future in as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    yield case, future.result(), None
                except Exception as exc:  # noqa: BLE001
                    yield case, None, exc
