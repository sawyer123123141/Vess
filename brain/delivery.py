"""Pure generation-scoped accounting for speech that was physically delivered."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field


FinalizeCallback = Callable[[int, str, str, str, str | None], None]


@dataclass
class _GenerationDelivery:
    user: str
    generated_clauses: list[str] = field(default_factory=list)
    completed_clauses: list[str] = field(default_factory=list)
    active_clause: str | None = None
    paused_clause: str | None = None
    llm_finished: bool = False
    playback_drained: bool = False


class DeliveryLedger:
    """Track generated text separately from speech confirmed by playback receipts."""

    def __init__(self, finalize: FinalizeCallback) -> None:
        self._finalize = finalize
        self._lock = threading.Lock()
        self._generations: dict[int, _GenerationDelivery] = {}
        self._finalized: set[int] = set()

    def begin(self, generation_id: int, user: str) -> None:
        """Start delivery accounting for one response generation."""
        with self._lock:
            if generation_id in self._finalized:
                return
            self._generations[generation_id] = _GenerationDelivery(user=user)

    def generated(self, generation_id: int, clause: str) -> None:
        """Record text produced by the LLM without claiming it was heard."""
        with self._lock:
            delivery = self._generations.get(generation_id)
            if delivery is not None:
                delivery.generated_clauses.append(clause)

    def handle(self, event_type: str, payload: dict[str, object]) -> None:
        """Consume one physical playback lifecycle receipt."""
        generation_id = payload.get("generation_id")
        if not isinstance(generation_id, int):
            return

        finalize_args: tuple[int, str, str, str, str | None] | None = None
        with self._lock:
            delivery = self._generations.get(generation_id)
            if delivery is None:
                return

            text_value = payload.get("text")
            text = text_value if isinstance(text_value, str) else None

            if event_type == "clause_started":
                delivery.active_clause = text
                delivery.paused_clause = None
            elif event_type == "clause_completed":
                if text is not None and delivery.active_clause == text:
                    delivery.completed_clauses.append(text)
                    delivery.active_clause = None
                    delivery.paused_clause = None
            elif event_type == "clause_paused":
                if text is not None and delivery.active_clause == text:
                    delivery.paused_clause = text
            elif event_type == "clause_resumed":
                if text is not None:
                    delivery.active_clause = text
                    delivery.paused_clause = None
            elif event_type == "clause_abandoned":
                if text is not None and delivery.active_clause == text:
                    delivery.paused_clause = text
                delivery.active_clause = None
            elif event_type == "generation_playback_drained":
                delivery.playback_drained = True

            if delivery.llm_finished and delivery.playback_drained:
                finalize_args = self._finalize_locked(
                    generation_id,
                    status="completed",
                    interrupted_clause=None,
                )

        if finalize_args is not None:
            self._finalize(*finalize_args)

    def llm_finished(self, generation_id: int) -> None:
        """Mark generation complete without treating that as playback completion."""
        finalize_args: tuple[int, str, str, str, str | None] | None = None
        with self._lock:
            delivery = self._generations.get(generation_id)
            if delivery is None:
                return
            delivery.llm_finished = True
            if delivery.playback_drained:
                finalize_args = self._finalize_locked(
                    generation_id,
                    status="completed",
                    interrupted_clause=None,
                )

        if finalize_args is not None:
            self._finalize(*finalize_args)

    def interrupt(self, generation_id: int) -> bool:
        """Finalize an interrupted response using only fully completed clauses."""
        with self._lock:
            delivery = self._generations.get(generation_id)
            if delivery is None:
                return False
            interrupted_clause = delivery.paused_clause or delivery.active_clause
            finalize_args = self._finalize_locked(
                generation_id,
                status="interrupted",
                interrupted_clause=interrupted_clause,
            )

        if finalize_args is not None:
            self._finalize(*finalize_args)
        return finalize_args is not None

    def _finalize_locked(
        self,
        generation_id: int,
        *,
        status: str,
        interrupted_clause: str | None,
    ) -> tuple[int, str, str, str, str | None] | None:
        delivery = self._generations.pop(generation_id, None)
        if delivery is None:
            return None
        self._finalized.add(generation_id)
        assistant = " ".join(delivery.completed_clauses).strip()
        return generation_id, delivery.user, assistant, status, interrupted_clause
