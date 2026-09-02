"""Proactive conversation without pretending observations are user requests."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any, Iterator

from brain.llm import ConversationWorker, _request_key, build_prompt, split_clauses


_PROACTIVE_MARKER = "\n\nCurrent request:\n"


@dataclass(frozen=True)
class _ProactiveRequest:
    generation_id: int
    trigger_name: str
    context: str

    def __iter__(self) -> Iterator[object]:
        """Stay replaceable by the base worker's latest-user-wins queue logic."""
        yield self.generation_id
        yield self.context


def build_proactive_prompt(
    config: dict[str, Any],
    moods: dict[str, dict[str, Any]],
    state: Any,
    trigger_name: str,
    context: str,
    *,
    performances: dict[str, dict[str, object]] | None = None,
    durable_memory: Any | None = None,
) -> str:
    """Reuse grounded prompt context while labeling the event as a system observation."""
    ordinary = build_prompt(
        config,
        moods,
        state,
        context,
        performances=performances,
        durable_memory=durable_memory,
    )
    prefix, marker, _ = ordinary.rpartition(_PROACTIVE_MARKER)
    if not marker:
        raise RuntimeError("ordinary prompt is missing its current-request boundary")

    instruction = (
        "Speak one short sentence as a natural observation or greeting. "
        "Do not ask a generic question. Do not mention triggers, timers, gates, or "
        "implementation details. Do not infer where the person was or what they were doing."
    )
    return (
        f"{prefix}\n\nProactive system observation:\n{context}\n\n"
        f"Speaking instruction:\n{instruction}"
    )


class ProactiveConversationWorker(ConversationWorker):
    """Add a low-priority proactive request kind to the proven conversation pipeline."""

    def submit(self, request: str) -> None:
        self._mark_user_interaction()
        super().submit(request)

    def submit_with_timing(
        self,
        request: str,
        timing: dict[str, object],
    ) -> None:
        self._mark_user_interaction()
        super().submit_with_timing(request, timing)

    def submit_proactive(self, trigger_name: str, context: str) -> bool:
        """Queue one proactive observation only when conversation is completely idle."""
        clean_name = trigger_name.strip()
        clean_context = context.strip()
        if not clean_name or not clean_context:
            return False

        with self._request_lock:
            if self._active_request_key is not None or self._pending_request_key is not None:
                self._state.record_debug(
                    "proactive_submission_rejected",
                    trigger=clean_name,
                    reason="conversation_busy",
                )
                return False

            self._next_generation += 1
            generation_id = self._next_generation
            self._latest_generation = generation_id
            self._latency_by_generation.clear()
            latency: dict[str, object] = {
                "first_clause_ready_at": None,
                "playback_reported": False,
                "first_tts_synthesis_reported": False,
            }
            self._latency_by_generation[generation_id] = latency

            key = _proactive_key(generation_id)
            self._pending_request_key = key
            self._voice.begin_generation(generation_id)
            self._requests.put_nowait(
                _ProactiveRequest(generation_id, clean_name, clean_context)
            )
            self._state.update_debug(
                conversation_queue=self._requests.qsize(),
                active_generation=generation_id,
                **self._latency_debug_values(generation_id, latency),
            )

        payload = {
            "trigger": clean_name,
            "generation_id": generation_id,
        }
        self._event_log.append("proactive_submitted", payload)
        self._state.record_debug("proactive_submitted", **payload)
        return True

    def _mark_user_interaction(self) -> None:
        with self._state.locked():
            self._state.last_interaction = time.time()

    def _run(self) -> None:
        while True:
            item = self._requests.get()
            if item is None:
                return

            if isinstance(item, _ProactiveRequest):
                generation_id = item.generation_id
                key = _proactive_key(generation_id)
                with self._request_lock:
                    if self._pending_request_key == key:
                        self._pending_request_key = None
                    self._active_request_key = key
                self._state.update_debug(conversation_queue=self._requests.qsize())
                try:
                    self._respond_proactive(item)
                finally:
                    with self._request_lock:
                        if self._active_request_key == key:
                            self._active_request_key = None
                continue

            generation_id, request = item
            key = _request_key(request)
            with self._request_lock:
                if self._pending_request_key == key:
                    self._pending_request_key = None
                self._active_request_key = key
            self._state.update_debug(conversation_queue=self._requests.qsize())
            try:
                self._respond(generation_id, request)
            finally:
                with self._request_lock:
                    if self._active_request_key == key:
                        self._active_request_key = None

    def _respond_proactive(self, request: _ProactiveRequest) -> None:
        generation_id = request.generation_id
        self._delivery.begin(generation_id, "")
        with self._state.locked():
            self._state.thinking = True

        llm_started_at = time.perf_counter()
        self._state.record_debug(
            "llm_started",
            request_kind="proactive",
            trigger=request.trigger_name,
            generation_id=generation_id,
        )
        try:
            prompt = build_proactive_prompt(
                self._config,
                self._moods,
                self._state,
                request.trigger_name,
                request.context,
                performances=self._performances,
                durable_memory=self._durable_memory,
            )
            first_clause = True
            for clause in split_clauses(
                self._client.stream(prompt, self._config),
                self._performances,
            ):
                if not self._is_latest(generation_id):
                    self._state.record_debug(
                        "stale_response_cancelled",
                        generation_id=generation_id,
                    )
                    return

                with self._state.locked():
                    self._state.thinking = False
                if first_clause:
                    self._record_first_clause_ready(
                        generation_id,
                        clause.text,
                        clause.performance.expression,
                        llm_started_at,
                    )
                    first_clause = False

                self._delivery.generated(generation_id, clause.text)
                if self._performances is None:
                    self._voice.enqueue(clause.text, generation_id=generation_id)
                else:
                    self._voice.enqueue(
                        clause.text,
                        generation_id=generation_id,
                        performance=clause.performance,
                    )

            if not self._is_latest(generation_id):
                self._state.record_debug(
                    "stale_response_cancelled",
                    generation_id=generation_id,
                )
                return

            self._delivery.llm_finished(generation_id)
            self._voice.finish_generation(generation_id)
            self._state.record_debug(
                "llm_complete",
                generation_id=generation_id,
                request_kind="proactive",
                trigger=request.trigger_name,
            )
        except Exception as error:
            payload = {
                "error": str(error),
                "trigger": request.trigger_name,
            }
            self._event_log.append("conversation_error", payload)
            self._state.record_debug("conversation_error", **payload)
        finally:
            with self._state.locked():
                self._state.thinking = False


def _proactive_key(generation_id: int) -> str:
    return f"<proactive:{generation_id}>"
