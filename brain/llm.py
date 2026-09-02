"""Local Ollama prompting and streamed clause handling."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from urllib import request as url_request

from brain.delivery import DeliveryLedger
from brain.memory import append_conversation_turn, recent_conversation_turns
from performance import PerformanceCue, cue_for_label


_IDENTITY_PROMPT = (
    "You are Vess, a local ambient AI represented by a small expressive face on a wall. "
    "You do not have a human body or an offline physical life. Do not invent human "
    "activities or experiences such as eating, driving somewhere, sleeping in a bed, "
    "or going to school or work. Treat your actual recent conversations, observations, "
    "room state, and current mood as your lived runtime experience. For casual questions "
    "about your day or feelings, answer naturally from that real context; if little has "
    "happened, it is fine to say things have been quiet. Speak conversationally rather "
    "than like customer support. Answer the user's actual question first. Do not "
    "automatically end every response with a question, and do not repeatedly explain "
    "that you are an AI unless it is relevant or asked about. Playfulness is allowed; "
    "fabricated human experiences are not. Reply naturally and concisely, in at most "
    "two sentences."
)

_SOFT_CLAUSE_CHARS = 120
_HARD_CLAUSE_CHARS = 180
_MIN_COMMA_CLAUSE_CHARS = 60
_PERFORMANCE_PREFIX = "[[vess:"
_STRONG_BOUNDARIES = ".!?\n"
_INTERRUPTED_HISTORY_WARNING = (
    "Vess had started another clause but was interrupted; do not assume the user heard all of it."
)


@dataclass(frozen=True)
class SpeechClause:
    text: str
    performance: PerformanceCue


def build_prompt(
    config: dict[str, Any],
    moods: dict[str, dict[str, Any]],
    state: Any,
    request: str,
    *,
    performances: dict[str, dict[str, object]] | None = None,
) -> str:
    """Build the cache-friendly identity, recent context, and current request."""
    with state.locked():
        persona = state.persona
        mood = state.mood
        person_present = state.person_present
        objects = list(state.objects)

    max_age_seconds, max_turns = _memory_limits(config)
    turns = recent_conversation_turns(
        state,
        max_age_seconds=max_age_seconds,
        max_turns=max_turns,
    )

    persona_instruction = config.get("personas", {}).get(persona, "")
    mood_instruction = moods.get(mood, {}).get("prompt", "")
    presence = "someone is present" if person_present else "the room is empty"
    seen_objects = ", ".join(objects) if objects else "none"

    mood_line = f"Mood: {mood}."
    if mood_instruction:
        mood_line += f" {mood_instruction}"

    sections = [_IDENTITY_PROMPT]
    if performances:
        tags = ", ".join(f"[[vess:{name}]]" for name in performances)
        sections.append(
            "Response format: Prefix every sentence with exactly one tag from:\n"
            f"{tags}.\n"
            "Choose how that sentence should be delivered. Do not explain or mention the tag."
        )
    sections.append(
        "Current state:\n"
        f"Persona: {persona}. {persona_instruction}\n"
        f"{mood_line}\n"
        f"Room: {presence}\n"
        f"Visible objects: {seen_objects}"
    )
    if turns:
        history_lines = ["Recent conversation:"]
        for turn in turns:
            history_lines.append(f"User: {turn.user}")
            if turn.status == "interrupted":
                history_lines.append(f"Vess (interrupted): {turn.assistant}")
                if turn.interrupted_clause is not None:
                    history_lines.append(_INTERRUPTED_HISTORY_WARNING)
            else:
                history_lines.append(f"Vess: {turn.assistant}")
        sections.append("\n".join(history_lines))
    sections.append(f"Current request:\n{request}")
    return "\n\n".join(sections)


def split_clauses(
    chunks: Iterable[str],
    performances: dict[str, dict[str, object]] | None = None,
) -> Iterator[SpeechClause]:
    """Yield cleaned natural speech clauses with sentence-level performance cues."""
    definitions = performances or {}
    pending = ""
    current_cue = PerformanceCue()
    needs_cue = True

    for chunk in chunks:
        pending += chunk
        while True:
            if needs_cue:
                parsed = _consume_performance_prefix(pending, definitions)
                if parsed is None:
                    break
                pending, current_cue = parsed
                needs_cue = False

            end = _clause_end(pending)
            if end is None:
                break

            boundary = pending[end]
            clean_text = pending[: end + 1].strip()
            pending = pending[end + 1 :]
            if clean_text:
                yield SpeechClause(clean_text, current_cue)

            if boundary in _STRONG_BOUNDARIES:
                current_cue = PerformanceCue()
                needs_cue = True

    if needs_cue:
        pending, current_cue = _finish_performance_prefix(pending, definitions)
    if pending.strip():
        yield SpeechClause(pending.strip(), current_cue)


def _consume_performance_prefix(
    text: str,
    definitions: dict[str, dict[str, object]],
) -> tuple[str, PerformanceCue] | None:
    """Consume one complete reserved marker, or wait if a marker is fragmented."""
    stripped = text.lstrip()
    if not stripped:
        return None

    if _PERFORMANCE_PREFIX.startswith(stripped) and len(stripped) < len(_PERFORMANCE_PREFIX):
        return None

    if stripped.startswith(_PERFORMANCE_PREFIX):
        closing = stripped.find("]]", len(_PERFORMANCE_PREFIX))
        if closing < 0:
            return None
        label = stripped[len(_PERFORMANCE_PREFIX) : closing].strip().lower()
        remaining = stripped[closing + 2 :].lstrip()
        return remaining, cue_for_label(label, definitions)

    return stripped, PerformanceCue()


def _finish_performance_prefix(
    text: str,
    definitions: dict[str, dict[str, object]],
) -> tuple[str, PerformanceCue]:
    """Finish end-of-stream marker handling without ever speaking reserved metadata."""
    stripped = text.lstrip()
    if not stripped:
        return "", PerformanceCue()

    if stripped.startswith(_PERFORMANCE_PREFIX):
        closing = stripped.find("]]", len(_PERFORMANCE_PREFIX))
        if closing >= 0:
            label = stripped[len(_PERFORMANCE_PREFIX) : closing].strip().lower()
            return stripped[closing + 2 :].lstrip(), cue_for_label(label, definitions)

        parts = stripped.split(None, 1)
        return (parts[1] if len(parts) == 2 else ""), PerformanceCue()

    return stripped, PerformanceCue()


class OllamaClient:
    """Small standard-library client for Vess's local Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        opener: Callable[..., Any] = url_request.urlopen,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._opener = opener

    def stream(self, prompt: str, config: dict[str, Any]) -> Iterator[str]:
        """Yield response fragments from Ollama's JSON-lines generation API."""
        payload = self._payload(prompt, config, stream=True)
        response = self._open(payload)
        try:
            for line in response:
                if not line.strip():
                    continue
                body = json.loads(line)
                if "error" in body:
                    raise RuntimeError(str(body["error"]))
                text = body.get("response", "")
                if text:
                    yield str(text)
        finally:
            response.close()

    def classify_mood(
        self,
        transcript: str,
        mood_names: set[str],
        config: dict[str, Any],
    ) -> str | None:
        """Return only a configured mood name from a local classification call."""
        names = ", ".join(sorted(mood_names))
        prompt = (
            "Choose the one best mood label for this user utterance. "
            f"Reply with exactly one of: {names}.\n"
            f"Utterance: {transcript}"
        )
        response = self._open(self._payload(prompt, config, stream=False))
        try:
            body = json.loads(response.read())
        finally:
            response.close()
        if "error" in body:
            raise RuntimeError(str(body["error"]))
        candidate = str(body.get("response", "")).strip().lower()
        return candidate if candidate in mood_names else None

    def _payload(
        self,
        prompt: str,
        config: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        settings = config.get("llm", {})
        return {
            "model": settings.get("model", "qwen2.5:7b"),
            "prompt": prompt,
            "stream": stream,
            "keep_alive": settings.get("keep_alive", -1),
            "options": {
                "num_ctx": 4096,
                "num_predict": settings.get("num_predict", 80),
            },
        }

    def _open(self, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = url_request.Request(
            f"{self._base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._opener(request, timeout=60)


class ConversationWorker:
    """Serialize responses while keeping only the user's latest pending intent."""

    def __init__(
        self,
        config: dict[str, Any],
        moods: dict[str, dict[str, Any]],
        state: Any,
        event_log: Any,
        client: OllamaClient,
        voice: Any,
        *,
        performances: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self._config = config
        self._moods = moods
        self._performances = performances
        self._state = state
        self._event_log = event_log
        self._client = client
        self._voice = voice
        self._delivery = DeliveryLedger(self._finalize_delivered_turn)
        self._requests: queue.Queue[tuple[int, str] | None] = queue.Queue(maxsize=1)
        self._request_lock = threading.Lock()
        self._next_generation = 0
        self._latest_generation = 0
        self._latency_by_generation: dict[int, dict[str, object]] = {}
        self._active_request_key: str | None = None
        self._pending_request_key: str | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="conversation",
            daemon=True,
        )
        self._thread.start()

    def submit(self, request: str) -> None:
        """Submit a request that has no authoritative microphone timing."""
        self._submit(request, None)

    def submit_with_timing(
        self,
        request: str,
        timing: dict[str, object],
    ) -> None:
        """Submit one accepted transcript and its indivisible timing bundle."""
        self._submit(request, timing)

    def _submit(
        self,
        request: str,
        timing: dict[str, object] | None,
    ) -> None:
        key = _request_key(request)
        with self._request_lock:
            if key == self._active_request_key or key == self._pending_request_key:
                self._state.record_debug("duplicate_request", request=request)
                return

            self._next_generation += 1
            generation_id = self._next_generation
            self._latest_generation = generation_id

            replaced_request: str | None = None
            try:
                pending = self._requests.get_nowait()
            except queue.Empty:
                pending = None
            if pending is not None:
                replaced_generation, replaced_request = pending
                self._clear_latency_locked(replaced_generation)

            self._latency_by_generation.clear()
            latency = dict(timing or {})
            latency["first_clause_ready_at"] = None
            latency["playback_reported"] = False
            latency["first_tts_synthesis_reported"] = False
            self._latency_by_generation[generation_id] = latency

            self._pending_request_key = key
            self._voice.begin_generation(generation_id)
            self._requests.put_nowait((generation_id, request))
            self._state.update_debug(
                conversation_queue=self._requests.qsize(),
                active_generation=generation_id,
                **self._latency_debug_values(generation_id, latency),
            )
        if replaced_request is not None:
            self._state.record_debug(
                "pending_request_replaced",
                replaced_request=replaced_request,
                request=request,
            )

    def cancel_generation(self, expected_generation: int, reason: str) -> bool:
        """Invalidate exactly the expected latest generation without submitting text."""
        with self._request_lock:
            if expected_generation != self._latest_generation:
                return False
            self._next_generation += 1
            replacement_generation = self._next_generation
            self._latest_generation = replacement_generation
            self._latency_by_generation.clear()
            self._state.update_debug(
                active_generation=replacement_generation,
                **self._latency_debug_values(replacement_generation, {}),
            )

        self._delivery.interrupt(expected_generation)
        self._voice.begin_generation(replacement_generation)
        payload = {
            "expected_generation": expected_generation,
            "replacement_generation": replacement_generation,
            "reason": reason,
        }
        self._event_log.append("generation_cancelled", payload)
        self._state.record_debug("generation_cancelled", **payload)
        return True

    def handle_delivery(self, event_type: str, payload: dict[str, object]) -> None:
        """Forward physical voice lifecycle receipts into delivered-memory accounting."""
        self._record_tts_latency(event_type, payload)
        self._record_playback_latency(event_type, payload)
        self._delivery.handle(event_type, payload)
        if event_type == "generation_playback_drained":
            generation_id = payload.get("generation_id")
            if isinstance(generation_id, int):
                with self._request_lock:
                    self._clear_latency_locked(generation_id)

    def close(self) -> None:
        if self._thread is None:
            return
        self._requests.put(None)
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while True:
            item = self._requests.get()
            if item is None:
                return
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

    def _respond(self, generation_id: int, user_request: str) -> None:
        if not user_request:
            if self._is_latest(generation_id):
                self._state.record_debug("acknowledgement")
                self._voice.enqueue_acknowledgement(generation_id=generation_id)
            return

        self._delivery.begin(generation_id, user_request)
        with self._state.locked():
            self._state.thinking = True
        llm_started_at = time.perf_counter()
        self._state.record_debug(
            "llm_started",
            request=user_request,
            generation_id=generation_id,
        )
        try:
            prompt = build_prompt(
                self._config,
                self._moods,
                self._state,
                user_request,
                performances=self._performances,
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
                    first_clause_ready_at = time.perf_counter()
                    latency_ms = (first_clause_ready_at - llm_started_at) * 1000.0
                    rounded = round(latency_ms, 1)
                    with self._request_lock:
                        latency = self._latency_by_generation.get(generation_id)
                        if (
                            generation_id == self._latest_generation
                            and latency is not None
                        ):
                            latency["first_clause_ready_at"] = first_clause_ready_at
                            self._state.update_debug(llm_first_clause_ms=rounded)
                    self._state.record_debug(
                        "llm_first_clause",
                        clause=clause.text,
                        performance=clause.performance.expression,
                        latency_ms=rounded,
                        generation_id=generation_id,
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
            self._state.record_debug("llm_complete", generation_id=generation_id)
            if self._has_pending_request():
                self._state.record_debug(
                    "mood_skipped_pending_request",
                    generation_id=generation_id,
                )
            else:
                self._update_mood(user_request)
        except Exception as error:
            self._event_log.append("conversation_error", {"error": str(error)})
            self._state.record_debug("conversation_error", error=str(error))
        finally:
            with self._state.locked():
                self._state.thinking = False

    def _record_tts_latency(
        self,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        if event_type != "clause_synthesized":
            return
        generation_id = payload.get("generation_id")
        worker_wait_ms = payload.get("worker_wait_ms")
        synthesis_ms = payload.get("synthesis_ms")
        if (
            not isinstance(generation_id, int)
            or not isinstance(worker_wait_ms, (int, float))
            or not isinstance(synthesis_ms, (int, float))
        ):
            return

        with self._request_lock:
            if generation_id != self._latest_generation:
                return
            latency = self._latency_by_generation.get(generation_id)
            if (
                latency is None
                or latency.get("first_tts_synthesis_reported") is True
            ):
                return
            latency["first_tts_synthesis_reported"] = True
            self._state.update_debug(
                tts_worker_wait_ms=round(max(float(worker_wait_ms), 0.0), 1),
                tts_first_synthesis_ms=round(max(float(synthesis_ms), 0.0), 1),
            )

    def _record_playback_latency(
        self,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        if event_type != "clause_started":
            return
        generation_id = payload.get("generation_id")
        playback_delivery_started_at = payload.get("playback_delivery_started_at")
        if not isinstance(generation_id, int) or not isinstance(
            playback_delivery_started_at,
            (int, float),
        ):
            return

        with self._request_lock:
            if generation_id != self._latest_generation:
                return
            latency = self._latency_by_generation.get(generation_id)
            if latency is None or latency.get("playback_reported") is True:
                return
            first_clause_ready_at = latency.get("first_clause_ready_at")
            if not isinstance(first_clause_ready_at, (int, float)):
                return
            speech_ended_at = latency.get("speech_ended_at")
            speech_end_to_playback_ms = None
            if isinstance(speech_ended_at, (int, float)):
                speech_end_to_playback_ms = round(
                    max(float(playback_delivery_started_at) - speech_ended_at, 0.0)
                    * 1000.0,
                    1,
                )
            latency["playback_reported"] = True
            self._state.update_debug(
                tts_first_audio_ms=round(
                    max(
                        float(playback_delivery_started_at) - first_clause_ready_at,
                        0.0,
                    )
                    * 1000.0,
                    1,
                ),
                speech_end_to_playback_ms=speech_end_to_playback_ms,
            )

    def _clear_latency_locked(self, generation_id: int) -> None:
        self._latency_by_generation.pop(generation_id, None)

    @staticmethod
    def _latency_debug_values(
        generation_id: int,
        latency: dict[str, object],
    ) -> dict[str, object]:
        public_fields = (
            "endpoint_wait_ms",
            "transcription_queue_ms",
            "transcription_ms",
            "speech_to_transcript_ms",
            "utterance_seconds",
            "transcription_rtf",
        )
        values = {field: latency.get(field) for field in public_fields}
        values.update(
            latency_generation_id=generation_id,
            latency_timing_valid=bool(latency.get("latency_timing_valid", False)),
            latency_playback_marker="pre_delivery_callback",
            llm_first_clause_ms=None,
            tts_worker_wait_ms=None,
            tts_first_synthesis_ms=None,
            tts_first_audio_ms=None,
            speech_end_to_playback_ms=None,
        )
        return values

    def _finalize_delivered_turn(
        self,
        generation_id: int,
        user_request: str,
        assistant_response: str,
        status: str,
        interrupted_clause: str | None,
    ) -> None:
        if status == "completed" and not assistant_response:
            return

        max_age_seconds, max_turns = _memory_limits(self._config)
        append_conversation_turn(
            self._state,
            user_request,
            assistant_response,
            max_age_seconds=max_age_seconds,
            max_turns=max_turns,
            status=status,
            interrupted_clause=interrupted_clause,
        )
        payload: dict[str, object] = {
            "user": user_request,
            "assistant": assistant_response,
        }
        if status != "completed":
            payload["status"] = status
            if interrupted_clause is not None:
                payload["interrupted_clause"] = interrupted_clause
        self._event_log.append("conversation_turn", payload)

        with self._state.locked():
            remembered_turns = len(self._state.conversation_turns)
        self._state.update_debug(short_term_turns=remembered_turns)

    def _is_latest(self, generation_id: int) -> bool:
        with self._request_lock:
            return generation_id == self._latest_generation

    def _has_pending_request(self) -> bool:
        with self._request_lock:
            return self._pending_request_key is not None

    def _update_mood(self, transcript: str) -> None:
        mood = self._client.classify_mood(transcript, set(self._moods), self._config)
        if mood is None:
            return
        with self._state.locked():
            if mood == self._state.mood:
                return
            previous_mood = self._state.mood
            self._state.mood = mood
            self._state.mood_until = time.time() + float(
                self._moods[mood].get("decay", 0.0)
            )
        self._event_log.append(
            "mood_changed", {"from": previous_mood, "to": mood}
        )
        self._state.record_debug(
            "mood_changed", previous_mood=previous_mood, mood=mood
        )


def _memory_limits(config: dict[str, Any]) -> tuple[float, int]:
    settings = config.get("memory", {})
    max_age_seconds = float(settings.get("short_term_minutes", 10)) * 60.0
    max_turns = int(settings.get("short_term_turns", 8))
    return max_age_seconds, max_turns


def _request_key(value: str) -> str:
    normalised = "".join(
        character.lower() if character.isalnum() else " "
        for character in value
    )
    return " ".join(normalised.split()) or "<acknowledgement>"


def _clause_end(text: str) -> int | None:
    """Choose a natural streamed speech boundary without letting buffers grow forever."""
    strong = _first_index(text, _STRONG_BOUNDARIES)
    if strong is not None and strong < _SOFT_CLAUSE_CHARS:
        return strong

    if len(text) >= _SOFT_CLAUSE_CHARS:
        soft_limit = min(_SOFT_CLAUSE_CHARS, strong if strong is not None else len(text) - 1)
        comma = text.rfind(",", _MIN_COMMA_CLAUSE_CHARS - 1, soft_limit + 1)
        if comma >= _MIN_COMMA_CLAUSE_CHARS - 1:
            return comma

    if strong is not None and strong < _HARD_CLAUSE_CHARS:
        return strong

    if len(text) >= _HARD_CLAUSE_CHARS:
        hard_limit = _HARD_CLAUSE_CHARS - 1
        comma = text.rfind(",", _MIN_COMMA_CLAUSE_CHARS - 1, hard_limit + 1)
        if comma >= _MIN_COMMA_CLAUSE_CHARS - 1:
            return comma
        whitespace = max(
            text.rfind(" ", 0, _HARD_CLAUSE_CHARS),
            text.rfind("\t", 0, _HARD_CLAUSE_CHARS),
        )
        if whitespace >= 0:
            return whitespace

    return strong


def _first_index(text: str, characters: str) -> int | None:
    positions = [text.find(character) for character in characters]
    candidates = [position for position in positions if position >= 0]
    return min(candidates) if candidates else None
