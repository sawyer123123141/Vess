"""Local Ollama prompting and streamed clause handling."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from urllib import request as url_request


def build_prompt(config: dict[str, Any], state: Any, request: str) -> str:
    """Build the cache-friendly stable instruction before live state."""
    with state.locked():
        persona = state.persona
        mood = state.mood
        person_present = state.person_present
        objects = list(state.objects)

    persona_instruction = config.get("personas", {}).get(persona, "")
    presence = "someone is present" if person_present else "the room is empty"
    seen_objects = ", ".join(objects) if objects else "none"
    return (
        "You are Vess. You live as a small expressive face on a wall. "
        "Reply naturally and concisely, in at most two sentences.\n\n"
        "Current state:\n"
        f"Persona: {persona}. {persona_instruction}\n"
        f"Mood: {mood}\n"
        f"Room: {presence}\n"
        f"Visible objects: {seen_objects}\n"
        f"Request: {request}"
    )


def split_clauses(chunks: Iterable[str]) -> Iterator[str]:
    """Yield complete speech clauses as Ollama response chunks arrive."""
    pending = ""
    for chunk in chunks:
        pending += chunk
        while True:
            end = _clause_end(pending)
            if end is None:
                break
            clause = pending[: end + 1].strip()
            pending = pending[end + 1 :]
            if clause:
                yield clause
    if pending.strip():
        yield pending.strip()


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
    ) -> None:
        self._config = config
        self._moods = moods
        self._state = state
        self._event_log = event_log
        self._client = client
        self._voice = voice
        self._requests: queue.Queue[tuple[int, str] | None] = queue.Queue(maxsize=1)
        self._request_lock = threading.Lock()
        self._next_generation = 0
        self._latest_generation = 0
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
                _, replaced_request = pending

            self._pending_request_key = key
            self._voice.begin_generation(generation_id)
            self._requests.put_nowait((generation_id, request))

        self._state.update_debug(
            conversation_queue=self._requests.qsize(),
            active_generation=generation_id,
        )
        if replaced_request is not None:
            self._state.record_debug(
                "pending_request_replaced",
                replaced_request=replaced_request,
                request=request,
            )

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

        with self._state.locked():
            self._state.thinking = True
        llm_started_at = time.perf_counter()
        self._state.record_debug(
            "llm_started",
            request=user_request,
            generation_id=generation_id,
        )
        try:
            prompt = build_prompt(self._config, self._state, user_request)
            first_clause = True
            for clause in split_clauses(self._client.stream(prompt, self._config)):
                if not self._is_latest(generation_id):
                    self._state.record_debug(
                        "stale_response_cancelled",
                        generation_id=generation_id,
                    )
                    return
                with self._state.locked():
                    self._state.thinking = False
                if first_clause:
                    latency_ms = (time.perf_counter() - llm_started_at) * 1000.0
                    rounded = round(latency_ms, 1)
                    self._state.update_debug(llm_first_clause_ms=rounded)
                    self._state.record_debug(
                        "llm_first_clause",
                        clause=clause,
                        latency_ms=rounded,
                        generation_id=generation_id,
                    )
                    first_clause = False
                self._voice.enqueue(clause, generation_id=generation_id)

            if not self._is_latest(generation_id):
                self._state.record_debug(
                    "stale_response_cancelled",
                    generation_id=generation_id,
                )
                return

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


def _request_key(value: str) -> str:
    normalised = "".join(
        character.lower() if character.isalnum() else " "
        for character in value
    )
    return " ".join(normalised.split()) or "<acknowledgement>"


def _clause_end(text: str) -> int | None:
    ends = [text.find(character) for character in ",.!?\n"]
    candidates = [end for end in ends if end >= 0]
    return min(candidates) if candidates else None
