"""Vess.

Steps 1-2: build the state, run the detector in its own thread, render the
face from what it finds. The render loop never waits on perception.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from brain.llm import ConversationWorker, OllamaClient
from brain.memory import EventLog, FactMemory
from brain.turn_coordinator import TurnCoordinator
from control.web import WebServer
from output.animator import FaceAnimator
from output.audio_player import SoundDeviceAudioPlayer
from output.display import Display, PreviewWindow
from output.voice import VoiceOutput
from perception import camera as camera_module
from perception.audio import AudioLoop
from perception.audio_preprocess import PassthroughCapturePreprocessor
from perception.detector import Detector, run_detection_loop
from perception.interruption import InterruptionDetector
from performance import load_performance_definitions
from state import State

ROOT = Path(__file__).resolve().parent
FRAME_TIME = 1.0 / 30.0

FAKE_PERSON_POSITIONS: tuple[tuple[float, float] | None, ...] = (
    None,
    (0.12, 0.50),
    (0.50, 0.25),
    (0.88, 0.50),
    (0.50, 0.78),
)


@dataclass
class VoiceRuntime:
    preprocessor: Any
    player: Any
    voice: Any
    conversation: Any
    coordinator: Any
    audio: Any
    durable_memory: Any | None = None

    def close(self) -> None:
        """Shut down producers before draining the workers they can affect."""
        self.audio.close()
        self.coordinator.close()
        self.conversation.close()
        if self.durable_memory is not None:
            self.durable_memory.close()
        self.voice.close()


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _load_performances() -> dict[str, dict[str, object]]:
    return load_performance_definitions(_load("performance.json"))


def _build_fact_memory(
    config: dict[str, Any],
    client: Any,
    *,
    path: Path = ROOT / "vess.db",
    factory: Any = FactMemory,
) -> Any:
    """Bind durable extraction to the one local Ollama client and config."""
    return factory(
        path,
        lambda text, known_keys: client.extract_facts(text, known_keys, config),
    )


def _start_perception(config: dict, state: State,
                      stop: threading.Event) -> tuple[threading.Thread | None, object]:
    """Bring up tier 1, or say why not and carry on without it."""
    try:
        camera = camera_module.open_camera(config)
    except RuntimeError as error:
        print(f"perception off: {error}")
        return None, None

    try:
        detector = Detector(config["detector"]["model"],
                            float(config["detector"].get("confidence", 0.4)))
    except Exception as error:
        print(f"perception off: cannot load detector -- {error}")
        camera.close()
        return None, None

    thread = threading.Thread(
        target=run_detection_loop,
        args=(state, camera, detector, float(config["detector"]["fps"]), stop),
        name="detector",
        daemon=True,
    )
    thread.start()
    source = config.get("camera", {}).get("source", "camera")
    mirrored = "mirrored" if config.get("camera", {}).get("mirror", True) else "not mirrored"
    print(f"perception on: {source}, {mirrored}, "
          f"{config['detector']['fps']}fps on cpu")
    return thread, camera


def _build_display(
    config: dict,
    state: State,
    event_log: EventLog | None = None,
) -> tuple[Display, WebServer | None]:
    """Build every enabled output target around the one rendered frame."""
    display_config = config.get("display", {})
    web_config = config.get("web", {})
    targets = []
    web_server = None

    if web_config.get("enabled", True):
        web_server = WebServer(
            state,
            int(web_config.get("port", 8080)),
            event_log,
        )
        targets.append(web_server.preview)
    if display_config.get("cv2_enabled", False):
        targets.append(PreviewWindow(scale=display_config.get("preview_scale", 8)))
    if not targets:
        raise RuntimeError("no display targets enabled")
    return Display(targets), web_server


def _build_voice_runtime(
    config: dict[str, Any],
    moods: dict[str, dict[str, Any]],
    performances: dict[str, dict[str, object]],
    state: State,
    event_log: Any,
    *,
    client: Any | None = None,
    durable_memory: Any | None = None,
    preprocessor: Any | None = None,
    interruption_detector: Any | None = None,
    player_factory: Any = SoundDeviceAudioPlayer,
    voice_factory: Any = VoiceOutput,
    conversation_factory: Any = ConversationWorker,
    coordinator_factory: Any = TurnCoordinator,
    audio_factory: Any = AudioLoop,
) -> VoiceRuntime:
    """Construct the complete voice graph without starting hardware or workers."""
    audio_settings = config.get("audio", {})
    barge_in = config.get("barge_in", {})

    if preprocessor is None:
        preprocessor_name = str(barge_in.get("preprocessor", "passthrough"))
        if preprocessor_name != "passthrough":
            raise RuntimeError(f"unsupported barge-in preprocessor: {preprocessor_name}")
        preprocessor = PassthroughCapturePreprocessor()

    if interruption_detector is None:
        interruption_detector = InterruptionDetector(
            int(audio_settings.get("sample_rate", 16_000)),
            float(audio_settings.get("vad_threshold", 0.015)),
            float(barge_in.get("pause_after_speech_seconds", 0.25)),
        )

    player = player_factory(render_callback=preprocessor.push_render_reference)
    conversation_holder: dict[str, Any] = {}

    def on_delivery(event_type: str, payload: dict[str, object]) -> None:
        conversation = conversation_holder.get("conversation")
        if conversation is not None:
            conversation.handle_delivery(event_type, payload)

    def on_synthesis_timing(payload: dict[str, object]) -> None:
        conversation = conversation_holder.get("conversation")
        if conversation is not None:
            conversation.handle_synthesis_timing(payload)

    voice = voice_factory(
        config,
        state,
        event_log,
        player=player,
        on_delivery=on_delivery,
        on_synthesis_timing=on_synthesis_timing,
    )
    conversation_kwargs: dict[str, Any] = {"performances": performances}
    if durable_memory is not None:
        conversation_kwargs["durable_memory"] = durable_memory
    conversation = conversation_factory(
        config,
        moods,
        state,
        event_log,
        client if client is not None else OllamaClient(),
        voice,
        **conversation_kwargs,
    )
    conversation_holder["conversation"] = conversation

    coordinator = coordinator_factory(
        state,
        event_log,
        voice,
        conversation,
        conversation.submit,
        false_timeout_seconds=float(
            barge_in.get("false_interruption_timeout_seconds", 2.0)
        ),
        decision_watchdog_seconds=float(
            barge_in.get("max_interruption_decision_seconds", 5.0)
        ),
        timed_transcript_submit=conversation.submit_with_timing,
    )
    audio = audio_factory(
        config,
        state,
        event_log,
        conversation.submit,
        on_timed_request=conversation.submit_with_timing,
        preprocessor=preprocessor,
        interruption_detector=interruption_detector,
        turn_coordinator=coordinator,
    )
    return VoiceRuntime(
        preprocessor=preprocessor,
        player=player,
        voice=voice,
        conversation=conversation,
        coordinator=coordinator,
        audio=audio,
        durable_memory=durable_memory,
    )


def _open_browser(port: int) -> None:
    webbrowser.open(f"http://127.0.0.1:{port}")


def _expire_mood(state: State, event_log: object, now: float) -> None:
    """Make expiry an explicit state transition and retain it in history."""
    transition = state.expire_mood(now)
    if transition is None:
        return
    previous_mood, _ = transition
    event_log.append("mood_changed", {"from": previous_mood, "to": "neutral"})


def main() -> None:
    config = _load("config.json")
    moods = _load("moods.json")
    performances = _load_performances()
    mood_names = list(moods)

    event_log = EventLog(ROOT / "vess.db")
    event_log.append("session_started", {})
    state = State()
    animator = FaceAnimator(moods, performances)
    display, web_server = _build_display(config, state, event_log)
    client = OllamaClient()
    try:
        durable_memory = _build_fact_memory(config, client)
    except Exception as error:
        durable_memory = None
        print(f"durable memory off: {error}")

    try:
        runtime = _build_voice_runtime(
            config,
            moods,
            performances,
            state,
            event_log,
            client=client,
            durable_memory=durable_memory,
        )
    except Exception:
        if durable_memory is not None:
            durable_memory.close()
        raise

    stop = threading.Event()
    thread, camera = _start_perception(config, state, stop)
    if web_server is not None:
        web_server.start()
        if config.get("web", {}).get("open_browser_on_start", True):
            threading.Thread(
                target=_open_browser,
                args=(int(config["web"].get("port", 8080)),),
                name="browser",
                daemon=True,
            ).start()

    runtime.voice.start()
    runtime.voice.prepare_acknowledgement()
    runtime.conversation.start()
    try:
        runtime.audio.start()
    except RuntimeError as error:
        print(f"voice input off: {error}")

    print("keys:")
    for index, name in enumerate(mood_names, start=1):
        print(f"  {index}  mood -> {name}")
    print("  t  cycle a fake person_pos")
    print("  k  toggle thinking   (face drifts up and away)")
    print("  l  toggle listening  (face settles and leans in)")
    print("  q  quit")

    fake = 0
    last = time.perf_counter()
    try:
        while display.is_open():
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now

            _expire_mood(state, event_log, time.time())
            display.show(animator.tick(state, dt))

            key = display.poll_key()
            if key in (27, ord("q")):
                break
            if ord("1") <= key < ord("1") + len(mood_names):
                mood = mood_names[key - ord("1")]
                with state.locked():
                    state.mood = mood
                    state.mood_until = 0.0
                print(f"mood -> {mood}")
            elif key == ord("t"):
                fake = (fake + 1) % len(FAKE_PERSON_POSITIONS)
                position = FAKE_PERSON_POSITIONS[fake]
                with state.locked():
                    state.person_pos = position
                    state.person_present = position is not None
                print(f"person_pos -> {position}")
            elif key in (ord("k"), ord("l")):
                field = "thinking" if key == ord("k") else "listening"
                with state.locked():
                    value = not getattr(state, field)
                    setattr(state, field, value)
                print(f"{field} -> {value}")

            time.sleep(max(0.0, FRAME_TIME - (time.perf_counter() - now)))
    finally:
        runtime.close()
        stop.set()
        if thread is not None:
            thread.join(timeout=2.0)
        if camera is not None:
            camera.close()
        if web_server is not None:
            web_server.close()
        display.close()
        event_log.close()


if __name__ == "__main__":
    main()
