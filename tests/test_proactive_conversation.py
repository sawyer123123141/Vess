"""Proactive speech must reuse delivery without pretending an event was user speech."""

import threading
import time
import unittest

from brain.llm import build_prompt
from brain.memory import append_conversation_turn
from brain.proactive import ProactiveConversationWorker, build_proactive_prompt
from performance import PerformanceCue
from state import State


CONFIG = {
    "personas": {"friendly": "Warm."},
    "memory": {"short_term_minutes": 10, "short_term_turns": 8},
}
MOODS = {"neutral": {"prompt": ""}}
PERFORMANCES = {
    "neutral": {"intensity": 0.0, "shape": {}, "movement": {}},
    "playful": {"intensity": 0.65, "shape": {}, "movement": {}},
}
CONTEXT = "A person just returned after being absent for about 4 hours."


class ProactiveConversationTests(unittest.TestCase):
    def test_proactive_prompt_is_observation_not_fake_user_request(self) -> None:
        state = State(person_present=True)

        prompt = build_proactive_prompt(
            CONFIG,
            MOODS,
            state,
            "returned_after_absence",
            CONTEXT,
            performances=PERFORMANCES,
        )

        self.assertIn("Proactive system observation:", prompt)
        self.assertIn(CONTEXT, prompt)
        self.assertNotIn(f"Current request:\n{CONTEXT}", prompt)
        self.assertIn("one short sentence", prompt.lower())
        self.assertIn("do not ask a generic question", prompt.lower())
        self.assertIn("do not infer", prompt.lower())

    def test_prompt_history_keeps_assistant_only_turn_without_fake_trigger_user_text(self) -> None:
        state = State()
        append_conversation_turn(
            state,
            "",
            "Back in the room, I see.",
            timestamp=time.time(),
            max_age_seconds=600.0,
            max_turns=8,
        )

        prompt = build_prompt(CONFIG, MOODS, state, "I know.")

        self.assertIn("Vess: Back in the room, I see.", prompt)
        self.assertNotIn(f"User: {CONTEXT}", prompt)
        self.assertIn("Current request:\nI know.", prompt)

    def test_proactive_submission_refuses_to_replace_pending_user_request(self) -> None:
        worker, state, voice, client, memory = make_worker()
        worker.submit("real user request")

        accepted = worker.submit_proactive("returned_after_absence", CONTEXT)

        self.assertFalse(accepted)
        self.assertEqual(voice.generations, [1])
        self.assertEqual(state.last_interaction > 0.0, True)
        self.assertEqual(memory.remembered, [])

    def test_real_user_request_can_supersede_pending_proactive_generation(self) -> None:
        worker, state, voice, client, memory = make_worker()
        self.assertTrue(worker.submit_proactive("returned_after_absence", CONTEXT))

        worker.submit("real user request")
        worker.start()
        worker.close()

        self.assertEqual(voice.generations, [1, 2])
        self.assertTrue(client.prompts)
        self.assertIn("Current request:\nreal user request", client.prompts[-1])
        self.assertNotIn("Proactive system observation:", client.prompts[-1])

    def test_proactive_generation_skips_command_selection_and_mood_classification(self) -> None:
        worker, state, voice, client, memory = make_worker(auto_deliver=True)

        self.assertTrue(worker.submit_proactive("returned_after_absence", CONTEXT))
        worker.start()
        worker.close()

        self.assertEqual(client.command_calls, 0)
        self.assertEqual(client.mood_calls, 0)
        self.assertEqual(len(client.prompts), 1)
        self.assertIn("Proactive system observation:", client.prompts[0])

    def test_delivered_proactive_line_is_assistant_only_memory_and_not_durable_fact(self) -> None:
        worker, state, voice, client, memory = make_worker(auto_deliver=True)

        self.assertTrue(worker.submit_proactive("quiet_interaction", "Quiet for about 30 minutes."))
        worker.start()
        worker.close()

        self.assertEqual(
            [(turn.user, turn.assistant) for turn in state.conversation_turns],
            [("", "Nice quiet stretch in here.")],
        )
        self.assertEqual(memory.remembered, [])

    def test_generated_but_undelivered_proactive_line_is_not_remembered(self) -> None:
        worker, state, voice, client, memory = make_worker(auto_deliver=False)

        self.assertTrue(worker.submit_proactive("quiet_interaction", "Quiet for about 30 minutes."))
        worker.start()
        worker.close()

        self.assertEqual(state.conversation_turns, [])
        self.assertEqual(memory.remembered, [])

    def test_active_proactive_work_rejects_second_proactive_submission(self) -> None:
        client = BlockingClient()
        worker, state, voice, _, memory = make_worker(client=client)
        worker.start()
        self.assertTrue(worker.submit_proactive("quiet_interaction", "Quiet for about 30 minutes."))
        self.assertTrue(client.started.wait(timeout=1.0))

        second = worker.submit_proactive("returned_after_absence", CONTEXT)

        self.assertFalse(second)
        client.release.set()
        worker.close()
        self.assertEqual(voice.generations, [1])


def make_worker(*, auto_deliver: bool = False, client=None):
    state = State()
    voice = RecordingVoice(auto_deliver=auto_deliver)
    log = RecordingLog()
    memory = RecordingMemory()
    actual_client = client or RecordingClient()
    worker = ProactiveConversationWorker(
        CONFIG,
        MOODS,
        state,
        log,
        actual_client,
        voice,
        performances=PERFORMANCES,
        durable_memory=memory,
        command_registry=RejectingRegistry(),
    )
    voice.delivery_callback = worker.handle_delivery
    return worker, state, voice, actual_client, memory


class RecordingClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.command_calls = 0
        self.mood_calls = 0

    def stream(self, prompt, config):
        self.prompts.append(prompt)
        if "Proactive system observation:" in prompt:
            return ["Nice quiet stretch in here."]
        return ["User response."]

    def select_command(self, *args):
        self.command_calls += 1
        return None

    def classify_mood(self, *args):
        self.mood_calls += 1
        return None


class BlockingClient(RecordingClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def stream(self, prompt, config):
        self.prompts.append(prompt)
        self.started.set()
        self.release.wait(timeout=2.0)
        return ["Nice quiet stretch in here."]


class RejectingRegistry:
    def is_candidate(self, text):
        return True

    def catalog(self):
        return {"set_color": {}}

    def validate(self, payload):
        return None


class RecordingMemory:
    def __init__(self) -> None:
        self.remembered: list[str] = []

    def relevant_facts(self, query, *, limit):
        return []

    def remember(self, text):
        self.remembered.append(text)


class RecordingVoice:
    def __init__(self, *, auto_deliver: bool) -> None:
        self.auto_deliver = auto_deliver
        self.generations: list[int] = []
        self.delivery_callback = None

    def begin_generation(self, generation_id):
        self.generations.append(generation_id)

    def enqueue(self, text, generation_id=None, performance: PerformanceCue | None = None):
        if self.auto_deliver and self.delivery_callback is not None:
            self.delivery_callback(
                "clause_started",
                {"generation_id": generation_id, "text": text},
            )
            self.delivery_callback(
                "clause_completed",
                {"generation_id": generation_id, "text": text},
            )

    def finish_generation(self, generation_id):
        if self.auto_deliver and self.delivery_callback is not None:
            self.delivery_callback(
                "generation_playback_drained",
                {"generation_id": generation_id},
            )

    def enqueue_acknowledgement(self, generation_id=None):
        pass


class RecordingLog:
    def __init__(self) -> None:
        self.events = []

    def append(self, event_type, payload):
        self.events.append((event_type, payload))


if __name__ == "__main__":
    unittest.main()
