"""Cancellable and resumable audio playback for barge-in."""

from __future__ import annotations

import threading
import unittest

import numpy as np

from output.audio_player import SoundDeviceAudioPlayer


class FakeOutputBackend:
    """Render deterministic frames and block the first play after two frames."""

    def __init__(self, *, block_first_play: bool = False) -> None:
        self.block_first_play = block_first_play
        self.started = threading.Event()
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.abort_requested = threading.Event()
        self.play_calls = 0
        self.frames_written = 0

    def play(self, audio: np.ndarray, sample_rate: int, on_render) -> int:
        self.play_calls += 1
        call_number = self.play_calls
        self.abort_requested.clear()
        completed = 0
        self.started.set()

        for sample in np.asarray(audio, dtype=np.float32).reshape(-1):
            if self.abort_requested.is_set():
                break
            frame = np.asarray([sample], dtype=np.float32)
            on_render(frame, sample_rate)
            completed += 1
            self.frames_written += 1
            if self.block_first_play and call_number == 1 and completed == 2:
                self.blocked.set()
                self.release.wait(timeout=1.0)

        return completed

    def abort(self) -> None:
        self.abort_requested.set()
        self.release.set()


class AudioPlayerTests(unittest.TestCase):
    def test_completed_playback_reports_all_frames(self) -> None:
        backend = FakeOutputBackend()
        player = SoundDeviceAudioPlayer(backend=backend)

        receipt = player.play(np.arange(4, dtype=np.float32), 4, 7)

        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.generation_id, 7)
        self.assertEqual(receipt.frames_started, 0)
        self.assertEqual(receipt.frames_completed, 4)
        self.assertEqual(receipt.total_frames, 4)
        self.assertEqual(receipt.sample_rate, 4)

    def test_pause_saves_cursor_and_resume_finishes_same_generation(self) -> None:
        backend = FakeOutputBackend(block_first_play=True)
        player = SoundDeviceAudioPlayer(backend=backend)
        original = np.arange(6, dtype=np.float32)
        playback_result: list[object] = []
        thread = threading.Thread(
            target=lambda: playback_result.append(player.play(original, 6, 3)),
        )
        thread.start()
        self.assertTrue(backend.blocked.wait(timeout=1.0))

        paused = player.pause_for_interruption()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(paused)
        assert paused is not None
        self.assertEqual(paused.status, "paused")
        self.assertEqual(paused.generation_id, 3)
        self.assertEqual(paused.frames_completed, 2)
        self.assertEqual(paused.total_frames, 6)
        self.assertEqual(len(playback_result), 1)
        self.assertEqual(playback_result[0], paused)

        resumed = player.resume()

        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.generation_id, 3)
        self.assertEqual(resumed.frames_started, 2)
        self.assertEqual(resumed.frames_completed, 6)
        self.assertEqual(resumed.total_frames, 6)
        self.assertEqual(backend.play_calls, 2)

    def test_discard_paused_prevents_resume(self) -> None:
        backend = FakeOutputBackend(block_first_play=True)
        player = SoundDeviceAudioPlayer(backend=backend)
        thread = threading.Thread(
            target=lambda: player.play(np.arange(5, dtype=np.float32), 5, 9),
        )
        thread.start()
        self.assertTrue(backend.blocked.wait(timeout=1.0))
        self.assertIsNotNone(player.pause_for_interruption())
        thread.join(timeout=1.0)

        player.discard_paused()

        with self.assertRaises(RuntimeError):
            player.resume()

    def test_render_callback_receives_only_physically_rendered_frames(self) -> None:
        backend = FakeOutputBackend(block_first_play=True)
        rendered: list[float] = []
        player = SoundDeviceAudioPlayer(
            backend=backend,
            render_callback=lambda block: rendered.extend(block.samples.tolist()),
        )
        thread = threading.Thread(
            target=lambda: player.play(np.arange(5, dtype=np.float32), 5, 4),
        )
        thread.start()
        self.assertTrue(backend.blocked.wait(timeout=1.0))

        player.pause_for_interruption()
        thread.join(timeout=1.0)

        self.assertEqual(rendered, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
