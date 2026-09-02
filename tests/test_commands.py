"""Closed command registry safety and execution contract."""

import unittest

from brain.commands import CommandCall, CommandRegistry
from state import State


class CommandRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = State()
        self.registry = CommandRegistry(
            {
                "commands": {
                    "colors": {
                        "blue": [10, 20, 30],
                        "red": [200, 40, 30],
                    }
                }
            },
            self.state,
        )

    def test_catalog_exposes_only_the_executable_set_color_command(self) -> None:
        self.assertEqual(
            self.registry.catalog(),
            {
                "set_color": {
                    "arguments": {
                        "name": {
                            "type": "string",
                            "values": ["blue", "red"],
                        }
                    }
                }
            },
        )

    def test_candidate_gate_requires_control_language_and_registered_value(self) -> None:
        self.assertTrue(self.registry.is_candidate("turn blue"))
        self.assertTrue(self.registry.is_candidate("Could you make yourself red?"))
        self.assertTrue(self.registry.is_candidate("please change your color to blue"))
        self.assertFalse(self.registry.is_candidate("why is the sky blue?"))
        self.assertFalse(self.registry.is_candidate("blue is my favorite color"))
        self.assertFalse(self.registry.is_candidate("turn green"))

    def test_validate_accepts_only_exact_allowlisted_call_shape(self) -> None:
        self.assertEqual(
            self.registry.validate(
                {"name": "set_color", "arguments": {"name": "blue"}}
            ),
            CommandCall("set_color", {"name": "blue"}),
        )

        rejected = (
            {"name": "open_app", "arguments": {"name": "browser"}},
            {"name": "set_color", "arguments": {"name": "green"}},
            {"name": "set_color", "arguments": {"name": [10, 20, 30]}},
            {"name": "set_color", "arguments": {"name": "blue", "extra": "x"}},
            {"name": "set_color", "arguments": "blue"},
            {"name": "set_color", "arguments": {"name": "blue"}, "extra": True},
            ["set_color", "blue"],
            None,
        )
        for payload in rejected:
            with self.subTest(payload=payload):
                self.assertIsNone(self.registry.validate(payload))

    def test_execute_maps_validated_color_name_to_human_authored_rgb(self) -> None:
        result = self.registry.execute(CommandCall("set_color", {"name": "blue"}))

        with self.state.locked():
            self.assertEqual(self.state.color, (10, 20, 30))
        self.assertEqual(result.spoken_response, "Blue.")
        self.assertEqual(
            result.event_payload,
            {"name": "set_color", "arguments": {"name": "blue"}},
        )

    def test_execute_rejects_unvalidated_calls_instead_of_trusting_callers(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.execute(CommandCall("set_color", {"name": "green"}))
        with self.assertRaises(ValueError):
            self.registry.execute(CommandCall("open_app", {"name": "browser"}))

        with self.state.locked():
            self.assertIsNone(self.state.color)

    def test_bad_palette_fails_at_construction_instead_of_exposing_bad_rgb(self) -> None:
        for value in ([1, 2], [0, 0, 999], [0, "1", 2], "#0000ff"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CommandRegistry(
                        {"commands": {"colors": {"blue": value}}},
                        State(),
                    )


if __name__ == "__main__":
    unittest.main()
