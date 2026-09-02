"""Closed command definitions, validation, and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_CONTROL_VERBS = {"change", "make", "set", "turn"}


@dataclass(frozen=True)
class CommandCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class CommandResult:
    spoken_response: str
    event_payload: dict[str, object]


class CommandRegistry:
    """Expose only human-declared actions and reject everything else."""

    def __init__(self, config: dict[str, Any], state: Any) -> None:
        colors = config.get("commands", {}).get("colors", {})
        if not isinstance(colors, dict) or not colors:
            raise ValueError("commands.colors must be a non-empty object")

        palette: dict[str, tuple[int, int, int]] = {}
        for raw_name, raw_color in colors.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("command color names must be non-empty strings")
            name = raw_name.strip().lower()
            if name != raw_name or not name.replace("_", "").isalnum():
                raise ValueError("command color names must be normalized identifiers")
            if (
                not isinstance(raw_color, list)
                or len(raw_color) != 3
                or any(type(channel) is not int for channel in raw_color)
                or any(channel < 0 or channel > 255 for channel in raw_color)
            ):
                raise ValueError(f"invalid RGB value for command color {name!r}")
            palette[name] = tuple(raw_color)

        self._colors = palette
        self._state = state

    def catalog(self) -> dict[str, object]:
        """Return the JSON-safe commands the model is actually allowed to select."""
        return {
            "set_color": {
                "arguments": {
                    "name": {
                        "type": "string",
                        "values": list(self._colors),
                    }
                }
            }
        }

    def is_candidate(self, text: str) -> bool:
        """Cheaply reject ordinary conversation before any command-model call."""
        tokens = set(_tokens(text))
        return bool(tokens & _CONTROL_VERBS) and bool(tokens & self._colors.keys())

    def validate(self, payload: object) -> CommandCall | None:
        """Accept only the exact declared set_color shape and allowlisted value."""
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            return None
        if payload.get("name") != "set_color":
            return None
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict) or set(arguments) != {"name"}:
            return None
        color_name = arguments.get("name")
        if not isinstance(color_name, str) or color_name not in self._colors:
            return None
        return CommandCall("set_color", {"name": color_name})

    def execute(self, call: CommandCall) -> CommandResult:
        """Execute one already-declared command, revalidating the public input."""
        validated = self.validate(
            {"name": call.name, "arguments": dict(call.arguments)}
        )
        if validated is None:
            raise ValueError("command call is not allowed by the registry")

        color_name = str(validated.arguments["name"])
        with self._state.locked():
            self._state.color = self._colors[color_name]

        arguments = {"name": color_name}
        return CommandResult(
            spoken_response=f"{color_name.capitalize()}.",
            event_payload={"name": "set_color", "arguments": arguments},
        )


def _tokens(value: str) -> list[str]:
    normalized = "".join(
        character.lower() if character.isalnum() or character == "_" else " "
        for character in value
    )
    return normalized.split()
