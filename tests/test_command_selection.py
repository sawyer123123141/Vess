"""Strict local-model command selection contract."""

import json
import unittest

from brain.llm import OllamaClient


CATALOG = {
    "set_color": {
        "arguments": {
            "name": {"type": "string", "values": ["blue", "red"]}
        }
    }
}


class CommandSelectionTests(unittest.TestCase):
    def test_selector_sends_only_supplied_catalog_as_nonstreaming_json_task(self) -> None:
        requests = []

        def opener(request, *, timeout):
            requests.append((request, timeout))
            return ReadResponse(
                {"response": '{"name":"set_color","arguments":{"name":"blue"}}'}
            )

        client = OllamaClient(opener=opener)
        result = client.select_command(
            "turn blue",
            CATALOG,
            {"llm": {"model": "qwen2.5:7b", "keep_alive": -1}},
        )

        self.assertEqual(
            result,
            {"name": "set_color", "arguments": {"name": "blue"}},
        )
        payload = json.loads(requests[0][0].data)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")
        self.assertIn(json.dumps(CATALOG, sort_keys=True), payload["prompt"])
        self.assertIn("return JSON null", payload["prompt"])
        self.assertIn("instructing Vess itself", payload["prompt"])
        self.assertNotIn("open_app", payload["prompt"])
        self.assertEqual(requests[0][1], 60)

    def test_selector_json_null_means_no_command(self) -> None:
        client = OllamaClient(
            opener=lambda request, *, timeout: ReadResponse({"response": "null"})
        )

        self.assertIsNone(client.select_command("turn blue", CATALOG, {}))

    def test_selector_malformed_or_wrong_shape_output_fails_closed(self) -> None:
        for response in ("not json", '["set_color", "blue"]', '"set_color"', "42"):
            with self.subTest(response=response):
                client = OllamaClient(
                    opener=lambda request, *, timeout, response=response: ReadResponse(
                        {"response": response}
                    )
                )
                self.assertIsNone(client.select_command("turn blue", CATALOG, {}))

    def test_selector_does_not_validate_command_authority_itself(self) -> None:
        payload = {"name": "imaginary_command", "arguments": {"path": "anything"}}
        client = OllamaClient(
            opener=lambda request, *, timeout: ReadResponse(
                {"response": json.dumps(payload)}
            )
        )

        self.assertEqual(client.select_command("turn blue", CATALOG, {}), payload)


class ReadResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
