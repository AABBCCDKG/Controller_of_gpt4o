from contextlib import redirect_stderr
import io
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import echogen


class TestConfiguration(unittest.TestCase):
    def test_load_api_key_reads_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "  test-key  "}, clear=True):
            self.assertEqual(echogen.load_api_key(), "test-key")

    def test_load_api_key_rejects_missing_value(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(echogen.ConfigError, "OPENAI_API_KEY"):
                echogen.load_api_key()


class TestSimulationParser(unittest.TestCase):
    def test_parse_functions_accepts_only_supported_literal_calls(self):
        response = """```python
add_curved_floor(space, [(0, 10), (20, 30)], 4, 0.5)
add_ball(space, (100, 50), 25, 1, (200, -20))
add_polygon(space, (200, 300), [(0, 0), (50, 0), (0, 50)], 10, (0, 0))
```"""

        calls = echogen.parse_functions(response)

        self.assertEqual(
            [call.name for call in calls],
            ["add_curved_floor", "add_ball", "add_polygon"],
        )
        self.assertEqual(calls[1].args[0], (100.0, 50.0))
        self.assertEqual(calls[1].args[3], (200.0, -20.0))

    def test_parse_functions_rejects_code_injection_without_running_it(self):
        response = (
            "add_ball(space, (100, 50), 25, 1, (0, 0))\n"
            "__import__('pathlib').Path('owned').write_text('unsafe')"
        )

        with self.assertRaises(echogen.SimulationParseError):
            echogen.parse_functions(response)

        self.assertFalse(os.path.exists("owned"))

    def test_parse_functions_rejects_unknown_function_and_expressions(self):
        invalid_responses = [
            "delete_everything(space, '/tmp')",
            "add_ball(space, (1 + 2, 3), 4, 1, (0, 0))",
            "add_ball(other_space, (1, 2), 4, 1, (0, 0))",
            "add_ball(space, (1, 2), -4, 1, (0, 0))",
        ]

        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(echogen.SimulationParseError):
                    echogen.parse_functions(response)

    def test_parse_functions_rejects_integer_too_large_for_float(self):
        huge_integer = "1" + ("0" * 1000)
        response = f"add_ball(space, (1, 2), {huge_integer}, 1, (0, 0))"

        with self.assertRaisesRegex(echogen.SimulationParseError, "finite number"):
            echogen.parse_functions(response)

    def test_render_round_trips_through_safe_parser(self):
        calls = [
            echogen.SimulationCall(
                "add_ball",
                ((10.0, 20.0), 5.0, 2.0, (-1.5, 3.25)),
            )
        ]

        rendered = echogen.render_simulation_calls(calls)

        self.assertEqual(echogen.parse_functions(rendered), calls)


class TestSafeExecution(unittest.TestCase):
    def test_execute_simulation_calls_dispatches_through_allowlist(self):
        recorded = []

        def record_ball(space, *args):
            recorded.append((space, args))

        calls = [
            echogen.SimulationCall(
                "add_ball",
                ((10.0, 20.0), 5.0, 2.0, (-1.5, 3.25)),
            )
        ]

        echogen.execute_simulation_calls(
            calls,
            space="test-space",
            handlers={"add_ball": record_ball},
        )

        self.assertEqual(
            recorded,
            [("test-space", ((10.0, 20.0), 5.0, 2.0, (-1.5, 3.25)))],
        )

    def test_execute_simulation_calls_rejects_missing_handler(self):
        calls = [
            echogen.SimulationCall(
                "add_ball",
                ((10.0, 20.0), 5.0, 2.0, (0.0, 0.0)),
            )
        ]

        with self.assertRaisesRegex(echogen.SimulationExecutionError, "add_ball"):
            echogen.execute_simulation_calls(calls, object(), handlers={})


class TestLLMClient(unittest.TestCase):
    def test_call_chatgpt_uses_timeout_and_validates_response(self):
        captured = {}

        def fake_transport(url, headers, payload, timeout):
            captured.update(
                url=url,
                headers=headers,
                payload=payload,
                timeout=timeout,
            )
            return {
                "choices": [
                    {
                        "message": {
                            "content": "add_ball(space, (1, 2), 3, 4, (5, 6))"
                        }
                    }
                ]
            }

        result = echogen.call_chatgpt(
            "describe the image",
            encoded_image="aW1hZ2U=",
            api_key="secret",
            model="test-model",
            timeout=7.5,
            transport=fake_transport,
        )

        self.assertEqual(result, "add_ball(space, (1, 2), 3, 4, (5, 6))")
        self.assertEqual(captured["timeout"], 7.5)
        self.assertEqual(captured["payload"]["model"], "test-model")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")

    def test_call_chatgpt_rejects_non_finite_timeout_before_transport(self):
        for timeout in (math.nan, math.inf, -math.inf):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(echogen.ConfigError, "finite"):
                    echogen.call_chatgpt(
                        "prompt",
                        encoded_image="aW1hZ2U=",
                        api_key="secret",
                        timeout=timeout,
                        transport=lambda *_: self.fail("transport must not run"),
                    )

    def test_cli_parser_rejects_non_finite_timeout(self):
        parser = echogen.build_argument_parser()

        for timeout in ("nan", "inf", "-inf"):
            with self.subTest(timeout=timeout):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args([f"--timeout={timeout}"])

    def test_call_chatgpt_rejects_malformed_response(self):
        def fake_transport(url, headers, payload, timeout):
            return {"error": {"message": "bad request"}}

        with self.assertRaisesRegex(echogen.LLMResponseError, "choices"):
            echogen.call_chatgpt(
                "prompt",
                encoded_image="aW1hZ2U=",
                api_key="secret",
                transport=fake_transport,
            )


class TestConsensus(unittest.TestCase):
    def test_extract_numbers_preserves_negative_decimals(self):
        self.assertEqual(
            echogen.extract_numbers_from_string("(-12.5, 3, +0.25)"),
            [-12.5, 3.0, 0.25],
        )

    def test_find_similar_and_average_ignores_missing_non_finite_values(self):
        result = echogen.find_similar_and_average(
            [None, 10.0, 11.0, math.inf, 1000.0]
        )

        self.assertEqual(result, 10.5)

    def test_numeric_helpers_ignore_overflowing_values(self):
        huge_integer = 10**1000

        self.assertEqual(
            echogen.find_similar_and_average([huge_integer, 10.0, 12.0]),
            11.0,
        )
        self.assertEqual(
            echogen.extract_numbers_from_string(f"{huge_integer} 5.5 1e9999"),
            [5.5],
        )

    def test_aggregate_simulation_responses_is_robust_to_one_invalid_sample(self):
        responses = [
            "add_ball(space, (-10, 20), 5, 1, (1.5, -2.5))",
            "add_ball(space, (-12, 22), 7, 1, (2.5, -4.5))",
            "__import__('os').system('echo unsafe')",
        ]

        calls = echogen.aggregate_simulation_responses(responses, min_samples=2)

        self.assertEqual(
            calls,
            [
                echogen.SimulationCall(
                    "add_ball",
                    ((-11.0, 21.0), 6.0, 1.0, (2.0, -3.5)),
                )
            ],
        )

    def test_aggregate_simulation_responses_matches_reordered_balls(self):
        responses = [
            "\n".join(
                [
                    "add_ball(space, (10, 20), 5, 1, (1, 0))",
                    "add_ball(space, (200, 220), 15, 2, (-1, 0))",
                ]
            ),
            "\n".join(
                [
                    "add_ball(space, (202, 218), 14, 2, (-2, 0))",
                    "add_ball(space, (12, 22), 6, 1, (2, 0))",
                ]
            ),
            "\n".join(
                [
                    "add_ball(space, (198, 222), 16, 2, (-1, 0))",
                    "add_ball(space, (8, 18), 4, 1, (1, 0))",
                ]
            ),
        ]

        calls = echogen.aggregate_simulation_responses(responses, min_samples=2)

        self.assertEqual(
            calls,
            [
                echogen.SimulationCall(
                    "add_ball",
                    ((10.0, 20.0), 5.0, 1.0, (1.0, 0.0)),
                ),
                echogen.SimulationCall(
                    "add_ball",
                    ((200.0, 220.0), 15.0, 2.0, (-1.0, 0.0)),
                ),
            ],
        )

    def test_aggregate_simulation_responses_rejects_ambiguous_object_matching(self):
        responses = [
            "\n".join(
                [
                    "add_ball(space, (-1, 0), 5, 1, (0, 0))",
                    "add_ball(space, (1, 0), 5, 1, (0, 0))",
                ]
            ),
            "\n".join(
                [
                    "add_ball(space, (0, 0.1), 5, 1, (0, 0))",
                    "add_ball(space, (0, -0.1), 5, 1, (0, 0))",
                ]
            ),
        ]

        with self.assertRaisesRegex(echogen.ConsensusError, "ambiguous"):
            echogen.aggregate_simulation_responses(responses, min_samples=2)

    def test_aggregate_simulation_responses_rejects_unmatched_objects(self):
        responses = [
            "\n".join(
                [
                    "add_ball(space, (10, 10), 5, 1, (0, 0))",
                    "add_ball(space, (100, 100), 5, 1, (0, 0))",
                ]
            ),
            "\n".join(
                [
                    "add_ball(space, (10000, 10000), 5, 1, (0, 0))",
                    "add_ball(space, (20000, 20000), 5, 1, (0, 0))",
                ]
            ),
        ]

        with self.assertRaisesRegex(echogen.ConsensusError, "reliably match"):
            echogen.aggregate_simulation_responses(responses, min_samples=2)

    def test_aggregate_simulation_responses_rejects_object_count_mismatch(self):
        responses = [
            "\n".join(
                [
                    "add_ball(space, (10, 10), 5, 1, (0, 0))",
                    "add_ball(space, (100, 100), 5, 1, (0, 0))",
                ]
            ),
            "add_ball(space, (11, 11), 5, 1, (0, 0))",
        ]

        with self.assertRaisesRegex(echogen.ConsensusError, "matching valid"):
            echogen.aggregate_simulation_responses(responses, min_samples=2)

    def test_aggregate_simulation_responses_requires_consensus(self):
        with self.assertRaisesRegex(echogen.ConsensusError, "valid"):
            echogen.aggregate_simulation_responses(
                ["not a call", "also not a call"],
                min_samples=2,
            )

    def test_adjustment_function_name_accepts_single_instance_per_type(self):
        parameters = [
            {
                "add_ball": (
                    (10.0, 20.0),
                    5.0,
                    2.0,
                    (0.0, 0.0),
                )
            }
        ]

        self.assertEqual(echogen.adjustment_function_name(parameters), parameters)


class TestBundledAssets(unittest.TestCase):
    def test_default_image_is_bundled_and_encodable(self):
        self.assertEqual(
            echogen.build_argument_parser().parse_args([]).image,
            str(echogen.DEFAULT_IMAGE_PATH),
        )
        self.assertTrue(Path(echogen.DEFAULT_IMAGE_PATH).is_file())
        self.assertGreater(len(echogen.encode_image(echogen.DEFAULT_IMAGE_PATH)), 100)


if __name__ == "__main__":
    unittest.main()
