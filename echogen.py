import argparse
import ast
import base64
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"
ALLOWED_FUNCTIONS = frozenset({"add_ball", "add_curved_floor", "add_polygon"})
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "echogen_assets" / "sample-input-image.jpg"
MAX_MATCHED_OBJECTS_PER_TYPE = 12
MAX_OBJECT_DISTANCE = 0.5
MIN_ASSIGNMENT_GAP = 0.05
OBJECT_DISTANCE_SCALE_FLOOR = 10.0


class EchoGenError(Exception):
    """Base class for expected EchoGen failures."""


class ConfigError(EchoGenError):
    """Raised when required runtime configuration is missing or invalid."""


class RuntimeDependencyError(EchoGenError):
    """Raised when optional simulation dependencies are unavailable."""


class SimulationParseError(EchoGenError):
    """Raised when model output is not a safe simulation specification."""


class SimulationExecutionError(EchoGenError):
    """Raised when a validated simulation call cannot be dispatched."""


class LLMRequestError(EchoGenError):
    """Raised when the remote model request fails."""


class LLMResponseError(EchoGenError):
    """Raised when the remote model response has an unexpected shape."""


class ConsensusError(EchoGenError):
    """Raised when model samples do not provide a usable consensus."""


@dataclass(frozen=True)
class SimulationCall:
    """A validated call to one allowlisted physics helper."""

    name: str
    args: Tuple[Any, ...]


Transport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]


def _load_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            f"Missing runtime dependency '{name}'. Install requirements.txt."
        ) from exc


def create_space(gravity):
    """
    build a pymunk space with gravity
    """
    pymunk = _load_module("pymunk")
    space = pymunk.Space()
    space.gravity = gravity
    return space


ƒ_space = create_space


def add_ball(space, position, radius, mass, velocity):
    """
    add a ball to the space
    """
    pymunk = _load_module("pymunk")
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, inertia)
    body.position = position
    body.velocity = velocity
    shape = pymunk.Circle(body, radius)
    shape.friction = 0.7
    space.add(body, shape)

def generate_spline(points, num_points=100):
    """
    use spline interpolation to generate a smooth curve
    """
    if len(points) < 2:
        raise ValueError("at least two points are required")
    if num_points < 2:
        raise ValueError("num_points must be at least two")

    np = _load_module("numpy")
    interpolate = _load_module("scipy.interpolate")
    points = np.array(points)
    spline_degree = min(3, len(points) - 1)
    tck, _ = interpolate.splprep(points.T, s=0, k=spline_degree)
    u = np.linspace(0, 1, num_points)
    spline_points = interpolate.splev(u, tck)
    return list(zip(spline_points[0], spline_points[1]))


def add_curved_floor(space, points, thickness, frictions):
    """
    add a curved floor to the space
    """
    pymunk = _load_module("pymunk")
    spline_points = generate_spline(points)
    for i in range(len(spline_points) - 1):
        segment = pymunk.Segment(space.static_body, spline_points[i], spline_points[i + 1], thickness)
        segment.friction = frictions
        space.add(segment)

def add_polygon(space, position, vertices, mass, velocity):
    """
    add a polygon to the space
    """
    pymunk = _load_module("pymunk")
    inertia = pymunk.moment_for_poly(mass, vertices)
    body = pymunk.Body(mass, inertia)
    body.position = position
    body.velocity = velocity
    shape = pymunk.Poly(body, vertices)
    shape.friction = 0.7
    space.add(body, shape)

def draw_space(space, screen):
    """
    build a pymunk space with gravity
    """
    pygame_util = _load_module("pymunk.pygame_util")
    draw_options = pygame_util.DrawOptions(screen)
    space.debug_draw(draw_options)

def load_api_key(environ=None):
    """Read the OpenAI API key without embedding credentials in source."""
    environment = os.environ if environ is None else environ
    api_key = environment.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "OPENAI_API_KEY is required for online generation. "
            "Set it in the environment or use --response-file for offline mode."
        )
    return api_key


def encode_image(image_path):
    """Return an image as a base64 string for the vision request."""
    path = Path(image_path)
    if not path.is_file():
        raise ConfigError(f"Image file does not exist: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _strip_markdown_fence(response):
    text = response.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        raise SimulationParseError("Incomplete Markdown code fence in model output")
    return "\n".join(lines[1:-1]).strip()


def _literal_value(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise SimulationParseError(
            "Simulation arguments must contain only numeric literals, lists, and tuples"
        ) from exc
    if isinstance(value, bool):
        raise SimulationParseError("Boolean simulation parameters are not allowed")
    return value


def _number(value, field, *, positive=False, non_negative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationParseError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimulationParseError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise SimulationParseError(f"{field} must be finite")
    if positive and result <= 0:
        raise SimulationParseError(f"{field} must be positive")
    if non_negative and result < 0:
        raise SimulationParseError(f"{field} must be non-negative")
    return result


def _point(value, field):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise SimulationParseError(f"{field} must be a two-number point")
    return (
        _number(value[0], f"{field}.x"),
        _number(value[1], f"{field}.y"),
    )


def _points(value, field, *, minimum):
    if not isinstance(value, (tuple, list)) or len(value) < minimum:
        raise SimulationParseError(
            f"{field} must contain at least {minimum} points"
        )
    return tuple(_point(point, f"{field}[{index}]") for index, point in enumerate(value))


def _validate_call(name, raw_args):
    if name == "add_ball":
        if len(raw_args) != 4:
            raise SimulationParseError("add_ball requires four arguments after space")
        position, radius, mass, velocity = raw_args
        return (
            _point(position, "position"),
            _number(radius, "radius", positive=True),
            _number(mass, "mass", positive=True),
            _point(velocity, "velocity"),
        )

    if name == "add_curved_floor":
        if len(raw_args) != 3:
            raise SimulationParseError(
                "add_curved_floor requires three arguments after space"
            )
        points, thickness, friction = raw_args
        return (
            _points(points, "points", minimum=2),
            _number(thickness, "thickness", positive=True),
            _number(friction, "friction", non_negative=True),
        )

    if name == "add_polygon":
        if len(raw_args) != 4:
            raise SimulationParseError("add_polygon requires four arguments after space")
        position, vertices, mass, velocity = raw_args
        return (
            _point(position, "position"),
            _points(vertices, "vertices", minimum=3),
            _number(mass, "mass", positive=True),
            _point(velocity, "velocity"),
        )

    raise SimulationParseError(f"Unsupported simulation function: {name}")


def parse_functions(response):
    """Parse LLM text into validated calls without executing Python code."""
    if not isinstance(response, str) or not response.strip():
        raise SimulationParseError("Model output must be a non-empty string")

    source = _strip_markdown_fence(response)
    try:
        module = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise SimulationParseError("Model output is not valid function-call syntax") from exc

    if not module.body:
        raise SimulationParseError("Model output did not contain simulation calls")

    parsed_calls = []
    for statement in module.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise SimulationParseError(
                "Each model output line must be a direct simulation function call"
            )

        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id not in ALLOWED_FUNCTIONS:
            raise SimulationParseError("Model output requested a non-allowlisted function")
        if call.keywords:
            raise SimulationParseError("Keyword and expanded arguments are not allowed")
        if not call.args or not isinstance(call.args[0], ast.Name) or call.args[0].id != "space":
            raise SimulationParseError(
                "The first simulation function argument must be the space identifier"
            )

        raw_args = tuple(_literal_value(argument) for argument in call.args[1:])
        parsed_calls.append(
            SimulationCall(call.func.id, _validate_call(call.func.id, raw_args))
        )

    return parsed_calls


def render_simulation_calls(calls):
    """Render validated calls as a human-readable hint for the next epoch."""
    rendered = []
    for call in calls:
        if call.name not in ALLOWED_FUNCTIONS:
            raise SimulationParseError(
                f"Cannot render unsupported simulation function: {call.name}"
            )
        validated_args = _validate_call(call.name, call.args)
        arguments = ", ".join(repr(argument) for argument in validated_args)
        rendered.append(f"{call.name}(space, {arguments})")
    return "\n".join(rendered)


def execute_simulation_calls(calls, space, handlers=None):
    """Dispatch already validated calls through an explicit function allowlist."""
    allowed_handlers = (
        {
            "add_ball": add_ball,
            "add_curved_floor": add_curved_floor,
            "add_polygon": add_polygon,
        }
        if handlers is None
        else dict(handlers)
    )

    for call in calls:
        if call.name not in ALLOWED_FUNCTIONS:
            raise SimulationExecutionError(
                f"Unsupported simulation function: {call.name}"
            )
        handler = allowed_handlers.get(call.name)
        if handler is None:
            raise SimulationExecutionError(
                f"No allowlisted handler configured for {call.name}"
            )
        try:
            validated_args = _validate_call(call.name, call.args)
            handler(space, *validated_args)
        except EchoGenError:
            raise
        except Exception as exc:
            raise SimulationExecutionError(
                f"Failed to execute {call.name}: {exc}"
            ) from exc


def _http_transport(url, headers, payload, timeout):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset("utf-8")
            body = response.read().decode(charset)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:500]
        raise LLMRequestError(
            f"OpenAI request failed with HTTP {exc.code}: {details}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise LLMRequestError(f"OpenAI request failed: {exc}") from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("OpenAI returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise LLMResponseError("OpenAI returned a non-object JSON response")
    return decoded


def call_chatgpt(
    prompt,
    *,
    encoded_image,
    api_key,
    model=DEFAULT_MODEL,
    timeout=30.0,
    transport=None,
):
    """Request one image-to-simulation sample and validate its response shape."""
    if not prompt.strip():
        raise ConfigError("Prompt must not be empty")
    if not api_key.strip():
        raise ConfigError("API key must not be empty")
    if isinstance(timeout, bool):
        raise ConfigError("Request timeout must be a positive finite number")
    try:
        request_timeout = float(timeout)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigError(
            "Request timeout must be a positive finite number"
        ) from exc
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ConfigError("Request timeout must be a positive finite number")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 500,
        "temperature": 0,
    }
    response = (transport or _http_transport)(
        DEFAULT_API_URL,
        headers,
        payload,
        request_timeout,
    )

    try:
        choices = response["choices"]
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError(
            "OpenAI response did not contain choices[0].message.content"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("OpenAI response content was empty")
    return content.strip()

def find_similar_and_average(values, eps=0.5, min_samples=2):
    """Return a robust mean while ignoring missing, non-finite, and outlier values."""
    finite_values = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(numeric_value):
            finite_values.append(numeric_value)

    if not finite_values:
        return None
    if len(finite_values) <= max(2, min_samples):
        return statistics.fmean(finite_values)

    median = statistics.median(finite_values)
    deviations = [abs(value - median) for value in finite_values]
    median_deviation = statistics.median(deviations)
    threshold = max(float(eps), 3.0 * median_deviation)
    inliers = [
        value for value in finite_values if abs(value - median) <= threshold
    ]
    if len(inliers) < min_samples:
        return median
    return statistics.fmean(inliers)


def extract_numbers_from_string(value):
    """Extract signed integers, decimals, and scientific notation as floats."""
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    numbers = []
    for token in re.findall(number_pattern, value):
        try:
            number = float(token)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def extract_parameters(function_calls):
    """Convert validated calls into a stable name-to-argument mapping."""
    calls = []
    for item in function_calls:
        if isinstance(item, SimulationCall):
            calls.append(item)
        elif isinstance(item, str):
            calls.extend(parse_functions(item))
        else:
            raise SimulationParseError(
                "extract_parameters accepts SimulationCall objects or call strings"
            )

    counts = defaultdict(int)
    parameters = {}
    for call in calls:
        counts[call.name] += 1
        suffix = "" if counts[call.name] == 1 else str(counts[call.name])
        parameters[f"{call.name}{suffix}"] = call.args
    return parameters


def add_function_name(parameters, function_prototype=None):
    """Safely render structured parameters; retained for prototype compatibility."""
    del function_prototype
    calls = []
    for keyed_name, arguments in parameters.items():
        name = re.sub(r"\d+$", "", keyed_name)
        if not isinstance(arguments, (tuple, list)):
            raise SimulationParseError(
                f"Arguments for {keyed_name} must be a sequence"
            )
        calls.append(SimulationCall(name, _validate_call(name, tuple(arguments))))
    return render_simulation_calls(calls)


def adjustment_function_name(parameters):
    """Validate legacy parameter groups without assuming duplicate object types."""
    adjusted = []
    for parameter_group in parameters:
        converted = {}
        for keyed_name, arguments in parameter_group.items():
            name = re.sub(r"\d+$", "", keyed_name)
            if isinstance(arguments, str):
                parsed = parse_functions(f"{name}(space, {arguments})")
                converted[keyed_name] = parsed[0].args
            elif isinstance(arguments, (tuple, list)):
                converted[keyed_name] = _validate_call(name, tuple(arguments))
            else:
                raise SimulationParseError(
                    f"Arguments for {keyed_name} must be structured values"
                )
        adjusted.append(converted)
    return adjusted


def _value_shape(value):
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_shape(item) for item in value))
    return "number"


def _call_signature(call):
    return call.name, tuple(_value_shape(argument) for argument in call.args)


def _calls_shape(calls):
    return tuple(sorted(
        (_call_signature(call) for call in calls),
        key=repr,
    ))


def _flatten_numeric_values(value):
    if isinstance(value, tuple):
        return tuple(
            number
            for item in value
            for number in _flatten_numeric_values(item)
        )
    return (float(value),)


def _call_distance(reference_call, candidate_call):
    if _call_signature(reference_call) != _call_signature(candidate_call):
        return math.inf

    reference_values = _flatten_numeric_values(reference_call.args)
    candidate_values = _flatten_numeric_values(candidate_call.args)
    normalized_differences = [
        abs(reference - candidate)
        / max(
            abs(reference),
            abs(candidate),
            OBJECT_DISTANCE_SCALE_FLOOR,
        )
        for reference, candidate in zip(reference_values, candidate_values)
    ]
    return max(normalized_differences, default=0.0)


def _two_best_assignments(cost_matrix):
    object_count = len(cost_matrix)

    @lru_cache(maxsize=None)
    def solve(reference_index, used_candidates):
        if reference_index == object_count:
            return ((0.0, ()),)

        assignments = []
        for candidate_index in range(object_count):
            candidate_bit = 1 << candidate_index
            if used_candidates & candidate_bit:
                continue
            for remaining_cost, remaining_assignment in solve(
                reference_index + 1,
                used_candidates | candidate_bit,
            ):
                assignments.append(
                    (
                        cost_matrix[reference_index][candidate_index]
                        + remaining_cost,
                        (candidate_index,) + remaining_assignment,
                    )
                )

        assignments.sort(key=lambda item: (item[0], item[1]))
        return tuple(assignments[:2])

    return solve(0, 0)


def _match_call_group(reference_calls, candidate_calls):
    if len(reference_calls) != len(candidate_calls):
        raise ConsensusError("Model samples disagree on object count")
    if len(reference_calls) > MAX_MATCHED_OBJECTS_PER_TYPE:
        raise ConsensusError(
            "Too many repeated objects to match reliably"
        )

    cost_matrix = tuple(
        tuple(
            _call_distance(reference_call, candidate_call)
            for candidate_call in candidate_calls
        )
        for reference_call in reference_calls
    )
    assignments = _two_best_assignments(cost_matrix)
    best_cost, best_assignment = assignments[0]
    matched_distances = [
        cost_matrix[index][candidate_index]
        for index, candidate_index in enumerate(best_assignment)
    ]
    if any(distance > MAX_OBJECT_DISTANCE for distance in matched_distances):
        raise ConsensusError(
            "Model samples do not reliably match the same objects"
        )

    if len(assignments) > 1:
        second_best_cost = assignments[1][0]
        average_gap = (second_best_cost - best_cost) / len(reference_calls)
        if average_gap <= MIN_ASSIGNMENT_GAP:
            raise ConsensusError(
                "Model samples have ambiguous object matching"
            )

    return [
        candidate_calls[candidate_index]
        for candidate_index in best_assignment
    ]


def _align_calls(reference_calls, candidate_calls):
    if _calls_shape(reference_calls) != _calls_shape(candidate_calls):
        raise ConsensusError("Model samples disagree on call structure")

    reference_groups = defaultdict(list)
    candidate_groups = defaultdict(list)
    for index, call in enumerate(reference_calls):
        reference_groups[_call_signature(call)].append((index, call))
    for call in candidate_calls:
        candidate_groups[_call_signature(call)].append(call)

    aligned_calls = [None] * len(reference_calls)
    for signature, indexed_reference_calls in reference_groups.items():
        reference_group = [call for _, call in indexed_reference_calls]
        matched_group = _match_call_group(
            reference_group,
            candidate_groups[signature],
        )
        for (reference_index, _), matched_call in zip(
            indexed_reference_calls,
            matched_group,
        ):
            aligned_calls[reference_index] = matched_call
    return aligned_calls


def _aggregate_values(values, min_samples):
    first = values[0]
    if isinstance(first, tuple):
        if not all(isinstance(value, tuple) and len(value) == len(first) for value in values):
            raise ConsensusError("Model samples disagree on geometry shape")
        return tuple(
            _aggregate_values(
                [value[index] for value in values],
                min_samples,
            )
            for index in range(len(first))
        )

    averaged = find_similar_and_average(values, min_samples=min_samples)
    if averaged is None:
        raise ConsensusError("Model samples did not contain finite numeric values")
    return averaged


def aggregate_simulation_responses(responses, min_samples=2):
    """Match a reliable common object layout and average numeric parameters."""
    if min_samples < 1:
        raise ValueError("min_samples must be at least one")

    grouped_samples = defaultdict(list)
    parse_errors = []
    for response in responses:
        try:
            calls = parse_functions(response)
        except SimulationParseError as exc:
            parse_errors.append(str(exc))
            continue
        grouped_samples[_calls_shape(calls)].append(calls)

    if not grouped_samples:
        detail = f": {parse_errors[0]}" if parse_errors else ""
        raise ConsensusError(f"No valid simulation responses were received{detail}")

    _, samples = max(
        grouped_samples.items(),
        key=lambda item: (len(item[1]), repr(item[0])),
    )
    if len(samples) < min_samples:
        raise ConsensusError(
            f"Only {len(samples)} matching valid response(s); "
            f"{min_samples} required for consensus"
        )

    aligned_samples = [samples[0]]
    aligned_samples.extend(
        _align_calls(samples[0], sample)
        for sample in samples[1:]
    )

    aggregated_calls = []
    for call_index, reference_call in enumerate(aligned_samples[0]):
        arguments = tuple(
            _aggregate_values(
                [
                    sample[call_index].args[arg_index]
                    for sample in aligned_samples
                ],
                min_samples,
            )
            for arg_index in range(len(reference_call.args))
        )
        aggregated_calls.append(
            SimulationCall(
                reference_call.name,
                _validate_call(reference_call.name, arguments),
            )
        )
    return aggregated_calls


INITIAL_PROMPT = """\
Recognize physical objects in the attached image and describe them only with
calls to these functions:

add_ball(space, position, radius, mass, velocity)
add_curved_floor(space, points, thickness, friction)
add_polygon(space, position, vertices, mass, velocity)

Use image coordinates with origin at the top-left, positive x to the right,
and positive y downward. Return direct function calls only. Use numeric
literals, lists, and tuples; do not use variables, imports, expressions, or
other Python statements.
"""


def generate_simulation_calls(
    image_path,
    *,
    api_key,
    model=DEFAULT_MODEL,
    epochs=3,
    samples_per_epoch=3,
    timeout=30.0,
    transport=None,
):
    """Iteratively obtain and aggregate image-to-physics model samples."""
    if epochs < 1 or samples_per_epoch < 1:
        raise ConfigError("epochs and samples_per_epoch must both be positive")

    encoded_image = encode_image(image_path)
    prompt = INITIAL_PROMPT
    consensus_calls = None
    required_samples = 1 if samples_per_epoch == 1 else 2

    for _ in range(epochs):
        responses = [
            call_chatgpt(
                prompt,
                encoded_image=encoded_image,
                api_key=api_key,
                model=model,
                timeout=timeout,
                transport=transport,
            )
            for _ in range(samples_per_epoch)
        ]
        consensus_calls = aggregate_simulation_responses(
            responses,
            min_samples=required_samples,
        )
        prompt = (
            f"{INITIAL_PROMPT}\nLikely parameters from the previous epoch:\n"
            f"{render_simulation_calls(consensus_calls)}"
        )

    return consensus_calls


def load_response_file(path):
    """Load one offline response or a JSON list of independent responses."""
    response_path = Path(path)
    if not response_path.is_file():
        raise ConfigError(f"Response file does not exist: {response_path}")
    content = response_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ConfigError(f"Response file is empty: {response_path}")

    if response_path.suffix.lower() == ".json":
        try:
            responses = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Response file contains invalid JSON: {exc}") from exc
        if not isinstance(responses, list) or not all(
            isinstance(response, str) for response in responses
        ):
            raise ConfigError("JSON response files must contain a list of strings")
        min_samples = 1 if len(responses) == 1 else 2
        return aggregate_simulation_responses(responses, min_samples=min_samples)

    return parse_functions(content)


def run_simulation(calls, *, width=1000, height=1000):
    """Display validated calls in a Pymunk/Pygame simulation window."""
    pygame = _load_module("pygame")
    pygame.init()
    try:
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((width, height))
        space = create_space((0, 0))
        execute_simulation_calls(calls, space)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            screen.fill((255, 255, 255))
            space.step(1 / 50.0)
            draw_space(space, screen)
            pygame.display.flip()
            clock.tick(50)
    finally:
        pygame.quit()


def _positive_finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "must be a positive finite number"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Convert an image or saved model response into a safe physics simulation."
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument(
        "--response-file",
        help="Offline text response or JSON list of response strings; skips all network calls.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--timeout", type=_positive_finite_float, default=30.0)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Validate and print the simulation specification without opening a window.",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.response_file:
        calls = load_response_file(args.response_file)
    else:
        calls = generate_simulation_calls(
            args.image,
            api_key=load_api_key(),
            model=args.model,
            epochs=args.epochs,
            samples_per_epoch=args.samples,
            timeout=args.timeout,
        )

    print(render_simulation_calls(calls))
    if not args.no_display:
        run_simulation(calls)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EchoGenError as exc:
        raise SystemExit(f"EchoGen error: {exc}") from exc
