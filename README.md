# EchoGen

EchoGen is an image-to-physics research prototype. It asks a vision-capable
language model to describe objects with a small simulation DSL, aggregates
several independent samples, validates the result, and optionally renders it
with Pymunk and Pygame.

This repository is a prototype, not a claim of physically accurate video
prediction. Model output can be incomplete or wrong and should be reviewed.

## Safety Model

Language-model text is untrusted input. EchoGen never uses `exec` or `eval`.
Instead, `echogen.py`:

1. parses output with Python's AST parser;
2. accepts only direct calls to `add_ball`, `add_curved_floor`, and
   `add_polygon`;
3. accepts only finite numeric literals inside lists and tuples;
4. validates argument counts, geometry, radius, mass, thickness, and friction;
5. dispatches validated calls through an explicit handler allowlist.

Imports, attribute access, expressions, keyword expansion, assignments, and
all other statements are rejected before simulation code runs.

## Requirements

- Python 3.9+
- Runtime packages listed in `requirements.txt`
- An OpenAI API key only for online image analysis

Create an isolated environment and install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Tests use only the Python standard library and do not need network access or
the simulation packages:

```bash
python -m unittest discover -s tests -v
```

## Online Usage

Provide credentials through the environment; never add a key to source code:

```bash
export OPENAI_API_KEY="your-key"
python echogen.py
```

The default command uses the bundled example image. To analyze another image,
pass it explicitly with `--image /path/to/image.jpg`.

Useful options:

```text
--model MODEL       Model name (default: gpt-4o)
--epochs N          Iterative refinement rounds (default: 3)
--samples N         Independent samples per round (default: 3)
--timeout SECONDS   HTTP timeout per request (default: 30)
--no-display        Validate and print calls without opening Pygame
```

Online mode performs network requests and may incur API charges.

## Offline Usage

Use a saved response to exercise parsing and simulation without any network
call:

```bash
python echogen.py --response-file examples/sample-simulation-response.txt --no-display
```

A `.txt` file contains one set of calls. A `.json` file may contain a list of
independent response strings. EchoGen groups structurally matching samples,
matches repeated objects one-to-one by geometry and parameters, and robustly
averages only reliable, unambiguous matches. It rejects consensus when object
counts, shapes, distances, or assignments do not agree reliably.

Remove `--no-display` to launch the physical simulation after installing the
runtime dependencies.

## Method

Each online epoch requests several model samples. EchoGen discards invalid
samples, groups the remaining samples by function and geometry shape, selects
the largest matching group, and computes a deterministic one-to-one assignment
for repeated objects before taking robust numeric means. Ambiguous or distant
assignments fail closed instead of creating averaged objects that no model
reported. The validated consensus is inserted into the next prompt as a hint.
This preserves the original iterative image-to-simulation idea without
executing generated code.

## Project Layout

```text
echogen.py                                  Safe parser, API client, and CLI
echogen_assets/sample-input-image.jpg       Bundled default input image
examples/sample-simulation-response.txt     Offline demonstration input
tests/test_echogen.py                       Network-free unit tests
requirements.txt                            Simulation runtime dependencies
pyproject.toml                              Python and test metadata
```

## Limitations

- Object recognition and inferred physical parameters are model estimates.
- Consensus requires structurally matching samples and does not prove truth.
- The current DSL supports circles, polygons, and curved static floors only.
- Pymunk rendering uses a simplified 2D model with no learned dynamics.

