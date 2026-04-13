# tooluse-generator

Offline synthetic data generation system that produces multi-turn conversations containing multi-step / multi-tool tool-use traces, grounded in ToolBench API schemas.

## Installation

```bash
# Clone the repo
git clone https://github.com/placeholder/tooluse-generator.git
cd tooluse-generator

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Copy and fill in your API keys (optional — offline mode works without them)
cp .env.example .env

# Verify installation
tooluse --help
```

> **Requires Python 3.10+.** The `sentence-transformers` embedding model (~90 MB) downloads automatically on the first `tooluse build` run.

## Quick Start

```bash
# 1. Build artifacts from ToolBench data (subset of 500 tools for speed)
python -m tooluse_gen.cli.main build \
    --input-dir data/toolenv/tools \
    --output-dir output/build \
    --force \
    --max-tools 500 \
    --no-semantic-edges

# 2. Generate conversations — Run B (steering ON, default)
python -m tooluse_gen.cli.main generate \
    --output output/run_b.jsonl \
    --build-dir output/build \
    --count 100 --seed 42

# 3. Generate conversations — Run A (steering OFF, for diversity experiment)
python -m tooluse_gen.cli.main generate \
    --output output/run_a.jsonl \
    --build-dir output/build \
    --count 100 --seed 42 \
    --no-cross-conversation-steering

# 4. Evaluate both runs
python -m tooluse_gen.cli.main evaluate output/run_a.jsonl
python -m tooluse_gen.cli.main evaluate output/run_b.jsonl
```

> **Performance note:** With `--max-tools 500 --no-semantic-edges`, the build step completes in ~15 seconds and generation runs at ~50 conversations/second. Building with the full 10K+ ToolBench corpus and semantic edges requires ~2 hours for embedding computation on CPU.

## CLI Reference

### Global options

| Flag | Default | Description |
|------|---------|-------------|
| `--verbose` / `-v` | 0 | Increase verbosity (repeat: `-vv` for debug) |
| `--quiet` / `-q` | off | Suppress non-essential output |
| `--config` / `-c` | `config/default.yaml` | Path to YAML configuration file |
| `--config-from` | — | Load config from a previous output JSONL for reproducibility |

### `build`

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` / `-i` | *(required)* | Directory containing raw ToolBench JSON files |
| `--output-dir` / `-o` | `output/build` | Directory for built artifacts |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence-transformer model for semantic edges |
| `--similarity-threshold` | `0.7` | Cosine threshold for semantic edges (0–1) |
| `--force` / `-f` | off | Overwrite existing artifacts |
| `--generate-pools` | off | Generate value pools for mock responses |
| `--max-tools` | all | Limit tools loaded (stratified sample across domains) |
| `--no-semantic-edges` | off | Skip embedding computation and semantic edges for faster builds |

### `generate`

| Flag | Default | Description |
|------|---------|-------------|
| `--output` / `-o` | *(required)* | Output JSONL file path |
| `--build-dir` / `-b` | `output/build` | Pre-built artifacts directory |
| `--count` / `-n` | `100` | Number of conversations to generate |
| `--seed` / `-s` | `42` | Random seed for reproducibility |
| `--min-steps` | `2` | Minimum tool calls per conversation |
| `--max-steps` | `5` | Maximum tool calls per conversation |
| `--domains` | all | Comma-separated domain filter (e.g. `Travel,Finance`) |
| `--no-cross-conversation-steering` | off | Disable diversity steering (Run A mode) |
| `--max-retries` | `3` | Max repair attempts per conversation |
| `--quality-threshold` | `3.5` | Minimum average judge score to accept (0–5) |
| `--no-cache` | off | Disable prompt caching |

### `evaluate`

| Flag | Default | Description |
|------|---------|-------------|
| `INPUT` *(positional)* | *(required)* | JSONL file of conversations to evaluate |
| `--output` / `-o` | — | Write enriched JSONL with updated scores |
| `--format` / `-f` | `table` | Output format: `json`, `table`, or `markdown` |
| `--rescore` | off | Re-run the judge on all conversations |

## Configuration

All settings live in `config/default.yaml`. Override via `--config path/to/custom.yaml` or `--config-from prev_output.jsonl`.

### Configuration Fields

| Section | Field | Default | Description |
|---------|-------|---------|-------------|
| `models` | `assistant` | `"gpt-4o-mini"` | LLM for the assistant agent |
| `models` | `judge` | `"gpt-4o-mini"` | LLM for quality scoring |
| `models` | `user_simulator` | `"gpt-4o-mini"` | LLM for user message generation |
| `models` | `mock_generator` | `"gpt-4o-mini"` | LLM for mock tool responses |
| `models` | `embedding` | `"all-MiniLM-L6-v2"` | Sentence-transformer for semantic edges |
| `quality` | `min_score` | `3.5` | Minimum average judge score to accept (1.0–5.0) |
| `quality` | `max_retries` | `3` | Maximum repair attempts per conversation |
| `quality` | `dimensions` | `[tool_correctness, ...]` | Scoring dimensions (order matters for prompts) |
| `sampling` | `min_steps` | `2` | Minimum tool calls per conversation |
| `sampling` | `max_steps` | `5` | Maximum tool calls per conversation |
| `sampling` | `similarity_threshold` | `0.7` | Cosine threshold for semantic graph edges (0–1) |
| `sampling` | `domains` | `null` | Restrict to these domains; `null` = use all |
| `sampling` | `excluded_tools` | `null` | Tool endpoint IDs to exclude; `null` = none |
| `diversity` | `enabled` | `true` | Enable cross-conversation diversity steering |
| `diversity` | `weight_decay` | `0.9` | Inverse-frequency decay for tool weights (0–1) |
| `diversity` | `min_domain_coverage` | `0.5` | Target domain coverage fraction before repeating |
| `paths` | `build_dir` | `"./output/build"` | Build artifacts directory |
| `paths` | `output_dir` | `"./output"` | Output root directory |
| `paths` | `cache_dir` | `"./.cache"` | Intermediate cache directory |
| *(root)* | `seed` | `42` | Global random seed for reproducibility |
| *(root)* | `verbose` | `0` | Verbosity level (0 = normal, 1 = info, 2 = debug) |

Environment variables in `.env` override API keys. The pipeline works fully offline without API keys — all agents use template-based generation and heuristic scoring when no LLM client is configured.

## Output Format

Each line in the output JSONL is a conversation record. See [DESIGN.md — Appendix: Output Schema Reference](DESIGN.md#appendix-output-schema-reference) for the complete field-by-field schema documentation.

```json
{
  "conversation_id": "683249ea-fd9c-41dc-8e0e-270a2db8ca45",
  "messages": [
    {"role": "user", "content": "Find me a hotel in Paris"},
    {"role": "assistant", "content": "What's your budget range?"},
    {"role": "user", "content": "Under 200 euros per night"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint": "hotels/search", "arguments": {"city": "Paris", "max_price": 200},
       "tool_name": "Hotels API", "call_id": "c1"}
    ]},
    {"role": "tool", "content": {"results": [{"id": "htl_881", "name": "Hotel du Marais", "price": 175}]}},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint": "hotels/book", "arguments": {"hotel_id": "htl_881"},
       "tool_name": "Hotels API", "call_id": "c2"}
    ]},
    {"role": "tool", "content": {"booking_id": "bk_3391", "status": "confirmed"}},
    {"role": "assistant", "content": "I've booked Hotel du Marais. Confirmation: bk_3391."}
  ],
  "judge_scores": {
    "tool_correctness": 5,
    "argument_grounding": 3,
    "task_completion": 5,
    "naturalness": 4
  },
  "metadata": {
    "seed": 42,
    "tools_used": ["hotels_api"],
    "num_turns": 8,
    "num_tool_calls": 2,
    "num_distinct_tools": 1,
    "domains": ["Travel"]
  }
}
```

## Diversity Experiment

The spec requires running the pipeline twice with the same seed to compare diversity steering's effect on tool coverage and quality.

### Running the experiment

```bash
# 1. Build once (shared by both runs)
python -m tooluse_gen.cli.main build \
    --input-dir data/toolenv/tools --output-dir output/build \
    --force --max-tools 500 --no-semantic-edges

# 2. Run A: steering disabled
python -m tooluse_gen.cli.main generate \
    --output output/run_a.jsonl --build-dir output/build \
    --count 100 --seed 42 --no-cross-conversation-steering

# 3. Run B: steering enabled (default)
python -m tooluse_gen.cli.main generate \
    --output output/run_b.jsonl --build-dir output/build \
    --count 100 --seed 42

# 4. Evaluate both
python -m tooluse_gen.cli.main evaluate output/run_a.jsonl
python -m tooluse_gen.cli.main evaluate output/run_b.jsonl
```

### Generated Datasets

- [output/run_a.jsonl](output/run_a.jsonl) — Run A: 100 conversations, steering **disabled**
- [output/run_b.jsonl](output/run_b.jsonl) — Run B: 100 conversations, steering **enabled**

### Results (100 conversations per run, seed=42, 500 tools, 49 domains)

| Metric | Run A (no steering) | Run B (steering) | Delta |
|--------|-------------------|-----------------|-------|
| Conversations | 100 | 100 | 0 |
| Pass rate | 100% | 100% | 0 |
| Mean score | 4.25 | 4.25 | 0.00 |
| Tool entropy | 7.4371 | 7.4323 | -0.005 |
| Unique tools | **197** | 196 | -1 |
| Unique tool combos | 94 | **100** | **+6** |
| Pattern repetition | 6% | **0%** | **-6%** |
| Unique domains | 38 | **41** | **+3** |
| Domain coverage | 78% | **84%** | **+6%** |

**Key finding**: Steering eliminates pattern repetition (every conversation uses a unique tool combination) and improves domain coverage (+6%), with zero quality cost. See [DESIGN.md Section 9](DESIGN.md#9-diversity--quality-analysis) for full analysis.

### Reproducibility

- **Same seed**: Both runs use `--seed 42` for valid comparison
- **Same data**: Both runs use the same build artifacts
- **Config replay**: Use `--config-from output/run_a.jsonl` to reproduce a previous run's exact configuration

## Running Tests

```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# E2E tests (skipped by default — generates 100+ conversations)
pytest --run-e2e

# All tests (unit + integration; E2E skipped without --run-e2e)
pytest

# Specific test file
pytest tests/integration/test_retry_repair.py -v

# With coverage
pytest --cov=tooluse_gen --cov-report=term-missing
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: tooluse_gen` | Run `pip install -e ".[dev]"` from the project root |
| Build fails: "No tools passed quality filter" | Check `--input-dir` points to valid ToolBench data. The default FAIR filter excludes poorly documented tools |
| Generate fails: "Build directory does not exist" | Run `tooluse build` first to create artifacts in `--build-dir` |
| Generate produces 0 conversations | Graph may be too sparse. Try `--min-steps 1 --max-steps 2` or lower `--similarity-threshold 0.1` during build |
| `EmbeddingService` download fails | Use `--no-semantic-edges` to skip embedding entirely, or ensure internet access for the ~90 MB model download |
| Evaluate reports all "invalid" | Ensure the JSONL contains `tool_calls` in assistant messages. The structural validator checks role alternation and tool call presence |
| `--config-from` fails | The JSONL must have a metadata header (first line with `__metadata__: true`). Files from `tooluse generate` include this automatically |
| Tests show `SKIPPED` | E2E tests require `pytest --run-e2e`. Without this flag they are skipped by design |
| Slow build with full ToolBench | Use `--max-tools 500 --no-semantic-edges` to build a fast subset (~15s). Full corpus with embeddings takes ~2 hours on CPU |

### Environment Requirements

- **Python**: 3.10 or later
- **OS**: macOS, Linux, Windows (tested on macOS and Linux)
- **Disk**: ~100 MB for ToolBench data, ~50 MB for build artifacts
- **Network**: Only needed for first-time embedding model download and LLM API calls (optional)

## Project Structure

```
src/tooluse_gen/
├── cli/          # Typer CLI commands (build, generate, evaluate)
├── core/         # Shared config, models, JSONL I/O, caching
├── registry/     # ToolBench ingestion → clean internal data model
├── graph/        # Tool Graph construction + MCTS chain sampler
├── agents/       # Multi-agent conversation generator
├── evaluation/   # LLM-as-judge + retry/repair + diversity report
└── utils/        # Logging, progress bars, seeding
```

See [DESIGN.md](DESIGN.md) for architecture decisions, prompt design, and diversity analysis.
