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

# Copy and fill in your API keys
cp .env.example .env
```

## Quick Start

```bash
# 1. Build artifacts from ToolBench data
tooluse build --data-dir data/toolbench --output-dir output/artifacts

# 2. Generate conversations (Run B — cross-conversation steering ON)
tooluse generate --num 100 --seed 42 --output output/conversations.jsonl

# 3. Generate without steering (Run A — for diversity experiment)
tooluse generate --num 100 --seed 42 --no-cross-conversation-steering \
    --output output/conversations_no_steering.jsonl

# 4. Evaluate generated conversations
tooluse evaluate --input output/conversations.jsonl
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `tooluse build` | Ingest ToolBench data and build graph/index artifacts |
| `tooluse generate` | Generate multi-turn tool-use conversations |
| `tooluse evaluate` | Score conversations with LLM-as-judge |

### `tooluse generate` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--num` / `-n` | 100 | Number of conversations |
| `--seed` / `-s` | 42 | Random seed |
| `--no-cross-conversation-steering` | off | Disable diversity steering (Run A) |
| `--artifacts-dir` | `output/artifacts` | Pre-built artifacts directory |
| `--output` / `-o` | `output/conversations.jsonl` | Output file |

## Configuration

Edit `config/default.yaml` to tune models, sampling weights, quality thresholds, and steering strategy. Environment variables in `.env` override model keys.

## Output Format

Each line in the output JSONL is a conversation record:

```jsonc
{
  "conversation_id": "conv_0042",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint": "hotels/search", "arguments": {"city": "Paris"}}
    ]},
    {"role": "tool", "content": {"results": [...]}},
    {"role": "assistant", "content": "Done."}
  ],
  "judge_scores": {"naturalness": 4.2, "tool_correctness": 4.8, "task_completion": 5.0},
  "metadata": {
    "seed": 42,
    "tools_used": ["hotels/search", "hotels/book"],
    "num_turns": 7,
    "domains": ["Travel"],
    "steering_enabled": true,
    "repair_attempts": 0
  }
}
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=tooluse_gen --cov-report=term-missing

# Specific suites
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

## Project Structure

```
src/tooluse_gen/
├── cli/          # Typer CLI commands (build, generate, evaluate)
├── core/         # Shared config, models, session state
├── registry/     # ToolBench ingestion → clean internal data model
├── graph/        # Tool Graph construction + chain sampler
├── agents/       # Multi-agent conversation generator
├── evaluation/   # LLM-as-judge + retry/repair pipeline
└── utils/        # Logging, serialization helpers
```

See [DESIGN.md](DESIGN.md) for architecture decisions, prompt design, and diversity analysis.
