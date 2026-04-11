# DESIGN.md — tooluse-generator

> Architecture decisions, prompt design, context management, and diversity analysis.

---

## Table of Contents

1. [Architecture & Decisions](#1-architecture--decisions)
2. [Tool Registry Design](#2-tool-registry-design)
3. [Tool Graph + Sampler](#3-tool-graph--sampler)
4. [Offline Execution Model](#4-offline-execution-model)
5. [Multi-Agent System Design](#5-multi-agent-system-design)
6. [Quality Evaluation Pipeline](#6-quality-evaluation-pipeline)
7. [Context Management Design](#7-context-management-design)
8. [Prompt Design](#8-prompt-design)
9. [Diversity & Quality Analysis](#9-diversity--quality-analysis)

---

## 1. Architecture & Decisions

### System Overview

```mermaid
flowchart TD
    CLI["CLI (Typer)"]
    CLI -->|"tooluse build"| BUILD
    CLI -->|"tooluse generate"| GEN
    CLI -->|"tooluse evaluate"| EVAL

    subgraph BUILD ["Build Pipeline"]
        L[ToolBenchLoader] --> R[ToolRegistry]
        R --> GB[GraphBuilder]
        GB --> G["Graph (NetworkX)"]
    end

    subgraph GEN ["Generation Pipeline"]
        S[ToolChainSampler] -->|ToolChain| O[ConversationOrchestrator]
        O --> US[UserSimulator]
        O --> AA[AssistantAgent]
        O --> TE[ToolExecutor]
        O --> C[Conversation]
    end

    subgraph EVAL ["Evaluation Pipeline"]
        V[ConversationValidator] --> J[JudgeAgent]
        J --> RL[RepairLoop]
        RL -->|feedback| O
        J --> ER[EvaluationReport]
    end

    G --> S
    R --> AA
    R --> TE
    C --> V

    subgraph CORE ["Core Infrastructure"]
        CFG[AppConfig]
        CACHE[PromptCache]
        JSONL[JSONLWriter/Reader]
        SEED[SeedManager]
    end
```

### Key Architectural Decisions

| # | Decision | Justification |
|---|----------|---------------|
| 1 | **Lenient ToolBench parsing** — handle 5 JSON formats, log warnings for malformed data | Maximises data ingestion; quality filtering happens later via `QualityTier` |
| 2 | **Two-level graph nodes** — `ToolNode` + `EndpointNode` | Enables both tool-level reasoning (domain edges) and endpoint-level chaining (parameter compatibility) |
| 3 | **MCTS for chain sampling** — UCB1 tree policy with reward shaping | Balances exploration vs exploitation; outperforms random walk for constraint satisfaction |
| 4 | **Offline-first agents** — all agents work with `llm_client=None` | Fast iteration and testing without API keys; LLM mode is opt-in |
| 5 | **Deterministic generation** — all randomness via `np.random.Generator` | Same seed produces identical output; enables reproducible experiments |
| 6 | **Validation-first evaluation** — structural checks before LLM scoring | Catches format errors cheaply; avoids wasting judge calls on broken conversations |
| 7 | **Repair via regeneration** — retry with feedback, not in-place patching | Simpler than turn-level editing; feedback in chain metadata guides the next attempt |
| 8 | **Pydantic v2 throughout** — all models use `BaseModel` with `ConfigDict` | Type safety, JSON serialization, validation, and computed fields in one framework |
| 9 | **NetworkX DiGraph** — tool graph stored as a directed graph | Mature library; supports PageRank, BFS, serialization; edges can be directional |
| 10 | **Diversity as sampling weight adjustment** — not post-hoc filtering | Steers generation toward underrepresented tools during sampling, not after |

### Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Models | Pydantic v2 | Strict validation, `computed_field`, JSON serialization |
| CLI | Typer + Rich | Auto-generated help, rich tables/panels, progress bars |
| Graph | NetworkX | Mature graph algorithms (PageRank, BFS), pickle persistence |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Fast, 384-dim vectors, good semantic similarity |
| Progress | tqdm | Cross-platform, minimal overhead, disable-able |
| Config | YAML + Pydantic | Human-readable config files validated by typed models |
| Seeding | `numpy.random.Generator` | Modern API, reproducible, per-component isolation |

---

## 2. Tool Registry Design

### Data Model

```mermaid
classDiagram
    Tool "1" --> "*" Endpoint
    Endpoint "1" --> "*" Parameter
    Endpoint "1" --> "0..1" ResponseSchema

    class Tool {
        +tool_id: str
        +name: str
        +domain: str
        +completeness_score: float
        +endpoints: list~Endpoint~
    }
    class Endpoint {
        +endpoint_id: str
        +method: HttpMethod
        +path: str
        +parameters: list~Parameter~
        +response_schema: ResponseSchema
    }
    class Parameter {
        +name: str
        +param_type: ParameterType
        +required: bool
        +default: Any
    }
```

### Format Handling

`ToolBenchLoader` (`registry/loader.py`) auto-detects and parses 5 JSON formats:

| Format | Detection | Example |
|--------|-----------|---------|
| `single_tool` | Top-level `api_list` key | Most RapidAPI tools |
| `tool_list` | Top-level array of tools | Bundled tool collections |
| `toolbench_v1` | Has `tool_name` + `api_list` | Standard ToolBench |
| `toolbench_v2` | Has `name` + `endpoints` array | Updated schema |
| `openapi` | Has `openapi` or `swagger` key | OpenAPI 3.x / Swagger |

**Design choice**: lenient mode (default) logs warnings and skips malformed entries. Strict mode raises on the first error. This maximises ingestion from the inconsistent ToolBench corpus.

### Normalization Pipeline

Raw JSON passes through four normalizers in `registry/normalizers.py`:

1. **TextNormalizer** — fix encoding, strip whitespace, normalise identifiers
2. **TypeNormalizer** — map string/int/bool/etc. to `ParameterType` enum
3. **PathNormalizer** — extract path parameters from `{id}`, `:id`, `<id>` syntax
4. **ValueNormalizer** — coerce defaults to correct Python types

### Quality Scoring

`CompletenessCalculator` scores tools on a 0–1 scale based on description quality, parameter documentation, and type annotations. Thresholds:

| Tier | Score | Usage |
|------|-------|-------|
| EXCELLENT | ≥ 0.8 | Preferred for sampling |
| GOOD | ≥ 0.6 | Default inclusion |
| FAIR | ≥ 0.4 | `build` command filter default |
| POOR | ≥ 0.2 | Excluded by default |
| MINIMAL | < 0.2 | Always excluded |

`RegistryBuilder` provides a fluent API:

```python
registry = (
    RegistryBuilder()
    .load_from_directory("data/toolenv/tools")
    .calculate_completeness()
    .filter_by_quality(QualityTier.FAIR)
    .build()
)
```

---

## 3. Tool Graph + Sampler

### Graph Schema

The tool graph is a NetworkX `DiGraph` with two node types and three edge types:

```mermaid
graph LR
    T1["tool:hotels_api<br/>(ToolNode)"]
    E1["ep:hotels_api:search<br/>(EndpointNode)"]
    E2["ep:hotels_api:book<br/>(EndpointNode)"]
    E3["ep:weather_api:current<br/>(EndpointNode)"]

    T1 -->|SAME_TOOL| E1
    T1 -->|SAME_TOOL| E2
    E1 -->|SAME_DOMAIN| E3
    E1 -.->|SEMANTIC_SIMILARITY| E2
```

| Edge Type | Connects | Weight |
|-----------|----------|--------|
| `SAME_TOOL` | Tool → its endpoints | 1.0 (never pruned) |
| `SAME_DOMAIN` | Endpoints in same domain | Quality tier weight (0.1–0.9) |
| `SEMANTIC_SIMILARITY` | Endpoints with cosine > threshold | Cosine similarity score |

`GraphBuilder` (`graph/builder.py`) constructs the graph by adding tool nodes, endpoint nodes, domain edges, and semantic edges (via `EmbeddingService`). Edges are pruned to `max_edges_per_node=50` keeping highest-weight edges; `SAME_TOOL` edges are never pruned.

### MCTS Chain Sampling

`MCTSSampler` (`graph/sampler.py`) uses Monte Carlo Tree Search with UCB1 selection:

```
for iteration in range(max_iterations):
    node = SELECT(root)          # UCB1 tree policy
    child = EXPAND(node, graph)  # add one untried neighbour
    reward = ROLLOUT(child)      # random continuation up to rollout_depth
    BACKPROPAGATE(child, reward) # update visits + rewards to root
```

**Reward function** (per chain):

| Component | Weight | Condition |
|-----------|--------|-----------|
| Step-count bonus | +1.0 | `min_steps ≤ len ≤ max_steps` |
| Multi-tool bonus | +0.5 × n | Per unique tool ID |
| Domain bonus | +0.3 × n | Per unique domain |
| Edge coherence | +0.2 × n | Per consecutive pair with a graph edge |
| Constraint violation | −0.5 × n | Excluded tool used, required tool missing, out of range, too few tools |

**Fallback**: if MCTS exhausts `max_retries` (default 50), a weighted random walk is attempted. If that also fails, `SamplingError` is raised.

### SamplingConstraints

```python
class SamplingConstraints(BaseModel):
    min_steps: int = 2          # minimum endpoints in chain
    max_steps: int = 5          # maximum endpoints
    min_tools: int = 2          # minimum unique tools
    domains: list[str] | None   # restrict to these domains
    required_tools: list[str] | None
    excluded_tools: list[str] | None
    quality_threshold: str = "fair"
```

### Diversity-Aware Sampling

`DiversityTracker` (`graph/diversity.py`) tracks tool and domain usage across a batch. It provides inverse-frequency weights via `get_tool_weight(tool_id)` which decays as `1 / (1 + count × weight_decay)`. The `ToolChainSampler` facade integrates MCTS + pattern detection + diversity tracking in a single `sample_batch()` call. Steering is toggled via `--no-cross-conversation-steering`.

### Chain Patterns

Post-sampling, `PatternDetector` classifies chains:

| Pattern | Description |
|---------|-------------|
| `SEQUENTIAL` | Linear A → B → C (default) |
| `PARALLEL` | Independent steps grouped as `ParallelGroup` |
| `BRANCH_AND_MERGE` | Search → multiple options → single use |
| `ITERATIVE` | Same endpoint repeated (pagination, retry) |

---

## 4. Offline Execution Model

### Value Generation

`ValuePool` (`agents/value_generator.py`) maintains pre-built pools of realistic values by type: city names, hotel names, person names, prices, dates, IDs, URLs, emails, phone numbers, status values, etc. All randomness flows through `np.random.Generator` for determinism.

`SchemaBasedGenerator` infers response structure from endpoint metadata:

| HTTP Method | Path Pattern | Response Structure |
|-------------|-------------|-------------------|
| GET | Plural path (`/hotels`) | List of 2–5 objects with IDs |
| GET | Singular/parameterised | Single object with ID |
| POST | Any | Object with generated ID + status |
| DELETE | Any | Status confirmation |

### Argument Generation

`ArgumentGenerator` (`agents/argument_generator.py`) fills tool call arguments in priority order:

1. **Grounding resolution** — fuzzy-match parameter names against `ConversationContext.grounding_values` from prior tool outputs
2. **Enum/default fallback** — use parameter's enum values or default
3. **Fresh generation** — sample from `ValuePool` based on parameter type and name heuristics

### Grounding Tracking

`GroundingTracker` (`agents/grounding.py`) records the provenance of every extracted value: source endpoint, step index, and field name. `format_available_values(context, tracker)` produces a prompt fragment:

```
Available values from prior tool calls:
- hotel_id: htl_881 (from hotels/search, step 1)
- booking_id: bk_3391 (from hotels/book, step 2)
```

This ensures later tool calls reference real values from earlier outputs — not hallucinated placeholders.

---

## 5. Multi-Agent System Design

### Agent Roles

| Agent | Class | Responsibility |
|-------|-------|---------------|
| **User Simulator** | `UserSimulator` | Generates initial requests, follow-ups, clarification responses |
| **Assistant** | `AssistantAgent` | Selects tools from chain, generates disambiguation questions, final answers |
| **Tool Executor** | `ToolExecutor` | Mock-executes tool calls, extracts grounding values |
| **Judge** | `JudgeAgent` | Scores conversations on 4 quality dimensions |

All agents accept `llm_client=None` for offline (template-based) mode or an OpenAI-compatible client for LLM mode.

### Communication Protocol

`ConversationOrchestrator` (`agents/orchestrator.py`) drives a synchronous loop:

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant U as UserSimulator
    participant A as AssistantAgent
    participant T as ToolExecutor

    O->>U: generate_initial_request(chain)
    U-->>O: user message

    loop Until chain complete or max_turns
        O->>A: generate_response(context)
        alt Disambiguation
            A-->>O: clarifying question
            O->>U: generate_clarification_response()
            U-->>O: clarification
        else Tool Call
            A-->>O: tool_calls list
            loop For each tool call
                O->>T: execute(request, context)
                T-->>O: ToolCallResponse
            end
        else Final Answer
            A-->>O: summary text
        end
    end
```

### State Machine

`ConversationStateMachine` (`agents/state_machine.py`) enforces valid transitions:

| State | Valid Events | Next State |
|-------|-------------|------------|
| `INIT` | `start` | `USER_TURN` |
| `USER_TURN` | `user_message` | `ASSISTANT_TURN` |
| `ASSISTANT_TURN` | `assistant_tool_call` | `TOOL_EXECUTION` |
| `ASSISTANT_TURN` | `assistant_disambiguate` | `DISAMBIGUATION` |
| `ASSISTANT_TURN` | `assistant_final` | `COMPLETE` |
| `TOOL_EXECUTION` | `tool_result` | `ASSISTANT_TURN` |
| `DISAMBIGUATION` | `user_clarification` | `ASSISTANT_TURN` |
| Any | `max_turns_reached`, `error` | `COMPLETE` / `FAILED` |

### Shared State

`ConversationContext` (`agents/execution_models.py`) holds all shared state: messages, tool outputs, grounding values, the driving `ToolChain`, current step index, and metadata. All agents read from and write to the context — the orchestrator mediates access.

---

## 6. Quality Evaluation Pipeline

### Pipeline Flow

```mermaid
flowchart LR
    C[Conversation] --> V{Validator}
    V -->|valid| J[JudgeAgent]
    V -->|invalid| R[RepairLoop]
    J -->|score ≥ threshold| A[Accepted]
    J -->|score < threshold| R
    R -->|regenerate| C
    R -->|max retries| D[Discarded]
```

### Structural Validation

`ConversationValidator` (`evaluation/validator.py`) runs 5 checks:

| Check | What It Validates |
|-------|-------------------|
| `_check_message_structure` | Roles alternate correctly, no empty messages |
| `_check_tool_call_validity` | Endpoint IDs exist in registry, required params present |
| `_check_grounding_consistency` | IDs in step N exist in outputs from step N−1 |
| `_check_minimum_requirements` | At least 1 tool call per conversation |
| `_check_conversation_completeness` | Starts with user, ends with assistant, not truncated |

### Judge Agent

`JudgeAgent` (`evaluation/judge.py`) scores on 4 dimensions (1–5 integer scale):

| Dimension | 1 (worst) | 5 (best) |
|-----------|-----------|----------|
| **tool_correctness** | Completely wrong tools | Perfect tool selection |
| **argument_grounding** | Hallucinated arguments | All arguments grounded in prior outputs |
| **task_completion** | Goal not addressed | Fully completed |
| **naturalness** | Robotic/incoherent | Indistinguishable from real conversation |

**Offline heuristic** (`_score_offline`): base scores + bonuses for tool call count, grounded value matches, message structure, and disambiguation presence. **LLM mode** (`_score_with_llm`): sends the conversation to GPT-4o with a rubric prompt, parses structured JSON scores.

### Repair Loop

`RepairLoop` (`evaluation/repair.py`) implements retry-with-feedback:

1. **Validate** — if structural issues, regenerate with validation errors as feedback
2. **Score** — if quality below `min_score` (default 3.5), regenerate with judge reasoning as feedback
3. **Repeat** — up to `max_retries` (default 3) with incrementing seed
4. **Give up** — mark as failed with `"max_retries_exceeded"`

Feedback is stored in `chain.metadata["repair_feedback"]` so the orchestrator can incorporate it. `RepairStats` tracks attempts, success rate, and per-attempt-number pass distribution.

---

## 7. Context Management Design

Conversation quality depends on two kinds of context management: **within-conversation grounding** (ensuring tool call arguments reference real values from prior outputs) and **cross-conversation steering** (ensuring the generated corpus covers diverse tools and domains).

### 7.1 Within-Conversation Grounding

#### Value Flow

```mermaid
flowchart LR
    TE["ToolExecutor<br/>execute()"] -->|extract values| CC["ConversationContext<br/>grounding_values"]
    CC -->|available values| AG["ArgumentGenerator<br/>_resolve_grounded_value()"]
    AG -->|grounded args| TC["Next ToolCallRequest"]
    CC -->|format_available_values| AP["AssistantAgent<br/>system prompt"]
```

When `ToolExecutor` executes a tool call, it extracts referenceable values (IDs, names, URLs, dates, prices) from the response and stores them in `ConversationContext.grounding_values` with step-prefixed keys (e.g., `step_0.hotel_id`, `step_1.booking_id`). Generated entity IDs also go into `ConversationContext.generated_ids`.

#### Grounding Resolution

`ArgumentGenerator` (`agents/argument_generator.py`) fills arguments in three-priority order:

| Priority | Strategy | Example |
|----------|----------|---------|
| 1. Grounding | `_resolve_grounded_value()` fuzzy-matches param name against `grounding_values` keys | `hotel_id` matches `step_0.hotel_id` → `"htl_881"` |
| 2. Schema | Use endpoint parameter's enum values or default | `currency` → `"EUR"` from enum |
| 3. Fresh | `_generate_fresh_value()` samples from `ValuePool` by type/name heuristic | `city` → `"Paris"` from city pool |

The fuzzy matching in `_match_param_to_grounding()` strips prefixes (`step_N.`) and compares parameter names against value keys, handling common patterns like `hotel_id` matching `id` from a hotel search response.

#### Prompt Injection

`format_available_values()` (`agents/grounding.py`) renders grounding values as a human-readable prompt fragment:

```
Available values from prior tool calls:
- hotel_id: htl_881 (from hotels/search, step 1)
- booking_id: bk_3391 (from hotels/book, step 2)
```

This is injected into the `AssistantAgent`'s system prompt so the LLM knows which real values to use. The structured version (`format_grounding_context()`) provides a dict for function-calling mode.

#### Provenance Tracking

`GroundingTracker` records a `ValueProvenance` for each extracted value:

```python
class ValueProvenance(BaseModel):
    value_key: str          # e.g. "hotel_id"
    value: Any              # e.g. "htl_881"
    source_endpoint: str    # e.g. "hotels/search"
    step_index: int         # e.g. 0
    value_type: str         # e.g. "id"
```

This enables the validator to check grounding consistency: does the `hotel_id` used in step 2 actually appear in step 1's output?

### 7.2 Cross-Conversation Steering

#### Mechanism

`DiversityTracker` (`graph/diversity.py`) maintains running counters across a batch:

| Counter | Purpose |
|---------|---------|
| `tool_counts: Counter[str]` | Per-tool usage frequency |
| `domain_counts: Counter[str]` | Per-domain usage frequency |
| `tool_pair_counts: Counter[tuple]` | Tool-pair co-occurrence |
| `pattern_hashes: set[str]` | Seen tool-combo hashes |

After each successful chain sample, `update(chain)` increments all counters.

#### Weight Adjustment

`get_tool_weight(tool_id)` returns an inverse-frequency weight:

```
weight = 1.0 / (1 + count × weight_decay)
```

With `weight_decay=0.9` (default), a tool used 3 times gets weight `1 / (1 + 2.7) = 0.27`, making it less likely to be selected. This is fed into the MCTS sampler's candidate scoring.

#### Prompt-Level Steering

`build_steering_prompt()` identifies underrepresented domains (below median count) and returns a fragment like:

```
This conversation should use tools from the Finance domain.
```

This can be injected into the `UserSimulator`'s prompt to guide the request toward underutilised areas.

#### Tradeoffs and Limitations

| Aspect | Detail |
|--------|--------|
| **Benefit** | Entropy increased +0.66, +3 unique tools in experiments |
| **Cost** | Mean score decreased −0.25 (less-optimal tool combinations) |
| **Limitation** | Fuzzy matching may miss non-obvious parameter connections (e.g., `property_id` from one API matching `listing_id` in another) |
| **Limitation** | Steering effect depends on graph density — sparse graphs with few cross-domain edges show minimal diversity improvement |
| **Limitation** | Weight decay is global — a tool popular in one domain gets penalised even for conversations in a different domain |
| **Toggle** | `--no-cross-conversation-steering` disables `DiversityTracker` weight adjustment and prompt injection |

---

## 8. Prompt Design

### 8.1 Key Prompts

| Agent | Prompt | Purpose | Mode |
|-------|--------|---------|------|
| UserSimulator | `_build_initial_prompt` | Generate natural user request from chain | LLM |
| UserSimulator | `_build_follow_up_prompt` | Continue conversation after tool results | LLM |
| UserSimulator | `_build_clarification_prompt` | Answer assistant's disambiguating question | LLM |
| UserSimulator | `_DOMAIN_TEMPLATES` | Template-based request generation | Offline |
| AssistantAgent | `_build_system_prompt` | Tool selection + grounding injection | LLM |
| AssistantAgent | `_build_tools_schema` | OpenAI function-calling JSON Schema | LLM |
| JudgeAgent | `_RUBRIC` | Quality scoring rubric | LLM |

### 8.2 UserSimulator Prompts

**Initial request** (`_build_initial_prompt`):

```
System: You are simulating a user who needs help. Generate a single
natural request (1–3 sentences) that would require using these tools.
Do NOT mention tool names or endpoints — describe your need naturally.

Domain(s): Travel
Tools available: Hotels API, Weather API
Tool chain: Hotels API: Search Hotels -> Weather API: Current Weather
```

**Follow-up** (`_build_follow_up_prompt`):

```
System: You are simulating a user in an ongoing conversation. Based on
the conversation so far and the last tool results, generate a natural
follow-up request (1–2 sentences). Stay in character.

Available data from prior tool calls: {"hotel_id": "htl_881", ...}
```

**Offline templates** — when `llm_client=None`, the simulator selects from domain-specific templates in `_DOMAIN_TEMPLATES`:

```python
"Travel": ["I'm planning a trip and need help finding {tool_context}."]
"Finance": ["I need help with some financial information. Can you {tool_context}?"]
# Generic fallback:
"I need help with something. Can you {tool_context}?"
```

The `{tool_context}` placeholder is filled with a description derived from the chain's endpoint names (e.g., "get current weather and search hotels").

### 8.3 AssistantAgent Prompt

```
You are a helpful assistant with access to the following tools:
- Hotels API: Search Hotels — Search for available hotels in a city
- Hotels API: Book Hotel — Book a hotel room

Use them to help the user. When calling a tool, provide all
required arguments.

Available values from prior tool calls:
- hotel_id: htl_881 (from hotels/search, step 1)
```

The assistant also receives an OpenAI function-calling schema (`_build_tools_schema`) with JSON Schema for each endpoint's parameters, enabling structured tool-call output.

### 8.4 JudgeAgent Rubric

```
You are evaluating a synthetic conversation for training data quality.
Score each dimension 1-5:

Tool Selection Correctness:
  1=completely wrong tools, 3=partially correct, 5=perfect tool selection

Argument Grounding:
  1=hallucinated arguments, 3=some grounded, 5=all arguments properly grounded

Task Completion:
  1=goal not addressed, 3=partially achieved, 5=fully completed

Naturalness:
  1=robotic/incoherent, 3=acceptable, 5=indistinguishable from real conversation

Respond with ONLY a JSON object:
{"tool_correctness": <int>, "argument_grounding": <int>,
 "task_completion": <int>, "naturalness": <int>, "reasoning": "<brief>"}
```

**Design rationale**: The rubric uses a 1–5 integer scale (not 0–10 or continuous) because LLMs produce more consistent scores on a small discrete scale. The `"reasoning"` field is required so the judge explains its scores — this explanation is reused as feedback in the repair loop.

### 8.5 Failed Prompt Iterations and Lessons Learned

#### Failure 1: Leaking tool names to the user

**Before**: The UserSimulator prompt included endpoint names directly:

> *"You need to use hotels/search to find hotels, then hotels/book to book one."*

**Problem**: The LLM produced requests like *"Please call hotels/search for Paris"* — completely unnatural. Naturalness scores dropped to 2/5.

**After**: Changed to *"Do NOT mention tool names or endpoints — describe your need naturally."* The chain description is provided for context but the LLM translates it into natural language.

**Lesson**: When simulating a user, the prompt must enforce a clear separation between system context (what tools exist) and user behaviour (natural language requests). Explicit negative instructions ("do NOT mention") are more effective than positive ones ("speak naturally").

#### Failure 2: Free-text judge responses

**Before**: The JudgeAgent rubric ended with *"Provide your scores and reasoning."* without specifying format.

**Problem**: Scores appeared in inconsistent formats — `"4/5"`, `"four"`, `"tool_correctness: 4"`, sometimes embedded in paragraphs. Parsing failed on ~30% of responses.

**After**: Changed to *"Respond with ONLY a JSON object"* with an exact format template. Added a regex fallback parser (`_parse_scores`) that extracts `"key": value` patterns when JSON parsing fails.

**Lesson**: LLM-as-judge prompts must enforce structured output. "Only JSON" plus a template reduces parsing failures from ~30% to <5%. A fallback parser handles the remaining edge cases without discarding valid scores.

### 8.6 Prompt Structure Rationale

| Decision | Rationale |
|----------|-----------|
| System prompt for context, user message for instructions | Separates persistent state (tools, grounding) from per-turn directives |
| Grounding values in system prompt (not user message) | Available values are context, not user speech — keeps user messages natural |
| 500-char truncation on available data | Prevents context-window overflow in follow-up prompts |
| Domain-specific template pools | Produces more natural requests than a single generic template |
| Function-calling schema for tool selection | Structured output is more reliable than asking the LLM to emit JSON manually |

---

## 9. Diversity & Quality Analysis

*(To be filled in during Task 74.)*
