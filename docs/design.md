# Spindle — Design Document

## Philosophy

Spindle is a Python framework for building multi-step LLM agents where **you control the loop, not the framework**.

Existing frameworks (ADK, LangChain, LangGraph) treat the agent turn as an atomic black box — you push a message in and observe events coming out. The internal components (session, events, tools) are implementation details wired together inside the runner, not independent building blocks you can use on their own.

Spindle inverts this. Every component is a first-class citizen:

- **Session** is yours to read, write, seed, and flush
- **Events** have builders — you create them, not the framework
- **Tools** are callable by the LLM, by your router, by your application code, or by your tests
- **The Runner** is a lazy generator (cold observable) — it does zero work until you pull the next step
- **Persistence** is explicit — you decide when to flush, not the framework

Each piece is useful on its own. Session without runner. Tools without LLM. Runner without persistence.

---

## Core Abstractions

### Event

The atomic unit of conversation. Every interaction — user message, agent response, tool call, tool result — is an Event.

```python
event = Event.user_message("hello")
event = Event.agent_message("Hi! How can I help?")
event = Event.tool_call("classify", {"url": "..."})
event = Event.tool_result("classify", {"type": "invoice", "confidence": 0.97}, call_id="...")
```

Events are Pydantic models. They are immutable after creation. Each has an auto-generated ID and timestamp.

**Fields:**
- `id: str` — UUID, auto-generated
- `timestamp: float` — Unix timestamp, auto-generated
- `role: EventRole` — `user`, `agent`, `tool`
- `type: EventType` — `message`, `tool_call`, `tool_result`
- `author: str` — "user" or agent name
- `content: str | None` — Text content for messages
- `tool_name: str | None` — For tool_call and tool_result
- `tool_args: dict | None` — For tool_call
- `tool_result_data: Any | None` — For tool_result
- `tool_call_id: str | None` — Links tool_result to tool_call
- `metadata: dict | None` — Arbitrary key-value pairs

### Session

An ordered sequence of events + key-value state. The session is an in-memory buffer that you explicitly flush to persistence.

```python
session = Session(id="s1", user_id="u1")

# Add events — nothing hits the database
session.add_user_message("hello")
session.add_agent_message("Hi there!")

# State management
session.state["timezone"] = "Asia/Riyadh"

# Flush to persistence when YOU decide
await session.flush(store)

# Or never flush — works entirely in-memory for tests
```

**Key properties:**
- `events: list[Event]` — Full event history (flushed + pending)
- `pending_events: list[Event]` — Events not yet flushed
- `state: dict[str, Any]` — Mutable key-value state

**Key methods:**
- `add_user_message(text) -> Event`
- `add_agent_message(text, author=None) -> Event`
- `add_tool_call(name, args, author=None) -> Event`
- `add_tool_result(name, result, call_id) -> Event`
- `flush(store) -> None` — Persist pending events + state
- `history(roles=None, types=None) -> list[Event]` — Filtered view

### Store (Persistence)

Abstract interface for event persistence. Session delegates to store on flush.

```python
class Store(ABC):
    async def save_session(self, session: Session) -> None: ...
    async def load_session(self, session_id: str) -> Session | None: ...
    async def append_events(self, session_id: str, events: list[Event]) -> None: ...
    async def save_state(self, session_id: str, state: dict) -> None: ...
```

**Built-in implementations:**
- `MemoryStore` — Dict-backed, for tests
- `PostgresStore` — Async PostgreSQL (future)

### Tool

A callable function with metadata. Created via decorator.

```python
@tool
async def search_products(query: str, limit: int = 10) -> list[dict]:
    """Search the product catalog.

    Args:
        query: Search query string
        limit: Maximum results to return
    """
    results = await catalog.search(query, limit=limit)
    return [{"name": r.name, "price": r.price} for r in results]
```

The decorator extracts:
- Name from function name
- Description from docstring
- Parameters from type hints + docstring
- JSON Schema for the parameters (for LLM function calling)

Tools are plain async functions. You can call them directly:
```python
results = await search_products("laptop", limit=5)
```

Or inject their results into a session without LLM:
```python
results = await search_products("laptop")
session.add_tool_call("search_products", {"query": "laptop"})
session.add_tool_result("search_products", results, call_id=...)
```

### LLM

Abstract interface for language model providers.

```python
class LLM(ABC):
    @abstractmethod
    async def generate(
        self,
        history: list[Event],
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        config: GenerateConfig | None = None,
    ) -> LLMResponse: ...
```

**`GenerateConfig`:**
- `temperature: float | None`
- `max_tokens: int | None`
- `stop_sequences: list[str] | None`
- `thinking: ThinkingConfig | None` — Enable/disable thinking, set budget
- `provider_config: dict | None` — Pass-through for provider-specific settings

**`LLMResponse`:**
- `content: str | None` — Text response
- `tool_calls: list[ToolCall] | None` — Requested tool calls
- `usage: UsageMetadata | None` — Token counts
- `thinking: str | None` — Thinking/reasoning content
- `model: str | None` — Model that generated the response

**Built-in providers:**
- `GeminiLLM` — Google Gemini via `google-genai` SDK (Vertex AI)

The LLM adapter converts between Spindle's Event format and the provider's message format. This conversion is internal to the provider — the caller only works with Events.

### Router

Application-level routing that runs before the LLM. Handles messages deterministically when the LLM isn't needed.

```python
router = Router()

@router.route
async def greet(ctx: RouteContext) -> bool:
    """Handle greetings without LLM."""
    if not re.match(r"^(hello|hi|hey|merhaba)", ctx.message, re.IGNORECASE):
        return False  # not handled, pass to LLM

    response = random.choice(["Hi!", "Hello!", "Hey there!"])
    ctx.session.add_agent_message(response)
    return True  # handled, skip LLM

@router.route
async def handle_receipt_upload(ctx: RouteContext) -> bool:
    """Pre-process receipt uploads with OCR before LLM."""
    if not ctx.has_attachment(type="receipt"):
        return False

    # Run OCR directly — no LLM needed
    result = await ocr_model.extract(ctx.attachment_url)
    ctx.session.add_tool_call("extract_receipt", {"url": ctx.attachment_url})
    ctx.session.add_tool_result("extract_receipt", result)
    return False  # pass to LLM WITH the OCR results already in context
```

Routes return `True` (handled, skip LLM) or `False` (not handled or enriched, continue to LLM). Routes run in registration order. First `True` wins.

### Runner

The cold observable. A lazy async generator that executes the agent loop one step at a time.

```python
async for step in runner.run(session, "What products do you have?"):
    match step.type:
        case StepType.LLM_RESPONSE:
            print(step.content)
        case StepType.TOOL_CALL:
            print(f"Calling {step.tool_name}({step.tool_args})")
        case StepType.TOOL_RESULT:
            print(f"Result: {step.result}")
```

**Pull-based execution:**
- Each `yield` is a checkpoint — the runner does zero work until you pull the next value
- Between pulls, you can inject messages, flush the session, or stop
- Stopping is just `break` — no cancellation tokens, no callbacks

**Mid-run injection:**
```python
async for step in runner.run(session, "find cheapest supplier"):
    if step.type == StepType.TOOL_RESULT:
        if user_input := await check_for_user_input():
            session.add_user_message(user_input)
            # Runner sees this on next iteration
    yield step
```

**Per-message controls:**
```python
runner.run(
    session,
    "explain this error",
    system_prompt="You are a debugging expert.",  # override for this message
    tools=[debug_tool, log_search],               # override tools for this message
    config=GenerateConfig(thinking=ThinkingConfig(enabled=True, budget=8192)),
)
```

### Agent

Composition of LLM + tools + router + instruction. The agent defines behavior, the runner executes it.

```python
agent = Agent(
    name="assistant",
    llm=GeminiLLM(model="gemini-2.5-flash"),
    instruction="You are a helpful inventory assistant.",
    tools=[search_products, create_order, check_stock],
    router=router,
)

# Sub-agents
receipt_agent = Agent(
    name="receipt_processor",
    llm=GeminiLLM(model="gemini-2.5-pro"),
    instruction="You extract and validate receipt data.",
    tools=[extract_receipt, validate_receipt],
)

agent.add_sub_agent(receipt_agent)
```

**Instruction** can be static or dynamic:
```python
# Static
Agent(instruction="You are a helpful assistant.")

# Dynamic — evaluated per message
Agent(instruction=lambda ctx: f"You are a helpful assistant. The time is {ctx.state['timezone']}.")
```

---

## Component Interaction

```
User Message
     │
     ▼
┌─────────┐     ┌──────────────┐
│  Router  │────▶│   Session    │◀──── You (add events, state, flush)
│         │     │              │
│ greet?  │     │  events[]    │
│ upload? │     │  state{}     │
│ enrich? │     │  pending[]   │
└────┬────┘     └──────┬───────┘
     │ not handled      │
     ▼                  │
┌─────────┐             │
│   LLM   │◀────────────┘ (history)
│         │
│ Gemini  │
│ OpenAI  │──── tool calls? ──┐
│ Claude  │                   │
└────┬────┘                   ▼
     │ text              ┌─────────┐
     │                   │  Tools  │
     ▼                   │         │
┌─────────┐              │ search  │
│  Event  │◀─────────────│ classify│
│ (yield) │  tool result │ extract │
└─────────┘              └─────────┘
     │
     ▼
  Consumer (your code)
  - flush?
  - inject?
  - stop?
  - next?
```

### Runner Loop (Pseudocode)

```
def run(session, message):
    session.add_user_message(message)

    # 1. Route
    handled = router.try_route(session, message)
    if handled:
        yield Step(type=ROUTE_HANDLED)
        return

    # 2. Agent loop
    while True:
        response = llm.generate(
            history=session.events,
            system_prompt=agent.instruction,
            tools=agent.tools,
        )

        if response.content:
            event = session.add_agent_message(response.content)
            yield Step(type=LLM_RESPONSE, event=event)

        if not response.tool_calls:
            break  # turn complete

        for call in response.tool_calls:
            call_event = session.add_tool_call(call.name, call.args)
            yield Step(type=TOOL_CALL, event=call_event)

            result = await tools[call.name].execute(call.args)
            result_event = session.add_tool_result(call.name, result, call_id=call.id)
            yield Step(type=TOOL_RESULT, event=result_event)

        # Check for injected user messages before next LLM call
        # (consumer may have added messages between yields)
```

---

## Testing

Spindle follows the **test trophy** — integration tests are the priority.

### Layer isolation makes testing natural

```python
# Test greeting router — no LLM, no persistence
async def test_greeting_skips_llm():
    session = Session(id="s1", user_id="u1")
    ctx = RouteContext(session=session, message="hello")
    handled = await greet(ctx)
    assert handled is True
    assert session.events[-1].content in ["Hi!", "Hello!", "Hey there!"]

# Test tool result summarization — seed session, assert LLM behavior
async def test_llm_summarizes_tool_result():
    session = Session(id="s1", user_id="u1")
    session.add_user_message("search for laptops")
    session.add_tool_call("search", {"query": "laptops"})
    session.add_tool_result("search", [{"name": "MacBook", "price": 999}])

    response = await llm.generate(session.events, system_prompt="Summarize search results.")
    assert "MacBook" in response.content

# Test full flow — mock LLM, real everything else
async def test_agent_calls_tool_and_responds():
    mock_llm = MockLLM(responses=[
        LLMResponse(tool_calls=[ToolCall(name="search", args={"q": "test"})]),
        LLMResponse(content="Found 3 results."),
    ])
    agent = Agent(name="test", llm=mock_llm, tools=[search])
    runner = Runner(agent=agent)
    session = Session(id="s1", user_id="u1")

    steps = [step async for step in runner.run(session, "search for test")]
    assert steps[-1].content == "Found 3 results."
    assert len(session.events) == 5  # user + tool_call + tool_result + agent + user_message

# Test persistence — real session + memory store
async def test_flush_persists_events():
    store = MemoryStore()
    session = Session(id="s1", user_id="u1")
    session.add_user_message("hello")
    session.add_agent_message("hi")

    await session.flush(store)
    assert len(session.pending_events) == 0

    loaded = await store.load_session("s1")
    assert len(loaded.events) == 2
```

### E2E tests with real LLM

Tagged separately, run selectively:
```python
@pytest.mark.e2e
async def test_gemini_tool_calling():
    llm = GeminiLLM(model="gemini-2.5-flash")
    ...
```

---

## Module Structure

```
spindle/
├── __init__.py          # Public API: Agent, Runner, Session, tool, Router
├── types.py             # Enums, configs, shared types
├── event.py             # Event model with factory methods
├── session.py           # Session with buffer + flush
├── tool.py              # @tool decorator, Tool class, parameter extraction
├── router.py            # Router + RouteContext
├── runner.py            # Runner (lazy async generator)
├── agent.py             # Agent composition
├── llm/
│   ├── __init__.py      # LLM, LLMResponse, GenerateConfig
│   ├── base.py          # Abstract LLM interface
│   └── gemini.py        # Gemini provider via google-genai
└── stores/
    ├── __init__.py      # Store, MemoryStore
    ├── base.py          # Abstract Store interface
    └── memory.py        # In-memory store for tests
```

---

## Implementation Order

1. **types.py** — Enums and configs (no dependencies)
2. **event.py** — Event model with builders (depends on types)
3. **session.py** — Session with buffer (depends on event)
4. **stores/** — Store interface + MemoryStore (depends on session, event)
5. **tool.py** — Tool decorator and registry (no dependencies)
6. **llm/** — LLM interface + Gemini provider (depends on event, tool, types)
7. **router.py** — Router (depends on session)
8. **runner.py** — Runner generator (depends on all above)
9. **agent.py** — Agent composition (depends on all above)
10. **__init__.py** — Public API exports
