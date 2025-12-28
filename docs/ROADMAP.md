# LLM Council Roadmap & Architecture

> Comprehensive development roadmap and architectural improvements for the LLM Council project.

## Table of Contents

- [Current Architecture](#current-architecture)
- [Priority 1: Core Experience](#priority-1-core-experience-improvements)
- [Priority 2: Architecture](#priority-2-architecture-enhancements)
- [Priority 3: Frontend UX](#priority-3-frontend-ux)
- [Priority 4: Security](#priority-4-security--reliability)
- [Priority 5: Analytics](#priority-5-analytics--insights)
- [Priority 6: Testing](#priority-6-testing-infrastructure)
- [Priority 7: New Features](#priority-7-new-features)
- [Implementation Timeline](#implementation-roadmap)
- [Proposed Structure](#proposed-file-structure)

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React/Vite)                     │
│  App.jsx → ChatInterface → Stage1/Stage2/Stage3 Components       │
│                         Port: 5173                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │ REST API + SSE Streaming
┌─────────────────────────▼───────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  main.py → council.py → openrouter.py                           │
│                         Port: 8001                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│              Storage (JSON) + OpenRouter API                     │
│  data/conversations/*.json → OpenRouter (multi-model)            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query
    ↓
Stage 1: Parallel queries → [individual responses]
    ↓
Stage 2: Anonymize → Parallel ranking queries → [evaluations + parsed rankings]
    ↓
Aggregate Rankings Calculation → [sorted by avg position]
    ↓
Stage 3: Chairman synthesis with full context
    ↓
Return: {stage1, stage2, stage3, metadata}
    ↓
Frontend: Display with tabs + validation UI
```

---

## Priority 1: Core Experience Improvements

### 1.1 UI-Based Model Configuration

| Current | Proposed |
|---------|----------|
| Models hardcoded in `config.py` | Settings panel in UI |

**Features:**
- Add/remove council models dynamically
- Drag-and-drop model ordering
- Chairman selection dropdown
- Model presets (fast, thorough, budget)
- Persist to `~/.llmcouncil/config.json`

**Implementation:**
```python
# backend/api/routes/settings.py
@router.get("/models")
async def get_available_models():
    """Fetch available models from OpenRouter."""
    pass

@router.put("/council")
async def update_council_config(config: CouncilConfig):
    """Update council model configuration."""
    pass
```

### 1.2 Streaming Responses

| Current | Proposed |
|---------|----------|
| Batch loading per stage | Token-by-token streaming |

**Features:**
- OpenRouter streaming API integration
- Real-time token display per model in Stage 1
- Progressive Stage 3 synthesis display
- WebSocket upgrade for bidirectional communication

**Implementation:**
```python
# backend/core/streaming.py
async def stream_model_response(model: str, messages: list):
    """Stream tokens from a single model."""
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload) as response:
            async for chunk in response.aiter_lines():
                yield parse_sse_chunk(chunk)
```

### 1.3 Multi-Turn Conversations

| Current | Proposed |
|---------|----------|
| Each query is stateless | Full conversation context |

**Features:**
- Pass conversation history to models
- Sliding window context (last N messages)
- Per-model context pruning for token limits
- "Continue this answer" capability

---

## Priority 2: Architecture Enhancements

### 2.1 Storage Layer Upgrade

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| SQLite | Simple, file-based, queryable | Single-writer | **v1.x** |
| PostgreSQL | Full ACID, concurrent | Requires setup | **v2.0+** |
| Redis + S3 | Fast cache + durable | Complexity | Future |

**Schema Design:**
```sql
-- conversations table
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at DATETIME,
    updated_at DATETIME,
    config_snapshot JSON
);

-- messages table
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    role TEXT CHECK(role IN ('user', 'assistant')),
    content TEXT,
    stage1 JSON,
    stage2 JSON,
    stage3 JSON,
    metadata JSON,
    created_at DATETIME
);

-- model_metrics table
CREATE TABLE model_metrics (
    id INTEGER PRIMARY KEY,
    message_id INTEGER REFERENCES messages(id),
    model_id TEXT,
    response_time_ms INTEGER,
    token_count INTEGER,
    ranking_position REAL,
    cost_usd REAL,
    created_at DATETIME
);
```

### 2.2 Async Task Queue

```
User → API → Redis Queue → Worker Pool → Webhook/SSE
```

**Components:**
- **Queue**: Redis with ARQ or Celery
- **Workers**: Dedicated model query workers
- **Progress**: Real-time stage tracking
- **Retries**: Exponential backoff per model

**Implementation:**
```python
# backend/workers/tasks.py
from arq import create_pool

async def run_council_task(ctx, query: str, conversation_id: str):
    """Background task for council processing."""
    redis = ctx['redis']

    # Stage 1
    await redis.publish(f"progress:{conversation_id}", "stage1_start")
    stage1 = await stage1_collect_responses(query)
    await redis.publish(f"progress:{conversation_id}", json.dumps({
        "type": "stage1_complete",
        "data": stage1
    }))
    # ... continue stages
```

### 2.3 Caching Layer

**Cache Keys:**
```
response:{query_hash}:{model}:{timestamp_bucket}
ranking:{response_hashes}:{evaluator_model}
aggregate:{ranking_hashes}
```

**TTL Strategy:**
| Cache Type | TTL | Invalidation |
|------------|-----|--------------|
| Model responses | 1 hour | On model update |
| Rankings | 30 min | On response change |
| Aggregates | 15 min | On ranking change |

---

## Priority 3: Frontend UX

### 3.1 Enhanced Visualization

**Model Comparison Cards:**
```jsx
<ModelCard
  model="openai/gpt-5.1"
  metrics={{
    responseTime: 1250,
    tokenCount: 847,
    avgRanking: 1.5,
    cost: 0.0034
  }}
  response={response}
/>
```

**Features:**
- Radar chart for multi-criteria evaluation
- Timeline view of stage progression
- Token usage per model display
- Cost estimation before submit

### 3.2 Export Features

| Format | Content | Use Case |
|--------|---------|----------|
| Markdown | All stages, formatted | Documentation |
| PDF | Branded report | Sharing |
| JSON | Raw data | Programmatic |
| Share Link | Hosted snapshot | Collaboration |

### 3.3 Theming

**Dark Mode Implementation:**
```css
:root {
  --bg-primary: #ffffff;
  --text-primary: #1a1a1a;
  --accent: #4a90e2;
}

[data-theme="dark"] {
  --bg-primary: #1a1a1a;
  --text-primary: #e5e5e5;
  --accent: #5a9ff2;
}
```

---

## Priority 4: Security & Reliability

### 4.1 API Security

| Feature | Implementation |
|---------|----------------|
| Rate limiting | 100 req/min per IP |
| API keys | JWT with refresh tokens |
| Request signing | HMAC-SHA256 |
| Audit logging | Structured JSON logs |

### 4.2 Circuit Breaker Pattern

```python
class ModelCircuitBreaker:
    """Circuit breaker for model API calls."""

    def __init__(self, model: str, failure_threshold: int = 3):
        self.model = model
        self.failures = 0
        self.threshold = failure_threshold
        self.state = "closed"  # closed | open | half-open
        self.last_failure = None

    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise CircuitOpenError(self.model)

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

### 4.3 Fallback Chains

```yaml
fallback_chains:
  openai/gpt-5.1:
    - openai/gpt-4o
    - anthropic/claude-sonnet-4.5

  google/gemini-3-pro:
    - google/gemini-2.5-flash
    - openai/gpt-4o
```

---

## Priority 5: Analytics & Insights

### 5.1 Model Performance Tracking

**Metrics Collected:**
```python
@dataclass
class ModelMetrics:
    model_id: str
    query_id: str
    response_time_ms: int
    token_count: int
    input_tokens: int
    output_tokens: int
    ranking_position: float
    cost_usd: Decimal
    error_type: Optional[str]
    timestamp: datetime
```

**Dashboard Views:**
- Historical ranking trends per model
- Cost analysis over time
- Response time percentiles (p50, p95, p99)
- Error rates by model
- Quality correlation with response time

### 5.2 Custom Evaluation Criteria

| Current | Proposed |
|---------|----------|
| Accuracy + Insight only | Configurable criteria |

**Configuration:**
```yaml
evaluation_criteria:
  default:
    - accuracy: 0.4
    - insight: 0.3
    - clarity: 0.2
    - brevity: 0.1

  coding:
    - correctness: 0.5
    - efficiency: 0.25
    - readability: 0.25

  creative:
    - originality: 0.4
    - coherence: 0.3
    - engagement: 0.3
```

---

## Priority 6: Testing Infrastructure

### 6.1 Test Structure

```
tests/
├── unit/
│   ├── test_council.py          # Stage logic
│   ├── test_openrouter.py       # API client
│   ├── test_storage.py          # CRUD operations
│   └── test_ranking_parser.py   # Parsing logic
├── integration/
│   ├── test_full_flow.py        # E2E stages
│   ├── test_streaming.py        # SSE handling
│   └── test_concurrent.py       # Parallel requests
├── fixtures/
│   ├── mock_responses.json      # Canned responses
│   └── sample_queries.json      # Test queries
└── conftest.py                  # Shared fixtures
```

### 6.2 Key Test Cases

**Unit Tests:**
```python
# tests/unit/test_council.py
class TestRankingParser:
    def test_parse_standard_format(self):
        text = """Analysis...
        FINAL RANKING:
        1. Response C
        2. Response A
        3. Response B"""
        assert parse_ranking_from_text(text) == [
            "Response C", "Response A", "Response B"
        ]

    def test_parse_missing_header(self):
        # Fallback behavior
        pass

    def test_parse_malformed_list(self):
        # Edge case handling
        pass
```

**Integration Tests:**
```python
# tests/integration/test_full_flow.py
@pytest.mark.asyncio
async def test_complete_council_flow(mock_openrouter):
    query = "What is the meaning of life?"
    stage1, stage2, stage3, metadata = await run_full_council(query)

    assert len(stage1) == len(COUNCIL_MODELS)
    assert len(stage2) == len(COUNCIL_MODELS)
    assert stage3["model"] == CHAIRMAN_MODEL
    assert "aggregate_rankings" in metadata
```

### 6.3 Load Testing

**Locust Configuration:**
```python
# tests/load/locustfile.py
class CouncilUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def submit_query(self):
        self.client.post("/api/conversations/{id}/message", json={
            "content": "Explain quantum computing"
        })

    @task(1)
    def list_conversations(self):
        self.client.get("/api/conversations")
```

---

## Priority 7: New Features

### 7.1 Reasoning Model Support

**Special Handling:**
```python
REASONING_MODELS = ["openai/o1", "openai/o3", "deepseek/r1"]

async def query_reasoning_model(model: str, messages: list):
    """Handle reasoning models with extended thinking."""
    response = await query_model(model, messages, timeout=300.0)

    return {
        "content": response.get("content"),
        "reasoning": response.get("reasoning_details"),
        "thinking_tokens": response.get("thinking_tokens", 0)
    }
```

**UI Display:**
- Collapsible reasoning chain
- Token count for thinking vs output
- Reasoning visualization graph

### 7.2 Multi-Modal Council

**Supported Inputs:**
| Type | Processing | Models |
|------|------------|--------|
| Image | Base64 encoding | GPT-5V, Gemini Vision |
| PDF | Text extraction + images | All text models |
| Audio | Whisper transcription | All text models |

### 7.3 Council Presets

```yaml
# presets.yaml
presets:
  coding:
    name: "Coding Council"
    models:
      - anthropic/claude-sonnet-4.5
      - openai/gpt-5.1
      - google/gemini-3-pro
    chairman: anthropic/claude-sonnet-4.5
    criteria:
      - correctness: 0.5
      - efficiency: 0.25
      - readability: 0.25

  research:
    name: "Research Council"
    models:
      - openai/gpt-5.1
      - google/gemini-3-pro
      - perplexity/sonar-pro
    chairman: google/gemini-3-pro
    criteria:
      - accuracy: 0.4
      - depth: 0.3
      - sources: 0.3

  creative:
    name: "Creative Council"
    models:
      - anthropic/claude-sonnet-4.5
      - openai/gpt-5.1
      - x-ai/grok-4
    chairman: anthropic/claude-sonnet-4.5
    criteria:
      - originality: 0.4
      - coherence: 0.3
      - engagement: 0.3
```

### 7.4 API Mode & SDK

**REST API:**
```
POST /api/v1/council/query
GET  /api/v1/council/status/{task_id}
GET  /api/v1/council/result/{task_id}
```

**Python SDK:**
```python
from llm_council import Council

council = Council(api_key="...")
result = await council.query(
    "Explain quantum entanglement",
    preset="research",
    stream=True
)

async for stage in result:
    print(f"Stage {stage.number}: {stage.status}")
```

---

## Implementation Roadmap

### Phase 1: v1.1 - Quick Wins (2-3 weeks)

| Feature | Effort | Impact |
|---------|--------|--------|
| UI model configuration | Medium | High |
| Dark mode | Low | Medium |
| Export to Markdown | Low | Medium |
| SQLite storage | Medium | High |

### Phase 2: v1.2 - Core Improvements (4-6 weeks)

| Feature | Effort | Impact |
|---------|--------|--------|
| Streaming responses | High | High |
| Multi-turn conversations | Medium | High |
| Testing infrastructure | Medium | High |
| Circuit breaker pattern | Medium | Medium |

### Phase 3: v2.0 - Scale & Features (6-8 weeks)

| Feature | Effort | Impact |
|---------|--------|--------|
| PostgreSQL migration | Medium | High |
| Task queue (ARQ/Celery) | High | High |
| Analytics dashboard | High | Medium |
| API mode + SDK | High | High |

### Phase 4: v3.0 - Advanced (8-12 weeks)

| Feature | Effort | Impact |
|---------|--------|--------|
| Multi-modal support | High | High |
| Reasoning visualization | Medium | Medium |
| Custom evaluation criteria | Medium | High |
| Plugin system | High | Medium |

---

## Proposed File Structure

```
llm-council/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── conversations.py    # Conversation CRUD
│   │   │   ├── models.py           # Model management
│   │   │   ├── settings.py         # Configuration
│   │   │   └── analytics.py        # Metrics endpoints
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py             # Authentication
│   │   │   ├── rate_limit.py       # Rate limiting
│   │   │   └── logging.py          # Request logging
│   │   └── deps.py                 # Dependency injection
│   ├── core/
│   │   ├── __init__.py
│   │   ├── council.py              # Stage orchestration
│   │   ├── streaming.py            # SSE/WebSocket
│   │   ├── cache.py                # Response caching
│   │   └── circuit_breaker.py      # Fault tolerance
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py               # SQLAlchemy models
│   │   ├── session.py              # Database session
│   │   └── migrations/             # Alembic migrations
│   ├── services/
│   │   ├── __init__.py
│   │   ├── openrouter.py           # API client
│   │   ├── analytics.py            # Metrics collection
│   │   └── export.py               # Export generation
│   ├── workers/
│   │   ├── __init__.py
│   │   └── tasks.py                # Background tasks
│   ├── config.py                   # Configuration
│   └── main.py                     # FastAPI app
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface/
│   │   │   ├── Stage1/
│   │   │   ├── Stage2/
│   │   │   ├── Stage3/
│   │   │   ├── Sidebar/
│   │   │   ├── Settings/
│   │   │   └── Analytics/
│   │   ├── hooks/
│   │   │   ├── useConversation.js
│   │   │   ├── useStreaming.js
│   │   │   └── useSettings.js
│   │   ├── stores/
│   │   │   ├── conversationStore.js
│   │   │   └── settingsStore.js
│   │   ├── utils/
│   │   │   ├── api.js
│   │   │   ├── export.js
│   │   │   └── theme.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── tests/
│   │   ├── components/
│   │   └── e2e/
│   └── package.json
├── sdk/
│   └── python/
│       ├── llm_council/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   └── models.py
│       ├── tests/
│       └── pyproject.toml
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── load/
│   └── fixtures/
├── docs/
│   ├── ROADMAP.md                  # This file
│   ├── API.md                      # API documentation
│   ├── DEVELOPMENT.md              # Dev setup guide
│   └── ARCHITECTURE.md             # Technical deep-dive
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── pyproject.toml
├── package.json
└── README.md
```

---

## Quick Reference

### Commands

```bash
# Development
./start.sh                    # Start both servers
uv run pytest                 # Run tests
npm run dev                   # Frontend only
uv run python -m backend.main # Backend only

# Testing
uv run pytest tests/unit      # Unit tests
uv run pytest tests/integration # Integration tests
npm run test                  # Frontend tests

# Production
docker-compose up -d          # Docker deployment
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes |
| `DATABASE_URL` | Database connection string | v2.0+ |
| `REDIS_URL` | Redis connection for queue | v2.0+ |
| `SECRET_KEY` | JWT signing key | v2.0+ |

---

*Generated by Claude Flow swarm analysis*
