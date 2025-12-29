# LLM Council Roadmap

Project evolution tracker for the LLM Council deliberation system.

## Project Vision

A multi-LLM deliberation platform where AI models collaborate through anonymized peer review to produce higher-quality answers than any single model.

---

## Completed Phases

### Phase 0: Foundation (v0)
**Status:** ✅ Complete

- [x] Core 3-stage deliberation pipeline
- [x] Stage 1: Parallel model queries
- [x] Stage 2: Anonymized peer ranking
- [x] Stage 3: Chairman synthesis
- [x] React frontend with tab views
- [x] FastAPI backend with OpenRouter integration
- [x] JSON file storage
- [x] Basic conversation management

### Phase 1: Production Ready
**Status:** ✅ Complete

- [x] SQLite database storage
- [x] Settings UI for model configuration
- [x] Dark/light theme support
- [x] Export conversations (Markdown/JSON)
- [x] Vercel frontend deployment
- [x] Railway backend deployment
- [x] OpenRouter model identifier fixes

### Phase 2: Reliability & UX
**Status:** ✅ Complete

- [x] Token-by-token streaming (`/stream-tokens` endpoint)
- [x] Multi-turn conversation context (history passed to models)
- [x] Circuit breaker pattern for fault tolerance
- [x] Testing infrastructure (pytest with mocks)
- [x] React state management fixes (spinner, follow-ups)

---

## Upcoming Phases

### Phase 3: Analytics & Insights
**Status:** 🔜 Next

**Goal:** Understand model performance and council dynamics over time.

| Feature | Description | Priority |
|---------|-------------|----------|
| Response timing | Track latency per model per stage | High |
| Token usage | Count tokens consumed per query | High |
| Ranking analytics | Visualize which models rank highest | Medium |
| Cost estimation | Estimate API costs per conversation | Medium |
| Performance dashboard | Aggregate stats over time | Low |

**Technical Notes:**
- Store metrics in SQLite alongside conversations
- Add `/api/analytics` endpoints
- Frontend dashboard component with charts

### Phase 4: Advanced Deliberation
**Status:** 📋 Planned

**Goal:** Enhance the deliberation process with more sophisticated interactions.

| Feature | Description | Priority |
|---------|-------------|----------|
| Custom ranking criteria | User-defined evaluation dimensions | High |
| Weighted voting | Give certain models more influence | Medium |
| Reasoning model support | Special handling for o1, o3 models | Medium |
| Dissent tracking | Highlight disagreements between models | Low |
| Confidence scores | Models rate their own confidence | Low |

### Phase 5: Model Management
**Status:** 📋 Planned

**Goal:** Better control over model selection and behavior.

| Feature | Description | Priority |
|---------|-------------|----------|
| Model presets | Save/load council configurations | High |
| Per-model parameters | Temperature, max_tokens per model | Medium |
| Model discovery | Browse available OpenRouter models | Medium |
| Rate limiting | Prevent API overuse | Medium |
| Fallback models | Auto-substitute when models fail | Low |

### Phase 6: Collaboration
**Status:** 💭 Future

**Goal:** Enable sharing and collaboration features.

| Feature | Description | Priority |
|---------|-------------|----------|
| Share conversations | Public links to council sessions | Medium |
| Embed widget | Embed council in other sites | Low |
| API access | Programmatic access to council | Low |
| Team workspaces | Multi-user support | Low |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Sidebar │  │  Chat   │  │Settings │  │Analytics│    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP/SSE
┌─────────────────────────┴───────────────────────────────┐
│                   Backend (FastAPI)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Council │  │ Storage │  │Streaming│  │ Circuit │    │
│  │  Logic  │  │ (SQLite)│  │ (SSE)   │  │ Breaker │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTPS
┌─────────────────────────┴───────────────────────────────┐
│                    OpenRouter API                        │
│         GPT-4o │ Claude │ Gemini │ Grok │ ...           │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, Vite, react-markdown |
| Backend | FastAPI, Python 3.10+, httpx |
| Database | SQLite (via SQLAlchemy) |
| Hosting | Vercel (frontend), Railway (backend) |
| LLM API | OpenRouter (multi-provider gateway) |
| Testing | pytest, pytest-asyncio, httpx mocks |

---

## Contributing

This project was started as a weekend hack but has evolved into a useful tool. While not actively maintained as a community project, PRs are welcome for:

- Bug fixes
- Performance improvements
- New model support
- Documentation improvements

---

## Changelog

See commit history for detailed changes. Major milestones:

- **v0** - Initial 3-stage deliberation system
- **Phase 1** - Production deployment with settings and themes
- **Phase 2** - Streaming, multi-turn context, circuit breaker
