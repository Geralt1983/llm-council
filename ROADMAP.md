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

### Phase 3: Analytics & Insights
**Status:** ✅ Complete

- [x] Response timing per model per stage
- [x] Token usage tracking (input/output/total)
- [x] Cost estimation based on model pricing
- [x] Analytics API endpoints (`/api/analytics/*`)
- [x] Frontend analytics dashboard with model performance table
- [x] Ranking position tracking and aggregation

### Phase 4: Advanced Deliberation
**Status:** ✅ Complete

- [x] Custom ranking criteria (user-defined evaluation dimensions)
- [x] Weighted voting (configurable model influence)
- [x] Reasoning model support (o1, o3, thinking models with special handling)
- [x] Dissent tracking (agreement scores, controversies, ranking spread)
- [x] Confidence scores (models self-rate, visual bar chart summary)

---

## Upcoming Phases

### Phase 5: Model Management
**Status:** 🔜 Next

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

### Phase 7: Intelligence & Trust
**Status:** 💭 Future

**Goal:** Make the council indispensable through trust calibration and intelligent automation.

**The Vision:** Transform from "interesting multi-LLM tool" to "tool I trust for important decisions."

#### 7.1 Trust Calibration (Highest Impact)

| Feature | Description | Priority | Impact |
|---------|-------------|----------|--------|
| Disagreement highlighting | Visual map showing WHERE models agree/disagree on specific claims | **Critical** | 🔥 Transforms usability |
| Confidence prediction | "⚠️ LOW CONSENSUS" vs "✓ HIGH CONFIDENCE" based on agreement metrics | **Critical** | 🔥 Prevents false trust |
| Claim extraction | Break synthesis into individual factual claims with per-claim confidence | High | Shows uncertainty granularly |
| Uncertainty flags | Explicit callouts: "Models disagree on: [exact cause of X]" | High | Actionable insights |
| Quality calibration | Historical tracking: "Council 94% accurate on math questions" | Medium | Builds trust over time |

**Why this matters:** Users currently don't know when to trust the answer. This tells them exactly where the answer is solid vs. where to verify.

#### 7.2 Smart Automation

| Feature | Description | Priority | Impact |
|---------|-------------|----------|--------|
| Automatic mode selection | AI decides: simple question → fast model, complex → full council | High | Saves 80% of time/cost |
| Cost/quality tradeoffs | User picks: Fast ($0.05, 10s) / Balanced / Thorough ($0.40, 45s) | High | Explicit value control |
| Question classifier | Detect: factual / creative / coding / research → route appropriately | Medium | Better decisions |
| Decision explanation | "Using full council because: models historically disagree on economics" | Low | Transparency |

**Why this matters:** Currently every query uses the full council. Most queries don't need it. This makes the tool fast for simple questions while maintaining rigor for complex ones.

#### 7.3 Disagreement Deep-Dive

| Feature | Description | Priority | Impact |
|---------|-------------|----------|--------|
| Controversy analyzer | Show specific claims with split votes: "3 models say X, 2 say Y" | High | 🔥 Actionable insights |
| Iterative refinement | Auto-trigger Round 2 when consensus < 70%, focused on contested points | **Critical** | 🔥 Better answers on hard questions |
| Reasoning comparison | For o1/o3 models, show where logical reasoning paths diverge | Medium | Educational + better synthesis |
| Claim-level tracking | Track each factual claim across all responses, show support/contradiction | Medium | Granular uncertainty |

**Why this matters:** "Models disagree" is vague. This shows WHAT they disagree about and triggers refinement to resolve it.

#### 7.4 Domain Intelligence

| Feature | Description | Priority | Impact |
|---------|-------------|----------|--------|
| Domain classification | Auto-detect question domain: coding / math / creative / research | High | Enables smart weighting |
| Expertise weighting | "For Python questions, GPT-4's ranking counts 2x" | High | More accurate rankings |
| Specialized councils | Pre-configured: `/council code`, `/council research`, `/council creative` | Medium | Optimized for use case |
| Performance tracking | Which models perform best on which domains over time | Medium | Continuous improvement |
| Learning from outcomes | Track ground truth when available, adjust weights | Low | Self-improving system |

**Why this matters:** Not all models are equal at all tasks. Weighting by domain makes deliberation smarter.

#### 7.5 Real-World Integration

| Feature | Description | Priority | Impact |
|---------|-------------|----------|--------|
| Slack bot | `/llm-council "should we use Postgres or MongoDB?"` | High | 🔥 Meets users where they work |
| CLI tool | `council ask "your question"` for terminal workflows | Medium | Developer adoption |
| VSCode extension | Highlight code → right-click → "Ask council" | Medium | IDE integration |
| Public API | REST API for programmatic access | Medium | Enables ecosystem |
| Share links | Public URLs for council sessions (read-only) | Low | Collaboration |

**Why this matters:** Currently requires opening a web app. Integration into existing workflows drives adoption.

---

## Implementation Priority

**Phase 7 Quick Wins** (Weekend projects with massive impact):

1. **Disagreement Highlighting** (Phase 7.1 + 7.3)
   - Parse Stage 1 responses into claims
   - Show which models support each claim
   - Visual diff of consensus vs. controversy
   - **Impact:** Transforms tool from "black box" to "transparent deliberation"

2. **Confidence Prediction** (Phase 7.1)
   - Use existing `agreement_score` and `controversies` metrics
   - Add simple heuristic: high agreement = high confidence
   - Show banner: "⚠️ LOW CONSENSUS - VERIFY" or "✓ HIGH CONFIDENCE"
   - **Impact:** Prevents users from trusting uncertain answers

3. **Iterative Refinement** (Phase 7.3)
   - When `agreement_score < 0.7`, auto-trigger Round 2
   - Prompt: "Models, address this disagreement: [extracted controversy]"
   - Re-run synthesis with refined inputs
   - **Impact:** Better answers on genuinely hard questions

4. **Smart Routing** (Phase 7.2)
   - Use fast model to classify question complexity
   - Simple/factual → single model, Complex/ambiguous → full council
   - **Impact:** 80% time/cost savings on routine questions

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
- **Phase 3** - Analytics dashboard with timing, tokens, cost tracking
- **Phase 4** - Advanced deliberation: custom criteria, weighted voting, dissent tracking, reasoning models, confidence scores
