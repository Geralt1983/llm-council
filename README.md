# LLM Council

![llmcouncil](header.jpg)

**A production-ready multi-LLM deliberation platform where AI models collaborate through anonymized peer review to produce higher-quality answers than any single model.**

Instead of relying on a single LLM, LLM Council orchestrates multiple frontier models (GPT-4o, Claude, Gemini, Grok, etc.) through a 3-stage deliberation process with anonymized peer review, preventing bias and surfacing the best collective answer.

## Why LLM Council?

**The Problem:** Different LLMs have different strengths. GPT-4 excels at code, Claude at writing, Gemini at analysis. Asking just one means missing insights from the others.

**The Solution:** LLM Council runs all models in parallel, has them anonymously review each other's work, then synthesizes the best collective answer - complete with transparency into where models agree and disagree.

**Key Benefits:**
- 🎯 **Higher Quality:** Peer review surfaces blind spots and errors
- 🔍 **Full Transparency:** See every model's response and ranking
- ⚖️ **Bias Prevention:** Anonymized evaluation prevents models playing favorites
- 📊 **Rich Analytics:** Track performance, cost, and consensus over time
- 🧠 **Trust Calibration:** Know when models agree (trust it) vs disagree (verify it)

## How It Works

### 3-Stage Deliberation Process

```
User Question
    ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Initial Responses (Parallel)                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ GPT-4o  │  │ Claude  │  │ Gemini  │  │  Grok   │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Anonymous Peer Review                         │
│  Each model ranks all responses (anonymized as A,B,C)   │
│  "Response A: accurate but lacks depth"                 │
│  "Response C: most comprehensive"                       │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Chairman Synthesis                            │
│  Designated chairman weighs all responses & rankings    │
│  → Final answer with collective wisdom                  │
└─────────────────────────────────────────────────────────┘
```

**Stage 1: Initial Responses**
- Query sent to all council models in parallel
- Each model responds independently
- Responses displayed in tabs for inspection

**Stage 2: Anonymous Peer Review**
- Responses anonymized as "Response A", "Response B", etc.
- Each model evaluates and ranks all responses
- Rankings aggregated to identify consensus

**Stage 3: Chairman Synthesis**
- Designated chairman model receives all responses + rankings
- Synthesizes final answer incorporating collective insights
- Result shown with full audit trail

## Features

### ✅ Production Ready (Phase 0-4 Complete)

**Core Deliberation**
- 3-stage council process with anonymized peer review
- Multi-turn conversations with full context history
- Token-by-token streaming for real-time feedback
- Circuit breaker pattern for fault tolerance

**Analytics & Insights**
- Response time tracking per model per stage
- Token usage (input/output/total) with cost estimation
- Aggregate rankings showing model performance
- Dissent tracking: agreement scores, controversies, unanimous winners
- Confidence scores with visual indicators

**Advanced Deliberation**
- Custom ranking criteria (define your own evaluation dimensions)
- Weighted voting (give expert models more influence)
- Reasoning model support (o1, o3 with extended thinking)
- Dissent metrics (see exactly where models disagree)

**User Experience**
- Clean React UI with stage-by-stage visualization
- Settings panel for council configuration
- Conversation management with SQLite storage
- Export to Markdown/JSON
- Dark/light theme support

### 🚀 Coming Soon (Phase 5-7 Roadmap)

**Trust Calibration**
- Disagreement highlighting (visual maps of consensus vs. controversy)
- Confidence prediction ("⚠️ LOW CONSENSUS" warnings)
- Claim-level verification

**Smart Automation**
- Automatic mode selection (simple questions → fast model, complex → full council)
- Cost/quality tradeoffs (Fast/Balanced/Thorough modes)
- Iterative refinement on disagreements

**Integrations**
- Slack bot, CLI, VSCode extension
- Public API for programmatic access

See [ROADMAP.md](ROADMAP.md) for full feature roadmap.

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 16+
- [OpenRouter API key](https://openrouter.ai/) (with credits)

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for Python package management.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Get your API key at [openrouter.ai](https://openrouter.ai/). Make sure to purchase credits or enable automatic top-up.

### 3. Configure Council (Optional)

Edit `backend/config.py` to customize your council:

```python
COUNCIL_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-pro",
    "x-ai/grok-3",
]

CHAIRMAN_MODEL = "anthropic/claude-sonnet-4.5"
```

**Tip:** Use models with complementary strengths (coding, writing, analysis, creativity).

### 4. Run the Application

**Option 1: Start script (recommended)**
```bash
./start.sh
```

**Option 2: Manual**

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Usage Examples

**General Knowledge:**
```
Q: "Explain quantum entanglement"
→ Physics models provide accuracy, general models add accessibility
→ Final answer balances technical depth with clarity
```

**Coding Questions:**
```
Q: "Best way to implement auth in a React app?"
→ GPT-4o suggests JWT, Claude prefers sessions, Gemini weighs trade-offs
→ Chairman synthesizes: "Use sessions for simpler apps, JWT for distributed systems"
```

**Decision Making:**
```
Q: "Should I use Postgres or MongoDB for my app?"
→ Models debate based on different criteria (ACID, scalability, schema flexibility)
→ Aggregate ranking shows consensus, dissent tracking highlights trade-offs
```

**Research:**
```
Q: "What are the latest developments in LLM reasoning?"
→ Multiple models cite different papers and perspectives
→ Synthesis provides comprehensive overview with diverse sources
```

## Architecture

**Backend:**
- FastAPI (Python 3.10+)
- Async httpx for parallel API calls
- OpenRouter API for multi-model access
- SQLite for conversation storage
- Circuit breaker pattern for reliability

**Frontend:**
- React 18 + Vite
- react-markdown for response rendering
- Server-Sent Events (SSE) for streaming
- Tab-based stage visualization

**Infrastructure:**
- uv for Python package management
- pytest for testing
- Deployable to Vercel (frontend) + Railway (backend)

## Testing

Run the full test suite:

```bash
cd backend
python -m pytest
```

Run specific test categories:

```bash
python -m pytest tests/unit              # Unit tests
python -m pytest tests/integration       # Integration tests
python -m pytest tests/test_council.py   # Council logic
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes |
| `DATABASE_URL` | SQLite database path | No (defaults to `data/conversations.db`) |

### Model Configuration

**Council Models** (`backend/config.py`):
```python
COUNCIL_MODELS = [...]  # List of model identifiers
```

Browse available models at [openrouter.ai/models](https://openrouter.ai/models).

**Chairman Model:**
- Can be same as a council member or different
- Recommended: Use a strong reasoning model (Claude, GPT-4o, Gemini Pro)

**Ranking Criteria** (optional):
```python
ranking_criteria = [
    {"name": "Accuracy", "description": "Factual correctness", "enabled": True},
    {"name": "Depth", "description": "Comprehensive analysis", "enabled": True},
    {"name": "Clarity", "description": "Clear explanation", "enabled": True},
]
```

**Model Weights** (optional):
```python
model_weights = {
    "openai/gpt-4o": 1.5,      # 1.5x weight for coding questions
    "anthropic/claude-sonnet-4.5": 1.2,
}
```

## Cost Management

LLM Council runs multiple models per query, so costs add up:

**Typical Query Costs:**
- 3 models, simple question: ~$0.02-0.05
- 5 models, complex question: ~$0.10-0.25
- With reasoning models (o1): ~$0.50-1.00

**Cost Optimization Tips:**
1. Use fewer/cheaper models for simple questions
2. Enable cost tracking in analytics dashboard
3. Set up model presets for different use cases
4. Use fast models like Gemini Flash for chairman synthesis

The analytics dashboard shows exact costs per query and model.

## Contributing

This project started as [Karpathy's weekend hack](https://x.com/karpathy/status/1990577951671509438) and has evolved into a production-ready deliberation platform.

**Contributions welcome for:**
- Bug fixes
- Performance improvements
- New model support
- Documentation improvements
- Phase 5-7 feature implementations (see ROADMAP.md)

**Development:**
```bash
# Run tests
cd backend && python -m pytest

# Format code
ruff format backend/
npm run format

# Type checking
mypy backend/
```

## Project Status

- ✅ **Phase 0-4:** Complete (core deliberation, analytics, advanced features)
- 🚧 **Phase 5:** Model Management (planned)
- 💭 **Phase 6:** Collaboration features (planned)
- 💭 **Phase 7:** Intelligence & Trust (planned)

See [ROADMAP.md](ROADMAP.md) for detailed feature roadmap and [CLAUDE.md](CLAUDE.md) for technical implementation notes.

## License

MIT

## Acknowledgments

- Original concept by [Andrej Karpathy](https://x.com/karpathy)
- Built with [OpenRouter](https://openrouter.ai/) for multi-model access
- Inspired by the idea that collective intelligence beats individual expertise

---

**Questions?** Check the [ROADMAP.md](ROADMAP.md) for planned features or [CLAUDE.md](CLAUDE.md) for implementation details.

**Issues?** Please report bugs and feature requests on GitHub Issues.
