# Deployment Guide

LLM Council uses a split deployment architecture:
- **Frontend**: Vercel (React/Vite)
- **Backend**: Railway (Python/FastAPI)

## Quick Start

### 1. Deploy Backend to Railway

1. **Create Railway account** at [railway.app](https://railway.app)

2. **Create new project from GitHub**:
   - Click "New Project" → "Deploy from GitHub repo"
   - Select `LLMCouncil` repository
   - Railway auto-detects Python

3. **Add environment variables** in Railway dashboard:
   ```
   OPENROUTER_API_KEY=your_openrouter_key
   FRONTEND_URL=https://your-app.vercel.app  # Add after Vercel deploy
   DATABASE_PATH=/data/llmcouncil.db
   ```

4. **Add persistent volume** (for SQLite):
   - Go to your service → Settings → Volumes
   - Add volume: Mount path `/data`
   - This persists your database between deploys

5. **Get your Railway URL**:
   - Found in Settings → Domains
   - Example: `https://llmcouncil-production.up.railway.app`

### 2. Deploy Frontend to Vercel

1. **Create Vercel account** at [vercel.com](https://vercel.com)

2. **Import project**:
   - Click "Add New" → "Project"
   - Import from GitHub
   - Set **Root Directory** to `frontend`

3. **Configure environment variables**:
   ```
   VITE_API_URL=https://your-railway-url.up.railway.app
   ```

4. **Deploy**:
   - Vercel auto-detects Vite
   - Click Deploy

### 3. Update CORS

After Vercel deploys, go back to Railway and update:
```
FRONTEND_URL=https://your-app.vercel.app
```

## Environment Variables Reference

### Backend (Railway)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `FRONTEND_URL` | Yes | Vercel frontend URL for CORS |
| `DATABASE_PATH` | No | SQLite path (default: `data/llmcouncil.db`) |
| `PORT` | No | Auto-set by Railway |

### Frontend (Vercel)

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_API_URL` | Yes | Railway backend URL |

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│      Vercel         │         │      Railway        │
│    (Frontend)       │ ──────▶ │     (Backend)       │
│                     │  HTTPS  │                     │
│  React + Vite       │         │  FastAPI + SQLite   │
│  Static hosting     │         │  Persistent volume  │
└─────────────────────┘         └─────────────────────┘
                                         │
                                         ▼
                                ┌─────────────────────┐
                                │    OpenRouter API   │
                                │   (LLM providers)   │
                                └─────────────────────┘
```

## Costs

| Service | Free Tier | Notes |
|---------|-----------|-------|
| **Vercel** | 100GB bandwidth/mo | Hobby plan is free |
| **Railway** | $5 credit/mo | ~500 hours at $0.01/hr |
| **OpenRouter** | Pay-per-use | Varies by model |

For light usage, you'll stay in free tiers. Railway's $5/mo covers ~24/7 operation of a small instance.

## Troubleshooting

### CORS Errors
- Verify `FRONTEND_URL` in Railway matches your Vercel domain exactly
- Check both `https://` and without trailing slash

### Database Not Persisting
- Ensure you added a volume in Railway mounted at `/data`
- Set `DATABASE_PATH=/data/llmcouncil.db`

### API Timeout
- Railway has no timeout limits for SSE
- If still timing out, check OpenRouter API key is valid

### Build Failures

**Railway**: Check `pyproject.toml` has all dependencies:
```bash
uv sync  # Verify locally first
```

**Vercel**: Check `frontend/package.json` and build logs:
```bash
cd frontend && npm run build  # Verify locally first
```

## Local Development

```bash
# Backend
cd LLMCouncil
cp .env.example .env  # Edit with your API key
uv run python -m backend.main

# Frontend (new terminal)
cd frontend
cp .env.example .env.local
npm run dev
```

## Custom Domains

### Vercel
1. Go to Project Settings → Domains
2. Add your domain
3. Update DNS records as instructed

### Railway
1. Go to Service Settings → Domains
2. Add custom domain
3. Update DNS CNAME to Railway URL
4. Update `FRONTEND_URL` if frontend domain changed
