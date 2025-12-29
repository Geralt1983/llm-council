import { useState, useEffect } from 'react';
import { api } from '../api';
import './Analytics.css';

export default function Analytics({ isOpen, onClose }) {
  const [summary, setSummary] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [modelDetails, setModelDetails] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isOpen) {
      loadSummary();
    }
  }, [isOpen]);

  useEffect(() => {
    if (selectedModel) {
      loadModelDetails(selectedModel);
    }
  }, [selectedModel]);

  const loadSummary = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getAnalyticsSummary();
      setSummary(data);
    } catch (e) {
      setError('Failed to load analytics');
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const loadModelDetails = async (modelId) => {
    try {
      const data = await api.getModelAnalytics(modelId);
      setModelDetails(data);
    } catch (e) {
      console.error('Failed to load model details:', e);
    }
  };

  const formatTime = (ms) => {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const formatTokens = (count) => {
    if (!count) return '0';
    if (count < 1000) return count.toString();
    if (count < 1000000) return `${(count / 1000).toFixed(1)}K`;
    return `${(count / 1000000).toFixed(2)}M`;
  };

  const formatCost = (cost) => {
    if (!cost) return '$0.00';
    if (cost < 0.01) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
  };

  const getModelShortName = (modelId) => {
    if (!modelId) return 'Unknown';
    const parts = modelId.split('/');
    return parts[parts.length - 1];
  };

  if (!isOpen) return null;

  return (
    <div className="analytics-overlay" onClick={onClose}>
      <div className="analytics-panel" onClick={(e) => e.stopPropagation()}>
        <div className="analytics-header">
          <h2>Analytics Dashboard</h2>
          <button className="close-button" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="analytics-content">
          {error && <div className="analytics-error">{error}</div>}
          {isLoading && <div className="analytics-loading">Loading analytics...</div>}

          {summary && (
            <>
              {/* Summary Stats */}
              <section className="analytics-section">
                <h3>Overview</h3>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-value">{summary.total_conversations}</div>
                    <div className="stat-label">Conversations</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{summary.total_messages}</div>
                    <div className="stat-label">Messages</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{summary.total_api_calls}</div>
                    <div className="stat-label">API Calls</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">{formatTokens(summary.total_tokens)}</div>
                    <div className="stat-label">Total Tokens</div>
                  </div>
                  <div className="stat-card highlight">
                    <div className="stat-value">{formatCost(summary.total_cost_usd)}</div>
                    <div className="stat-label">Estimated Cost</div>
                  </div>
                </div>
              </section>

              {/* Model Performance */}
              <section className="analytics-section">
                <h3>Model Performance</h3>
                {summary.models && summary.models.length > 0 ? (
                  <div className="model-table">
                    <div className="model-table-header">
                      <span className="col-model">Model</span>
                      <span className="col-queries">Queries</span>
                      <span className="col-time">Avg Time</span>
                      <span className="col-tokens">Tokens</span>
                      <span className="col-ranking">Avg Rank</span>
                      <span className="col-cost">Cost</span>
                    </div>
                    {summary.models.map((model) => (
                      <div
                        key={model.model_id}
                        className={`model-table-row ${selectedModel === model.model_id ? 'selected' : ''}`}
                        onClick={() => setSelectedModel(
                          selectedModel === model.model_id ? null : model.model_id
                        )}
                      >
                        <span className="col-model" title={model.model_id}>
                          {getModelShortName(model.model_id)}
                        </span>
                        <span className="col-queries">{model.query_count}</span>
                        <span className="col-time">
                          {formatTime(model.avg_response_time_ms)}
                        </span>
                        <span className="col-tokens">
                          {formatTokens(model.total_tokens)}
                        </span>
                        <span className="col-ranking">
                          {model.avg_ranking ? `#${model.avg_ranking}` : '-'}
                        </span>
                        <span className="col-cost">
                          {formatCost(model.total_cost_usd)}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="no-data">No model data yet. Start a conversation!</div>
                )}
              </section>

              {/* Model Details Panel */}
              {selectedModel && modelDetails && (
                <section className="analytics-section model-details">
                  <h3>
                    {getModelShortName(selectedModel)} Details
                    <button
                      className="details-close"
                      onClick={() => setSelectedModel(null)}
                    >
                      &times;
                    </button>
                  </h3>
                  <div className="details-grid">
                    <div className="detail-group">
                      <h4>Response Time</h4>
                      <div className="detail-stats">
                        <div className="detail-item">
                          <span className="detail-label">Average</span>
                          <span className="detail-value">
                            {formatTime(modelDetails.response_time?.avg_ms || 0)}
                          </span>
                        </div>
                        <div className="detail-item">
                          <span className="detail-label">Min</span>
                          <span className="detail-value">
                            {formatTime(modelDetails.response_time?.min_ms || 0)}
                          </span>
                        </div>
                        <div className="detail-item">
                          <span className="detail-label">Max</span>
                          <span className="detail-value">
                            {formatTime(modelDetails.response_time?.max_ms || 0)}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="detail-group">
                      <h4>Token Usage</h4>
                      <div className="detail-stats">
                        <div className="detail-item">
                          <span className="detail-label">Total</span>
                          <span className="detail-value">
                            {formatTokens(modelDetails.tokens?.total || 0)}
                          </span>
                        </div>
                        <div className="detail-item">
                          <span className="detail-label">Avg/Query</span>
                          <span className="detail-value">
                            {formatTokens(modelDetails.tokens?.avg_per_query || 0)}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="detail-group">
                      <h4>Ranking</h4>
                      <div className="detail-stats">
                        <div className="detail-item">
                          <span className="detail-label">Avg Position</span>
                          <span className="detail-value">
                            {modelDetails.ranking?.avg_position
                              ? `#${modelDetails.ranking.avg_position}`
                              : '-'}
                          </span>
                        </div>
                        <div className="detail-item">
                          <span className="detail-label">Times Ranked</span>
                          <span className="detail-value">
                            {modelDetails.ranking?.times_ranked || 0}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="detail-group">
                      <h4>Cost & Errors</h4>
                      <div className="detail-stats">
                        <div className="detail-item">
                          <span className="detail-label">Total Cost</span>
                          <span className="detail-value">
                            {formatCost(modelDetails.cost?.total_usd)}
                          </span>
                        </div>
                        <div className="detail-item">
                          <span className="detail-label">Error Rate</span>
                          <span className={`detail-value ${modelDetails.errors?.rate > 5 ? 'error-rate-high' : ''}`}>
                            {modelDetails.errors?.rate || 0}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </section>
              )}
            </>
          )}
        </div>

        <div className="analytics-footer">
          <button className="refresh-button" onClick={loadSummary} disabled={isLoading}>
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </button>
          <button className="close-panel-button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
