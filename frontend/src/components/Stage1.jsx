import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage1.css';

const getConfidenceLevel = (score) => {
  if (score >= 0.8) return 'high';
  if (score >= 0.5) return 'medium';
  return 'low';
};

const getConfidenceLabel = (score) => {
  if (score >= 0.9) return 'Very High';
  if (score >= 0.7) return 'High';
  if (score >= 0.5) return 'Moderate';
  if (score >= 0.3) return 'Low';
  return 'Very Low';
};

export default function Stage1({ responses }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  const hasAnyConfidence = responses.some(r => r.confidence !== null && r.confidence !== undefined);

  return (
    <div className="stage stage1">
      <h3 className="stage-title">Stage 1: Individual Responses</h3>

      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {resp.model.split('/')[1] || resp.model}
            {resp.confidence !== null && resp.confidence !== undefined && (
              <span className={`tab-confidence ${getConfidenceLevel(resp.confidence)}`}>
                {Math.round(resp.confidence * 100)}%
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-header">
          <div className="model-name">{responses[activeTab].model}</div>
          {responses[activeTab].confidence !== null && responses[activeTab].confidence !== undefined && (
            <div className={`confidence-badge ${getConfidenceLevel(responses[activeTab].confidence)}`}>
              Confidence: {getConfidenceLabel(responses[activeTab].confidence)} ({Math.round(responses[activeTab].confidence * 100)}%)
            </div>
          )}
        </div>
        <div className="response-text markdown-content">
          <ReactMarkdown>{responses[activeTab].response}</ReactMarkdown>
        </div>
      </div>

      {hasAnyConfidence && (
        <div className="confidence-summary">
          <h4>Confidence Overview</h4>
          <div className="confidence-bars">
            {responses.map((resp, index) => (
              <div key={index} className="confidence-bar-item">
                <span className="confidence-bar-label">
                  {resp.model.split('/')[1] || resp.model}
                </span>
                <div className="confidence-bar-track">
                  <div
                    className={`confidence-bar-fill ${getConfidenceLevel(resp.confidence || 0)}`}
                    style={{ width: `${(resp.confidence || 0) * 100}%` }}
                  />
                </div>
                <span className="confidence-bar-value">
                  {resp.confidence !== null && resp.confidence !== undefined
                    ? `${Math.round(resp.confidence * 100)}%`
                    : 'N/A'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
