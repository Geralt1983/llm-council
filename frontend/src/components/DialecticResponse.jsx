import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './DialecticResponse.css';

const STAGE_INFO = {
  first_responder: { label: 'First Response', icon: '💡', color: '#3498db' },
  devils_advocate: { label: "Devil's Advocate", icon: '🔥', color: '#e74c3c' },
  deep_insight: { label: 'Deep Insight', icon: '🎯', color: '#9b59b6' },
  action_coach: { label: 'Action Coach', icon: '🚀', color: '#27ae60' },
  final_synthesis: { label: 'Final Synthesis', icon: '✨', color: '#f39c12' },
};

export default function DialecticResponse({ stages, finalResponse }) {
  const [expandedStage, setExpandedStage] = useState(null);

  const toggleStage = (stageKey) => {
    setExpandedStage(expandedStage === stageKey ? null : stageKey);
  };

  const stageOrder = ['first_responder', 'devils_advocate', 'deep_insight', 'action_coach'];

  return (
    <div className="dialectic-response">
      {/* Final synthesized response - the main output */}
      <div className="final-response">
        <div className="final-response-header">
          <span className="final-icon">✨</span>
          <span className="final-label">Life-Changing Response</span>
        </div>
        <div className="final-response-content markdown-content">
          <ReactMarkdown>{finalResponse}</ReactMarkdown>
        </div>
      </div>

      {/* Expandable thinking process */}
      {Object.keys(stages).length > 0 && (
        <div className="thinking-process">
          <div className="thinking-header">
            <span>View Thinking Process</span>
            <span className="thinking-count">{Object.keys(stages).length} stages</span>
          </div>
          <div className="thinking-stages">
            {stageOrder.map((stageKey) => {
              const stageData = stages[stageKey];
              if (!stageData) return null;

              const info = STAGE_INFO[stageKey];
              const isExpanded = expandedStage === stageKey;

              return (
                <div key={stageKey} className="thinking-stage">
                  <button
                    className={`stage-toggle ${isExpanded ? 'expanded' : ''}`}
                    onClick={() => toggleStage(stageKey)}
                    style={{ '--stage-color': info.color }}
                  >
                    <span className="toggle-icon">{info.icon}</span>
                    <span className="toggle-label">{info.label}</span>
                    <span className="toggle-model">{stageData.model?.split('/').pop()}</span>
                    <span className="toggle-arrow">{isExpanded ? '▼' : '▶'}</span>
                  </button>
                  {isExpanded && (
                    <div className="stage-content markdown-content">
                      <ReactMarkdown>{stageData.content}</ReactMarkdown>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
