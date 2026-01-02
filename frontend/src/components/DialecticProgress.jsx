import './DialecticProgress.css';

const STAGES = [
  { key: 'first_responder', label: 'First Response', icon: '💡', description: 'Comprehensive initial answer' },
  { key: 'devils_advocate', label: "Devil's Advocate", icon: '🔥', description: 'Challenges & alternatives' },
  { key: 'deep_insight', label: 'Deep Insight', icon: '🎯', description: 'Non-obvious wisdom' },
  { key: 'action_coach', label: 'Action Coach', icon: '🚀', description: 'Specific action steps' },
  { key: 'final_synthesis', label: 'Final Synthesis', icon: '✨', description: 'Life-changing response' },
];

export default function DialecticProgress({ loading, stages, currentStage, currentModel }) {
  const getStageStatus = (stageKey) => {
    if (loading[stageKey]) return 'active';
    if (stages[stageKey]) return 'complete';
    return 'pending';
  };

  const completedCount = Object.keys(stages).length;
  const isActive = Object.values(loading).some(Boolean);

  if (!isActive && completedCount === 0) return null;

  return (
    <div className="dialectic-progress">
      <div className="dialectic-header">
        <span className="dialectic-title">Dialectic Chain</span>
        <span className="dialectic-subtitle">
          {isActive ? `Stage ${completedCount + 1} of 5` : `${completedCount} stages complete`}
        </span>
      </div>

      <div className="dialectic-stages">
        {STAGES.map((stage, index) => {
          const status = getStageStatus(stage.key);
          const stageData = stages[stage.key];

          return (
            <div key={stage.key} className={`dialectic-stage ${status}`}>
              <div className="stage-icon">{stage.icon}</div>
              <div className="stage-info">
                <div className="stage-label">{stage.label}</div>
                {status === 'active' && currentModel && (
                  <div className="stage-model">{currentModel.split('/').pop()}</div>
                )}
                {status === 'complete' && stageData?.model && (
                  <div className="stage-model">{stageData.model.split('/').pop()}</div>
                )}
              </div>
              {status === 'active' && <div className="stage-spinner" />}
              {status === 'complete' && <div className="stage-check">✓</div>}
              {index < STAGES.length - 1 && <div className="stage-connector" />}
            </div>
          );
        })}
      </div>
    </div>
  );
}
