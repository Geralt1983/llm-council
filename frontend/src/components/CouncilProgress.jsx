import { useState, useEffect } from 'react';
import './CouncilProgress.css';

/**
 * CouncilProgress - Visual progress indicator for the 3-stage council deliberation
 *
 * Design goals:
 * - Show clear visual progress through all 3 stages
 * - Display which models are actively responding
 * - Provide timing information
 * - Animate to show activity
 */

export default function CouncilProgress({
  stage1Loading,
  stage1Data,
  stage2Loading,
  stage2Data,
  stage3Loading,
  stage3Data,
  councilModels = [],
  chairmanModel = null,
}) {
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime] = useState(Date.now());

  // Track elapsed time while any stage is loading
  const isAnyLoading = stage1Loading || stage2Loading || stage3Loading;

  useEffect(() => {
    if (!isAnyLoading) return;

    const timer = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => clearInterval(timer);
  }, [isAnyLoading, startTime]);

  // Calculate current stage
  const getCurrentStage = () => {
    if (stage3Loading || stage3Data) return 3;
    if (stage2Loading || stage2Data) return 2;
    if (stage1Loading || stage1Data) return 1;
    return 0;
  };

  const currentStage = getCurrentStage();

  // Determine stage states
  const getStageState = (stageNum) => {
    if (stageNum === 1) {
      if (stage1Data) return 'complete';
      if (stage1Loading) return 'active';
      return 'pending';
    }
    if (stageNum === 2) {
      if (stage2Data) return 'complete';
      if (stage2Loading) return 'active';
      if (stage1Data) return 'pending';
      return 'locked';
    }
    if (stageNum === 3) {
      if (stage3Data) return 'complete';
      if (stage3Loading) return 'active';
      if (stage2Data) return 'pending';
      return 'locked';
    }
    return 'locked';
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  // Get model display name
  const getModelName = (model) => {
    if (!model) return '';
    return model.split('/')[1] || model;
  };

  // Get count of completed responses for Stage 1
  const stage1ResponseCount = stage1Data?.length || 0;
  const stage1TotalModels = councilModels.length || 4;

  // Get count of completed rankings for Stage 2
  const stage2RankingCount = stage2Data?.length || 0;

  // Don't show if nothing is happening
  if (!isAnyLoading && !stage1Data && !stage2Data && !stage3Data) {
    return null;
  }

  // Hide once everything is complete
  if (stage3Data && !isAnyLoading) {
    return null;
  }

  return (
    <div className="council-progress">
      <div className="progress-header">
        <div className="progress-title">
          <span className="pulse-dot"></span>
          Council Deliberation
        </div>
        <div className="progress-timer">
          {formatTime(elapsedTime)}
        </div>
      </div>

      <div className="progress-stages">
        {/* Stage 1 */}
        <div className={`progress-stage ${getStageState(1)}`}>
          <div className="stage-indicator">
            {getStageState(1) === 'complete' ? (
              <span className="stage-check">✓</span>
            ) : getStageState(1) === 'active' ? (
              <span className="stage-spinner"></span>
            ) : (
              <span className="stage-number">1</span>
            )}
          </div>
          <div className="stage-info">
            <div className="stage-name">Collecting Responses</div>
            <div className="stage-detail">
              {getStageState(1) === 'active' && (
                <>
                  <span className="model-count">{stage1ResponseCount}/{stage1TotalModels} models</span>
                  <div className="model-dots">
                    {councilModels.slice(0, 4).map((model, i) => (
                      <span
                        key={i}
                        className={`model-dot ${i < stage1ResponseCount ? 'filled' : 'pending'}`}
                        title={getModelName(model)}
                      />
                    ))}
                    {councilModels.length > 4 && <span className="model-dot-more">+{councilModels.length - 4}</span>}
                  </div>
                </>
              )}
              {getStageState(1) === 'complete' && (
                <span className="complete-text">{stage1ResponseCount} responses</span>
              )}
            </div>
          </div>
        </div>

        {/* Connector */}
        <div className={`progress-connector ${currentStage >= 2 ? 'active' : ''}`}>
          <div className="connector-line"></div>
        </div>

        {/* Stage 2 */}
        <div className={`progress-stage ${getStageState(2)}`}>
          <div className="stage-indicator">
            {getStageState(2) === 'complete' ? (
              <span className="stage-check">✓</span>
            ) : getStageState(2) === 'active' ? (
              <span className="stage-spinner"></span>
            ) : (
              <span className="stage-number">2</span>
            )}
          </div>
          <div className="stage-info">
            <div className="stage-name">Peer Evaluation</div>
            <div className="stage-detail">
              {getStageState(2) === 'active' && (
                <>
                  <span className="model-count">{stage2RankingCount}/{stage1TotalModels} rankings</span>
                  <div className="model-dots">
                    {councilModels.slice(0, 4).map((model, i) => (
                      <span
                        key={i}
                        className={`model-dot ${i < stage2RankingCount ? 'filled' : 'pending'}`}
                        title={getModelName(model)}
                      />
                    ))}
                  </div>
                </>
              )}
              {getStageState(2) === 'complete' && (
                <span className="complete-text">{stage2RankingCount} evaluations</span>
              )}
              {getStageState(2) === 'locked' && (
                <span className="locked-text">Waiting...</span>
              )}
            </div>
          </div>
        </div>

        {/* Connector */}
        <div className={`progress-connector ${currentStage >= 3 ? 'active' : ''}`}>
          <div className="connector-line"></div>
        </div>

        {/* Stage 3 */}
        <div className={`progress-stage ${getStageState(3)}`}>
          <div className="stage-indicator">
            {getStageState(3) === 'complete' ? (
              <span className="stage-check">✓</span>
            ) : getStageState(3) === 'active' ? (
              <span className="stage-spinner"></span>
            ) : (
              <span className="stage-number">3</span>
            )}
          </div>
          <div className="stage-info">
            <div className="stage-name">Chairman Synthesis</div>
            <div className="stage-detail">
              {getStageState(3) === 'active' && chairmanModel && (
                <span className="chairman-active">
                  {getModelName(chairmanModel)} writing...
                </span>
              )}
              {getStageState(3) === 'active' && !chairmanModel && (
                <span className="chairman-active">Synthesizing...</span>
              )}
              {getStageState(3) === 'complete' && (
                <span className="complete-text">Complete</span>
              )}
              {getStageState(3) === 'locked' && (
                <span className="locked-text">Waiting...</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Activity indicator */}
      {isAnyLoading && (
        <div className="progress-activity">
          <div className="activity-bar">
            <div className="activity-fill"></div>
          </div>
        </div>
      )}
    </div>
  );
}
