import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage2.css';

function deAnonymizeText(text, labelToModel) {
  if (!labelToModel) return text;

  let result = text;
  // Replace each "Response X" with the actual model name
  Object.entries(labelToModel).forEach(([label, model]) => {
    const modelShortName = model.split('/')[1] || model;
    result = result.replace(new RegExp(label, 'g'), `**${modelShortName}**`);
  });
  return result;
}

export default function Stage2({ rankings, labelToModel, aggregateRankings, dissent }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!rankings || rankings.length === 0) {
    return null;
  }

  const getShortModelName = (model) => model.split('/')[1] || model;

  return (
    <div className="stage stage2">
      <h3 className="stage-title">Stage 2: Peer Rankings</h3>

      <h4>Raw Evaluations</h4>
      <p className="stage-description">
        Each model evaluated all responses (anonymized as Response A, B, C, etc.) and provided rankings.
        Below, model names are shown in <strong>bold</strong> for readability, but the original evaluation used anonymous labels.
      </p>

      <div className="tabs">
        {rankings.map((rank, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {rank.model.split('/')[1] || rank.model}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="ranking-model">
          {rankings[activeTab].model}
        </div>
        <div className="ranking-content markdown-content">
          <ReactMarkdown>
            {deAnonymizeText(rankings[activeTab].ranking, labelToModel)}
          </ReactMarkdown>
        </div>

        {rankings[activeTab].parsed_ranking &&
         rankings[activeTab].parsed_ranking.length > 0 && (
          <div className="parsed-ranking">
            <strong>Extracted Ranking:</strong>
            <ol>
              {rankings[activeTab].parsed_ranking.map((label, i) => (
                <li key={i}>
                  {labelToModel && labelToModel[label]
                    ? labelToModel[label].split('/')[1] || labelToModel[label]
                    : label}
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>

      {aggregateRankings && aggregateRankings.length > 0 && (
        <div className="aggregate-rankings">
          <h4>Aggregate Rankings (Street Cred)</h4>
          <p className="stage-description">
            Combined results across all peer evaluations (lower score is better):
          </p>
          <div className="aggregate-list">
            {aggregateRankings.map((agg, index) => (
              <div key={index} className="aggregate-item">
                <span className="rank-position">#{index + 1}</span>
                <span className="rank-model">
                  {agg.model.split('/')[1] || agg.model}
                </span>
                <span className="rank-score">
                  Avg: {agg.average_rank.toFixed(2)}
                </span>
                <span className="rank-count">
                  ({agg.rankings_count} votes)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {dissent && (
        <div className="dissent-metrics">
          <h4>Consensus Analysis</h4>
          <div className="dissent-content">
            <div className="agreement-score">
              <span className="dissent-label">Agreement Score:</span>
              <span className={`dissent-value ${dissent.agreement_score >= 0.7 ? 'high' : dissent.agreement_score >= 0.4 ? 'medium' : 'low'}`}>
                {(dissent.agreement_score * 100).toFixed(0)}%
              </span>
              <span className="dissent-hint">
                {dissent.agreement_score >= 0.7 ? '(Strong consensus)' :
                 dissent.agreement_score >= 0.4 ? '(Moderate agreement)' :
                 '(Significant disagreement)'}
              </span>
            </div>

            {dissent.unanimous_winner && (
              <div className="unanimous-winner">
                <span className="dissent-label">Unanimous Winner:</span>
                <span className="dissent-value winner">
                  {getShortModelName(dissent.unanimous_winner)}
                </span>
              </div>
            )}

            {dissent.controversies && dissent.controversies.length > 0 && (
              <div className="controversies">
                <span className="dissent-label">Controversial Responses:</span>
                <div className="controversy-list">
                  {dissent.controversies.map((model, index) => (
                    <span key={index} className="controversy-item">
                      {getShortModelName(model)}
                    </span>
                  ))}
                </div>
                <span className="dissent-hint">
                  (Received both first and last place votes)
                </span>
              </div>
            )}

            {dissent.ranking_spread && Object.keys(dissent.ranking_spread).length > 0 && (
              <div className="ranking-spread">
                <span className="dissent-label">Ranking Spread:</span>
                <div className="spread-list">
                  {Object.entries(dissent.ranking_spread).map(([model, spread]) => (
                    <div key={model} className="spread-item">
                      <span className="spread-model">{getShortModelName(model)}</span>
                      <span className="spread-value">±{spread.toFixed(1)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
