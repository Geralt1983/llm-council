import { useState, useEffect } from 'react';
import { api } from '../api';
import './Settings.css';

// Common OpenRouter models
const AVAILABLE_MODELS = [
  { id: 'openai/gpt-4o', name: 'GPT-4o', provider: 'OpenAI' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini', provider: 'OpenAI' },
  { id: 'openai/o1', name: 'o1', provider: 'OpenAI', isReasoning: true },
  { id: 'openai/o1-mini', name: 'o1 Mini', provider: 'OpenAI', isReasoning: true },
  { id: 'openai/o1-preview', name: 'o1 Preview', provider: 'OpenAI', isReasoning: true },
  { id: 'openai/o3-mini', name: 'o3 Mini', provider: 'OpenAI', isReasoning: true },
  { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', provider: 'Anthropic' },
  { id: 'anthropic/claude-3-opus', name: 'Claude 3 Opus', provider: 'Anthropic' },
  { id: 'anthropic/claude-3-haiku', name: 'Claude 3 Haiku', provider: 'Anthropic' },
  { id: 'google/gemini-2.5-flash', name: 'Gemini 2.5 Flash', provider: 'Google' },
  { id: 'google/gemini-2.5-pro', name: 'Gemini 2.5 Pro', provider: 'Google' },
  { id: 'google/gemini-2.0-flash-thinking', name: 'Gemini 2.0 Flash Thinking', provider: 'Google', isReasoning: true },
  { id: 'google/gemini-flash-1.5', name: 'Gemini Flash 1.5', provider: 'Google' },
  { id: 'x-ai/grok-3', name: 'Grok 3', provider: 'xAI' },
  { id: 'x-ai/grok-2', name: 'Grok 2', provider: 'xAI' },
  { id: 'meta-llama/llama-3.1-405b-instruct', name: 'Llama 3.1 405B', provider: 'Meta' },
  { id: 'meta-llama/llama-3.1-70b-instruct', name: 'Llama 3.1 70B', provider: 'Meta' },
  { id: 'mistralai/mistral-large', name: 'Mistral Large', provider: 'Mistral' },
  { id: 'deepseek/deepseek-chat', name: 'DeepSeek Chat', provider: 'DeepSeek' },
  { id: 'deepseek/deepseek-reasoner', name: 'DeepSeek Reasoner', provider: 'DeepSeek', isReasoning: true },
  { id: 'perplexity/sonar-pro', name: 'Sonar Pro', provider: 'Perplexity' },
];

// Helper to check if a model is a reasoning model
const isReasoningModel = (modelId) => {
  const model = AVAILABLE_MODELS.find(m => m.id === modelId);
  if (model?.isReasoning) return true;
  // Also check pattern for custom models
  const patterns = ['o1', 'o3', 'reasoning', 'thinking'];
  return patterns.some(p => modelId.toLowerCase().includes(p));
};

// Default ranking criteria
const DEFAULT_CRITERIA = [
  { id: 'accuracy', name: 'Accuracy', description: 'Factual correctness and precision', weight: 1.0, enabled: true },
  { id: 'completeness', name: 'Completeness', description: 'Thoroughness and coverage of the topic', weight: 1.0, enabled: true },
  { id: 'clarity', name: 'Clarity', description: 'Clear and easy to understand', weight: 1.0, enabled: true },
];

export default function Settings({ isOpen, onClose, onThemeChange }) {
  const [config, setConfig] = useState({
    council_models: [],
    chairman_model: '',
    theme: 'light',
    ranking_criteria: DEFAULT_CRITERIA,
    model_weights: {},
    enable_confidence: false,
    enable_dissent_tracking: true,
  });
  const [customModel, setCustomModel] = useState('');
  const [newCriterion, setNewCriterion] = useState({ name: '', description: '' });
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [activeTab, setActiveTab] = useState('models');

  useEffect(() => {
    if (isOpen) {
      loadConfig();
    }
  }, [isOpen]);

  const loadConfig = async () => {
    try {
      const data = await api.getCouncilConfig();
      setConfig(data);
      setError(null);
    } catch (e) {
      setError('Failed to load settings');
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    setSuccess(false);

    try {
      const updatedConfig = await api.updateCouncilConfig(config);
      setConfig(updatedConfig);
      setSuccess(true);

      // Notify parent about theme change
      if (onThemeChange) {
        onThemeChange(updatedConfig.theme);
      }

      setTimeout(() => setSuccess(false), 2000);
    } catch (e) {
      setError('Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  };

  const addModel = (modelId) => {
    if (!config.council_models.includes(modelId)) {
      setConfig({
        ...config,
        council_models: [...config.council_models, modelId],
      });
    }
  };

  const removeModel = (modelId) => {
    setConfig({
      ...config,
      council_models: config.council_models.filter((m) => m !== modelId),
      // If removing the chairman, reset it
      chairman_model:
        config.chairman_model === modelId ? '' : config.chairman_model,
    });
  };

  const addCustomModel = () => {
    if (customModel.trim() && !config.council_models.includes(customModel)) {
      setConfig({
        ...config,
        council_models: [...config.council_models, customModel.trim()],
      });
      setCustomModel('');
    }
  };

  const moveModel = (index, direction) => {
    const models = [...config.council_models];
    const newIndex = index + direction;
    if (newIndex >= 0 && newIndex < models.length) {
      [models[index], models[newIndex]] = [models[newIndex], models[index]];
      setConfig({ ...config, council_models: models });
    }
  };

  // Phase 4: Ranking Criteria Management
  const addCriterion = () => {
    if (newCriterion.name.trim()) {
      const id = newCriterion.name.toLowerCase().replace(/\s+/g, '_');
      const criteria = [...(config.ranking_criteria || [])];
      if (!criteria.find(c => c.id === id)) {
        criteria.push({
          id,
          name: newCriterion.name.trim(),
          description: newCriterion.description.trim(),
          weight: 1.0,
          enabled: true,
        });
        setConfig({ ...config, ranking_criteria: criteria });
        setNewCriterion({ name: '', description: '' });
      }
    }
  };

  const removeCriterion = (id) => {
    const criteria = (config.ranking_criteria || []).filter(c => c.id !== id);
    setConfig({ ...config, ranking_criteria: criteria });
  };

  const toggleCriterion = (id) => {
    const criteria = (config.ranking_criteria || []).map(c =>
      c.id === id ? { ...c, enabled: !c.enabled } : c
    );
    setConfig({ ...config, ranking_criteria: criteria });
  };

  const updateCriterionWeight = (id, weight) => {
    const criteria = (config.ranking_criteria || []).map(c =>
      c.id === id ? { ...c, weight: parseFloat(weight) || 1.0 } : c
    );
    setConfig({ ...config, ranking_criteria: criteria });
  };

  // Phase 4: Model Weights Management
  const updateModelWeight = (modelId, weight) => {
    const weights = { ...(config.model_weights || {}) };
    weights[modelId] = parseFloat(weight) || 1.0;
    setConfig({ ...config, model_weights: weights });
  };

  const getModelWeight = (modelId) => {
    return (config.model_weights || {})[modelId] || 1.0;
  };

  // Get short model name for display
  const getShortModelName = (modelId) => {
    const parts = modelId.split('/');
    return parts.length > 1 ? parts[1] : modelId;
  };

  if (!isOpen) return null;

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="close-button" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="settings-content">
          {error && <div className="settings-error">{error}</div>}
          {success && <div className="settings-success">Settings saved!</div>}

          {/* Settings Tabs */}
          <div className="settings-tabs">
            <button
              className={`settings-tab ${activeTab === 'models' ? 'active' : ''}`}
              onClick={() => setActiveTab('models')}
            >
              Models
            </button>
            <button
              className={`settings-tab ${activeTab === 'deliberation' ? 'active' : ''}`}
              onClick={() => setActiveTab('deliberation')}
            >
              Deliberation
            </button>
            <button
              className={`settings-tab ${activeTab === 'display' ? 'active' : ''}`}
              onClick={() => setActiveTab('display')}
            >
              Display
            </button>
          </div>

          {/* Models Tab */}
          {activeTab === 'models' && (
            <>
              {/* Council Models */}
              <section className="settings-section">
                <h3>Council Models</h3>
                <p className="settings-description">
                  Select the models that will participate in the council deliberation.
                </p>

                <div className="model-list">
                  {config.council_models.map((model, index) => (
                    <div key={model} className="model-item">
                      <div className="model-info">
                        <span className="model-name">{model}</span>
                        {isReasoningModel(model) && (
                          <span className="reasoning-badge" title="Reasoning model - uses extended thinking">R</span>
                        )}
                        {model === config.chairman_model && (
                          <span className="chairman-badge">Chairman</span>
                        )}
                      </div>
                      <div className="model-actions">
                        <button
                          className="model-action-btn"
                          onClick={() => moveModel(index, -1)}
                          disabled={index === 0}
                          title="Move up"
                        >
                          ↑
                        </button>
                        <button
                          className="model-action-btn"
                          onClick={() => moveModel(index, 1)}
                          disabled={index === config.council_models.length - 1}
                          title="Move down"
                        >
                          ↓
                        </button>
                        <button
                          className="model-action-btn remove"
                          onClick={() => removeModel(model)}
                          title="Remove"
                        >
                          &times;
                        </button>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Add Model Dropdown */}
                <div className="add-model">
                  <select
                    onChange={(e) => {
                      if (e.target.value) {
                        addModel(e.target.value);
                        e.target.value = '';
                      }
                    }}
                    defaultValue=""
                  >
                    <option value="">Add a model...</option>
                    {AVAILABLE_MODELS.filter(
                      (m) => !config.council_models.includes(m.id)
                    ).map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name} ({model.provider}){model.isReasoning ? ' [Reasoning]' : ''}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Custom Model Input */}
                <div className="custom-model">
                  <input
                    type="text"
                    placeholder="Or enter custom model ID (e.g., provider/model-name)"
                    value={customModel}
                    onChange={(e) => setCustomModel(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && addCustomModel()}
                  />
                  <button onClick={addCustomModel} disabled={!customModel.trim()}>
                    Add
                  </button>
                </div>
              </section>

              {/* Chairman Selection */}
              <section className="settings-section">
                <h3>Chairman Model</h3>
                <p className="settings-description">
                  Select the model that will synthesize the final answer from all council responses.
                </p>
                <select
                  className="chairman-select"
                  value={config.chairman_model}
                  onChange={(e) =>
                    setConfig({ ...config, chairman_model: e.target.value })
                  }
                >
                  <option value="">Select chairman...</option>
                  {config.council_models.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </section>

              {/* Model Weights */}
              <section className="settings-section">
                <h3>Model Voting Weights</h3>
                <p className="settings-description">
                  Adjust how much influence each model has in the ranking aggregation.
                  Higher weights give more voting power.
                </p>
                <div className="weight-list">
                  {config.council_models.map((model) => (
                    <div key={model} className="weight-item">
                      <span className="weight-model-name">{getShortModelName(model)}</span>
                      <input
                        type="range"
                        min="0.5"
                        max="2.0"
                        step="0.1"
                        value={getModelWeight(model)}
                        onChange={(e) => updateModelWeight(model, e.target.value)}
                        className="weight-slider"
                      />
                      <span className="weight-value">{getModelWeight(model).toFixed(1)}x</span>
                    </div>
                  ))}
                </div>
              </section>
            </>
          )}

          {/* Deliberation Tab */}
          {activeTab === 'deliberation' && (
            <>
              {/* Ranking Criteria */}
              <section className="settings-section">
                <h3>Ranking Criteria</h3>
                <p className="settings-description">
                  Define the criteria models use to evaluate each other&apos;s responses in Stage 2.
                </p>

                <div className="criteria-list">
                  {(config.ranking_criteria || []).map((criterion) => (
                    <div key={criterion.id} className={`criteria-item ${!criterion.enabled ? 'disabled' : ''}`}>
                      <div className="criteria-header">
                        <label className="criteria-toggle">
                          <input
                            type="checkbox"
                            checked={criterion.enabled}
                            onChange={() => toggleCriterion(criterion.id)}
                          />
                          <span className="criteria-name">{criterion.name}</span>
                        </label>
                        <button
                          className="criteria-remove"
                          onClick={() => removeCriterion(criterion.id)}
                          title="Remove criterion"
                        >
                          &times;
                        </button>
                      </div>
                      <p className="criteria-description">{criterion.description}</p>
                      {criterion.enabled && (
                        <div className="criteria-weight">
                          <span>Weight:</span>
                          <input
                            type="range"
                            min="0.5"
                            max="2.0"
                            step="0.1"
                            value={criterion.weight}
                            onChange={(e) => updateCriterionWeight(criterion.id, e.target.value)}
                          />
                          <span>{criterion.weight.toFixed(1)}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Add New Criterion */}
                <div className="add-criterion">
                  <input
                    type="text"
                    placeholder="Criterion name"
                    value={newCriterion.name}
                    onChange={(e) => setNewCriterion({ ...newCriterion, name: e.target.value })}
                  />
                  <input
                    type="text"
                    placeholder="Description"
                    value={newCriterion.description}
                    onChange={(e) => setNewCriterion({ ...newCriterion, description: e.target.value })}
                  />
                  <button onClick={addCriterion} disabled={!newCriterion.name.trim()}>
                    Add Criterion
                  </button>
                </div>
              </section>

              {/* Feature Toggles */}
              <section className="settings-section">
                <h3>Advanced Features</h3>

                <div className="toggle-list">
                  <label className="toggle-item">
                    <input
                      type="checkbox"
                      checked={config.enable_dissent_tracking}
                      onChange={(e) => setConfig({ ...config, enable_dissent_tracking: e.target.checked })}
                    />
                    <div className="toggle-info">
                      <span className="toggle-name">Dissent Tracking</span>
                      <span className="toggle-description">
                        Track disagreements between models and highlight controversial responses
                      </span>
                    </div>
                  </label>

                  <label className="toggle-item">
                    <input
                      type="checkbox"
                      checked={config.enable_confidence}
                      onChange={(e) => setConfig({ ...config, enable_confidence: e.target.checked })}
                    />
                    <div className="toggle-info">
                      <span className="toggle-name">Confidence Scores</span>
                      <span className="toggle-description">
                        Ask models to rate their confidence in their responses (coming soon)
                      </span>
                    </div>
                  </label>
                </div>
              </section>
            </>
          )}

          {/* Display Tab */}
          {activeTab === 'display' && (
            <section className="settings-section">
              <h3>Theme</h3>
              <div className="theme-toggle">
                <button
                  className={`theme-button ${config.theme === 'light' ? 'active' : ''}`}
                  onClick={() => setConfig({ ...config, theme: 'light' })}
                >
                  Light
                </button>
                <button
                  className={`theme-button ${config.theme === 'dark' ? 'active' : ''}`}
                  onClick={() => setConfig({ ...config, theme: 'dark' })}
                >
                  Dark
                </button>
              </div>
            </section>
          )}
        </div>

        <div className="settings-footer">
          <button className="cancel-button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="save-button"
            onClick={handleSave}
            disabled={isSaving || config.council_models.length === 0}
          >
            {isSaving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
}
