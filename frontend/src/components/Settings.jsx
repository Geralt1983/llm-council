import { useState, useEffect } from 'react';
import { api } from '../api';
import './Settings.css';

// Common OpenRouter models
const AVAILABLE_MODELS = [
  { id: 'openai/gpt-4o', name: 'GPT-4o', provider: 'OpenAI' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini', provider: 'OpenAI' },
  { id: 'openai/o1', name: 'o1', provider: 'OpenAI' },
  { id: 'openai/o1-mini', name: 'o1 Mini', provider: 'OpenAI' },
  { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', provider: 'Anthropic' },
  { id: 'anthropic/claude-3-opus', name: 'Claude 3 Opus', provider: 'Anthropic' },
  { id: 'anthropic/claude-3-haiku', name: 'Claude 3 Haiku', provider: 'Anthropic' },
  { id: 'google/gemini-2.0-flash', name: 'Gemini 2.0 Flash', provider: 'Google' },
  { id: 'google/gemini-2.5-flash', name: 'Gemini 2.5 Flash', provider: 'Google' },
  { id: 'google/gemini-pro-1.5', name: 'Gemini Pro 1.5', provider: 'Google' },
  { id: 'x-ai/grok-2', name: 'Grok 2', provider: 'xAI' },
  { id: 'meta-llama/llama-3.1-405b-instruct', name: 'Llama 3.1 405B', provider: 'Meta' },
  { id: 'meta-llama/llama-3.1-70b-instruct', name: 'Llama 3.1 70B', provider: 'Meta' },
  { id: 'mistralai/mistral-large', name: 'Mistral Large', provider: 'Mistral' },
  { id: 'deepseek/deepseek-chat', name: 'DeepSeek Chat', provider: 'DeepSeek' },
  { id: 'perplexity/sonar-pro', name: 'Sonar Pro', provider: 'Perplexity' },
];

export default function Settings({ isOpen, onClose, onThemeChange }) {
  const [config, setConfig] = useState({
    council_models: [],
    chairman_model: '',
    theme: 'light',
  });
  const [customModel, setCustomModel] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

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

          {/* Theme Selection */}
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
                    {model.name} ({model.provider})
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
