import { useState, useEffect, useCallback } from 'react';
import { api } from '../api';
import './Settings.css';

// Helper to check if a model is a reasoning model
const isReasoningModel = (modelId) => {
  const patterns = ['o1', 'o3', 'reasoning', 'thinking', 'reasoner'];
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
    model_parameters: {},
    enable_confidence: false,
    enable_dissent_tracking: true,
  });
  const [customModel, setCustomModel] = useState('');
  const [newCriterion, setNewCriterion] = useState({ name: '', description: '' });
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [activeTab, setActiveTab] = useState('models');

  // Preset state
  const [presets, setPresets] = useState([]);
  const [activePreset, setActivePreset] = useState(null);
  const [newPresetName, setNewPresetName] = useState('');
  const [newPresetDescription, setNewPresetDescription] = useState('');
  const [showSavePreset, setShowSavePreset] = useState(false);

  // Model discovery state
  const [popularModels, setPopularModels] = useState([]);
  const [searchedModels, setSearchedModels] = useState([]);
  const [modelSearch, setModelSearch] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [showModelSearch, setShowModelSearch] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadConfig();
      loadPresets();
      loadPopularModels();
    }
  }, [isOpen]);

  const loadPopularModels = async () => {
    try {
      const data = await api.getPopularModels();
      setPopularModels(data.models || []);
    } catch (e) {
      console.error('Failed to load popular models:', e);
    }
  };

  // Debounced model search
  useEffect(() => {
    if (!modelSearch.trim()) {
      setSearchedModels([]);
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const data = await api.getAvailableModels(modelSearch.trim());
        setSearchedModels(data.models || []);
      } catch (e) {
        console.error('Failed to search models:', e);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [modelSearch]);

  const loadConfig = async () => {
    try {
      const data = await api.getCouncilConfig();
      setConfig(data);
      setError(null);
    } catch (e) {
      setError('Failed to load settings');
    }
  };

  const loadPresets = async () => {
    try {
      const [presetList, defaultPreset] = await Promise.all([
        api.listPresets(),
        api.getDefaultPreset(),
      ]);
      setPresets(presetList);
      setActivePreset(defaultPreset.preset);
    } catch (e) {
      console.error('Failed to load presets:', e);
    }
  };

  const handleApplyPreset = async (presetId) => {
    try {
      const result = await api.applyPreset(presetId);
      setConfig({
        ...config,
        council_models: result.config.council_models,
        chairman_model: result.config.chairman_model,
        model_weights: result.config.model_weights || {},
        model_parameters: result.config.model_parameters || {},
        ranking_criteria: result.config.ranking_criteria || config.ranking_criteria,
      });
      await loadPresets();
      setSuccess(true);
      setTimeout(() => setSuccess(false), 2000);
    } catch (e) {
      setError('Failed to apply preset');
    }
  };

  const handleSavePreset = async () => {
    if (!newPresetName.trim()) return;

    try {
      await api.createPreset({
        name: newPresetName.trim(),
        description: newPresetDescription.trim() || null,
        council_models: config.council_models,
        chairman_model: config.chairman_model,
        model_weights: config.model_weights,
        model_parameters: config.model_parameters,
        ranking_criteria: config.ranking_criteria,
      });
      await loadPresets();
      setNewPresetName('');
      setNewPresetDescription('');
      setShowSavePreset(false);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 2000);
    } catch (e) {
      setError('Failed to save preset');
    }
  };

  const handleDeletePreset = async (presetId) => {
    if (!confirm('Delete this preset?')) return;

    try {
      await api.deletePreset(presetId);
      await loadPresets();
    } catch (e) {
      setError('Failed to delete preset');
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

  // Phase 5: Model Parameters Management
  const getModelParameter = (modelId, param) => {
    const params = (config.model_parameters || {})[modelId] || {};
    if (param === 'temperature') {
      return params.temperature ?? '';
    }
    if (param === 'max_tokens') {
      return params.max_tokens ?? '';
    }
    return '';
  };

  const updateModelParameter = (modelId, param, value) => {
    const params = { ...(config.model_parameters || {}) };
    if (!params[modelId]) {
      params[modelId] = {};
    }
    if (value === '' || value === null || value === undefined) {
      delete params[modelId][param];
      // Clean up empty objects
      if (Object.keys(params[modelId]).length === 0) {
        delete params[modelId];
      }
    } else {
      if (param === 'temperature') {
        params[modelId][param] = parseFloat(value);
      } else if (param === 'max_tokens') {
        params[modelId][param] = parseInt(value, 10);
      }
    }
    setConfig({ ...config, model_parameters: params });
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
              className={`settings-tab ${activeTab === 'presets' ? 'active' : ''}`}
              onClick={() => setActiveTab('presets')}
            >
              Presets
            </button>
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

          {/* Presets Tab */}
          {activeTab === 'presets' && (
            <>
              <section className="settings-section">
                <h3>Council Presets</h3>
                <p className="settings-description">
                  Save and load different council configurations quickly.
                </p>

                {/* Current Config Summary */}
                <div className="current-config-summary">
                  <h4>Current Configuration</h4>
                  <div className="config-summary-content">
                    <span className="config-summary-item">
                      <strong>Models:</strong> {config.council_models.length}
                    </span>
                    <span className="config-summary-item">
                      <strong>Chairman:</strong> {config.chairman_model ? getShortModelName(config.chairman_model) : 'Not set'}
                    </span>
                  </div>
                  {!showSavePreset ? (
                    <button
                      className="save-preset-btn"
                      onClick={() => setShowSavePreset(true)}
                      disabled={config.council_models.length === 0}
                    >
                      Save as Preset
                    </button>
                  ) : (
                    <div className="save-preset-form">
                      <input
                        type="text"
                        placeholder="Preset name"
                        value={newPresetName}
                        onChange={(e) => setNewPresetName(e.target.value)}
                      />
                      <input
                        type="text"
                        placeholder="Description (optional)"
                        value={newPresetDescription}
                        onChange={(e) => setNewPresetDescription(e.target.value)}
                      />
                      <div className="save-preset-actions">
                        <button onClick={handleSavePreset} disabled={!newPresetName.trim()}>
                          Save
                        </button>
                        <button onClick={() => setShowSavePreset(false)} className="cancel">
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Preset List */}
                <div className="preset-list">
                  {presets.length === 0 ? (
                    <p className="no-presets">No presets saved yet. Create one above!</p>
                  ) : (
                    presets.map((preset) => (
                      <div
                        key={preset.id}
                        className={`preset-item ${activePreset?.id === preset.id ? 'active' : ''}`}
                      >
                        <div className="preset-info">
                          <span className="preset-name">
                            {preset.name}
                            {activePreset?.id === preset.id && (
                              <span className="active-badge">Active</span>
                            )}
                          </span>
                          {preset.description && (
                            <span className="preset-description">{preset.description}</span>
                          )}
                          <span className="preset-details">
                            {preset.council_models.length} models &middot; Chairman: {getShortModelName(preset.chairman_model)}
                          </span>
                        </div>
                        <div className="preset-actions">
                          <button
                            className="apply-preset-btn"
                            onClick={() => handleApplyPreset(preset.id)}
                            disabled={activePreset?.id === preset.id}
                          >
                            {activePreset?.id === preset.id ? 'Active' : 'Apply'}
                          </button>
                          <button
                            className="delete-preset-btn"
                            onClick={() => handleDeletePreset(preset.id)}
                            title="Delete preset"
                          >
                            &times;
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </section>
            </>
          )}

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

                {/* Add Model - Popular Models Dropdown */}
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
                    <option value="">Add from popular models...</option>
                    {popularModels.filter(
                      (m) => !config.council_models.includes(m.id)
                    ).map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name}{model.is_reasoning ? ' [Reasoning]' : ''}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Model Search */}
                <div className="model-search-container">
                  <button
                    className="model-search-toggle"
                    onClick={() => setShowModelSearch(!showModelSearch)}
                  >
                    {showModelSearch ? 'Hide Search' : 'Search All Models'}
                  </button>

                  {showModelSearch && (
                    <div className="model-search">
                      <input
                        type="text"
                        placeholder="Search OpenRouter models..."
                        value={modelSearch}
                        onChange={(e) => setModelSearch(e.target.value)}
                      />
                      {isSearching && <span className="search-indicator">Searching...</span>}

                      {searchedModels.length > 0 && (
                        <div className="search-results">
                          {searchedModels.filter(
                            (m) => !config.council_models.includes(m.id)
                          ).slice(0, 20).map((model) => (
                            <div
                              key={model.id}
                              className="search-result-item"
                              onClick={() => {
                                addModel(model.id);
                                setModelSearch('');
                                setSearchedModels([]);
                              }}
                            >
                              <div className="search-result-name">
                                {model.name}
                                {model.is_reasoning && <span className="reasoning-tag">R</span>}
                              </div>
                              <div className="search-result-id">{model.id}</div>
                              {model.pricing && (
                                <div className="search-result-pricing">
                                  ${model.pricing.prompt_per_million?.toFixed(2) || '?'}/M input
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
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

              {/* Model Parameters */}
              <section className="settings-section">
                <h3>Model Parameters</h3>
                <p className="settings-description">
                  Configure temperature and max tokens for each model. Leave blank to use defaults.
                  Reasoning models (o1, o3) ignore temperature settings.
                </p>
                <div className="parameters-list">
                  {config.council_models.map((model) => (
                    <div key={model} className="parameters-item">
                      <span className="parameters-model-name">
                        {getShortModelName(model)}
                        {isReasoningModel(model) && (
                          <span className="reasoning-note">(reasoning)</span>
                        )}
                      </span>
                      <div className="parameters-inputs">
                        <div className="parameter-field">
                          <label>Temperature</label>
                          <input
                            type="number"
                            min="0"
                            max="2"
                            step="0.1"
                            placeholder="0.7"
                            value={getModelParameter(model, 'temperature')}
                            onChange={(e) => updateModelParameter(model, 'temperature', e.target.value)}
                            disabled={isReasoningModel(model)}
                            title={isReasoningModel(model) ? 'Reasoning models do not support temperature' : 'Response creativity (0=deterministic, 2=creative)'}
                          />
                        </div>
                        <div className="parameter-field">
                          <label>Max Tokens</label>
                          <input
                            type="number"
                            min="100"
                            max="32000"
                            step="100"
                            placeholder="4096"
                            value={getModelParameter(model, 'max_tokens')}
                            onChange={(e) => updateModelParameter(model, 'max_tokens', e.target.value)}
                            title="Maximum response length in tokens"
                          />
                        </div>
                      </div>
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
