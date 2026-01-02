/**
 * API client for the LLM Council backend.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export const api = {
  /**
   * List all conversations.
   */
  async listConversations() {
    const response = await fetch(`${API_BASE}/api/conversations`);
    if (!response.ok) {
      throw new Error('Failed to list conversations');
    }
    return response.json();
  },

  /**
   * Create a new conversation.
   */
  async createConversation() {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      throw new Error('Failed to create conversation');
    }
    return response.json();
  },

  /**
   * Get a specific conversation.
   */
  async getConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}`
    );
    if (!response.ok) {
      throw new Error('Failed to get conversation');
    }
    return response.json();
  },

  /**
   * Delete a conversation.
   */
  async deleteConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}`,
      { method: 'DELETE' }
    );
    if (!response.ok) {
      throw new Error('Failed to delete conversation');
    }
    return response.json();
  },

  /**
   * Send a message in a conversation.
   */
  async sendMessage(conversationId, content) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    return response.json();
  },

  /**
   * Send a message and receive streaming updates.
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {function} onEvent - Callback function for each event: (eventType, data) => void
   * @returns {Promise<void>}
   */
  async sendMessageStream(conversationId, content, onEvent) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
      }
    );

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep the last incomplete line in the buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e, 'Data:', data);
          }
        }
      }
    }

    // Process any remaining data in buffer
    if (buffer.startsWith('data: ')) {
      try {
        const event = JSON.parse(buffer.slice(6));
        onEvent(event.type, event);
      } catch (e) {
        console.error('Failed to parse final SSE event:', e);
      }
    }
  },

  /**
   * Export a conversation to markdown or JSON.
   */
  async exportConversation(conversationId, format = 'markdown') {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/export`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ format }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to export conversation');
    }

    if (format === 'markdown') {
      return response.text();
    }
    return response.json();
  },

  // Settings API

  /**
   * Get the current council configuration.
   */
  async getCouncilConfig() {
    const response = await fetch(`${API_BASE}/api/settings/council`);
    if (!response.ok) {
      throw new Error('Failed to get council config');
    }
    return response.json();
  },

  /**
   * Update the council configuration.
   */
  async updateCouncilConfig(config) {
    const response = await fetch(`${API_BASE}/api/settings/council`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });
    if (!response.ok) {
      throw new Error('Failed to update council config');
    }
    return response.json();
  },

  /**
   * Send a message with token-by-token streaming for Stage 3.
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {function} onEvent - Callback function for each event: (eventType, data) => void
   * @returns {Promise<void>}
   */
  async sendMessageStreamTokens(conversationId, content, onEvent) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/stream-tokens`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
      }
    );

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep the last incomplete line in the buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e, 'Data:', data);
          }
        }
      }
    }

    // Process any remaining data in buffer
    if (buffer.startsWith('data: ')) {
      try {
        const event = JSON.parse(buffer.slice(6));
        onEvent(event.type, event);
      } catch (e) {
        console.error('Failed to parse final SSE event:', e);
      }
    }
  },

  /**
   * Get circuit breaker status for all models.
   */
  async getCircuitBreakerStatus() {
    const response = await fetch(`${API_BASE}/api/circuit-breaker/status`);
    if (!response.ok) {
      throw new Error('Failed to get circuit breaker status');
    }
    return response.json();
  },

  // Analytics API

  /**
   * Get analytics summary across all conversations.
   */
  async getAnalyticsSummary() {
    const response = await fetch(`${API_BASE}/api/analytics/summary`);
    if (!response.ok) {
      throw new Error('Failed to get analytics summary');
    }
    return response.json();
  },

  /**
   * Get detailed analytics for a specific model.
   */
  async getModelAnalytics(modelId) {
    const response = await fetch(
      `${API_BASE}/api/analytics/models/${encodeURIComponent(modelId)}`
    );
    if (!response.ok) {
      throw new Error('Failed to get model analytics');
    }
    return response.json();
  },

  /**
   * Get recent metrics for timeline display.
   */
  async getRecentMetrics(limit = 100) {
    const response = await fetch(`${API_BASE}/api/analytics/recent?limit=${limit}`);
    if (!response.ok) {
      throw new Error('Failed to get recent metrics');
    }
    return response.json();
  },

  // Preset API

  /**
   * List all presets.
   */
  async listPresets() {
    const response = await fetch(`${API_BASE}/api/presets`);
    if (!response.ok) {
      throw new Error('Failed to list presets');
    }
    return response.json();
  },

  /**
   * Get the currently active preset.
   */
  async getDefaultPreset() {
    const response = await fetch(`${API_BASE}/api/presets/default`);
    if (!response.ok) {
      throw new Error('Failed to get default preset');
    }
    return response.json();
  },

  /**
   * Get a specific preset by ID.
   */
  async getPreset(presetId) {
    const response = await fetch(`${API_BASE}/api/presets/${presetId}`);
    if (!response.ok) {
      throw new Error('Failed to get preset');
    }
    return response.json();
  },

  /**
   * Create a new preset.
   */
  async createPreset(preset) {
    const response = await fetch(`${API_BASE}/api/presets`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(preset),
    });
    if (!response.ok) {
      throw new Error('Failed to create preset');
    }
    return response.json();
  },

  /**
   * Update an existing preset.
   */
  async updatePreset(presetId, preset) {
    const response = await fetch(`${API_BASE}/api/presets/${presetId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(preset),
    });
    if (!response.ok) {
      throw new Error('Failed to update preset');
    }
    return response.json();
  },

  /**
   * Delete a preset.
   */
  async deletePreset(presetId) {
    const response = await fetch(`${API_BASE}/api/presets/${presetId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to delete preset');
    }
    return response.json();
  },

  /**
   * Apply a preset to the current council configuration.
   */
  async applyPreset(presetId) {
    const response = await fetch(`${API_BASE}/api/presets/${presetId}/apply`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error('Failed to apply preset');
    }
    return response.json();
  },

  /**
   * Save the current configuration as a new preset.
   */
  async saveCurrentAsPreset(name, description = null) {
    const params = new URLSearchParams({ name });
    if (description) {
      params.append('description', description);
    }
    const response = await fetch(`${API_BASE}/api/presets/save-current?${params}`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error('Failed to save current config as preset');
    }
    return response.json();
  },

  // Model Discovery API

  /**
   * Get all available models from OpenRouter.
   * @param {string} search - Optional search term to filter models
   * @param {boolean} refresh - Force refresh from API instead of cache
   */
  async getAvailableModels(search = null, refresh = false) {
    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (refresh) params.append('refresh', 'true');

    const url = `${API_BASE}/api/models${params.toString() ? '?' + params : ''}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to get available models');
    }
    return response.json();
  },

  /**
   * Get curated list of popular models.
   */
  async getPopularModels() {
    const response = await fetch(`${API_BASE}/api/models/popular`);
    if (!response.ok) {
      throw new Error('Failed to get popular models');
    }
    return response.json();
  },

  /**
   * Send a message using the Dialectic Chain workflow.
   * This is a fundamentally different approach:
   * - Sequential refinement instead of parallel voting
   * - Each model builds on and challenges the previous
   * - Produces more specific, actionable, life-changing output
   *
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {function} onEvent - Callback function for each event: (eventType, data) => void
   * @returns {Promise<void>}
   */
  async sendMessageDialectic(conversationId, content, onEvent) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/dialectic`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
      }
    );

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep the last incomplete line in the buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e, 'Data:', data);
          }
        }
      }
    }

    // Process any remaining data in buffer
    if (buffer.startsWith('data: ')) {
      try {
        const event = JSON.parse(buffer.slice(6));
        onEvent(event.type, event);
      } catch (e) {
        console.error('Failed to parse final SSE event:', e);
      }
    }
  },
};
