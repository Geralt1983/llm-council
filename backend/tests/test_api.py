"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    # Mock the database initialization
    with patch("backend.main.init_db"):
        with patch("backend.main.migrate_from_json"):
            from backend.main import app
            return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test root endpoint returns health status."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "LLM Council API"


class TestConversationEndpoints:
    """Tests for conversation CRUD endpoints."""

    def test_list_conversations(self, client):
        """Test listing conversations."""
        with patch("backend.main.storage.list_conversations") as mock_list:
            mock_list.return_value = [
                {"id": "123", "title": "Test", "created_at": "2024-01-01", "message_count": 0}
            ]

            response = client.get("/api/conversations")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == "123"

    def test_create_conversation(self, client):
        """Test creating a new conversation."""
        with patch("backend.main.storage.create_conversation") as mock_create:
            mock_create.return_value = {
                "id": "new-123",
                "title": "New Conversation",
                "created_at": "2024-01-01T00:00:00",
                "messages": []
            }

            response = client.post(
                "/api/conversations",
                json={}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "new-123"

    def test_get_conversation(self, client):
        """Test getting a specific conversation."""
        with patch("backend.main.storage.get_conversation") as mock_get:
            mock_get.return_value = {
                "id": "123",
                "title": "Test Conversation",
                "created_at": "2024-01-01T00:00:00",
                "messages": []
            }

            response = client.get("/api/conversations/123")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "123"

    def test_get_conversation_not_found(self, client):
        """Test getting non-existent conversation returns 404."""
        with patch("backend.main.storage.get_conversation") as mock_get:
            mock_get.return_value = None

            response = client.get("/api/conversations/nonexistent")

            assert response.status_code == 404

    def test_delete_conversation(self, client):
        """Test deleting a conversation."""
        with patch("backend.main.storage.delete_conversation") as mock_delete:
            mock_delete.return_value = True

            response = client.delete("/api/conversations/123")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "deleted"

    def test_delete_conversation_not_found(self, client):
        """Test deleting non-existent conversation returns 404."""
        with patch("backend.main.storage.delete_conversation") as mock_delete:
            mock_delete.return_value = False

            response = client.delete("/api/conversations/nonexistent")

            assert response.status_code == 404


class TestSettingsEndpoints:
    """Tests for settings endpoints."""

    def test_get_council_config(self, client):
        """Test getting council configuration."""
        with patch("backend.main.storage.get_council_config") as mock_get:
            mock_get.return_value = {
                "council_models": ["openai/gpt-4o"],
                "chairman_model": "openai/gpt-4o",
                "theme": "light"
            }

            response = client.get("/api/settings/council")

            assert response.status_code == 200
            data = response.json()
            assert "council_models" in data
            assert data["theme"] == "light"

    def test_update_council_config(self, client):
        """Test updating council configuration."""
        with patch("backend.main.storage.update_council_config") as mock_update:
            mock_update.return_value = {
                "council_models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
                "chairman_model": "openai/gpt-4o",
                "theme": "dark"
            }

            response = client.put(
                "/api/settings/council",
                json={
                    "council_models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
                    "theme": "dark"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["council_models"]) == 2
            assert data["theme"] == "dark"


class TestCircuitBreakerEndpoint:
    """Tests for circuit breaker status endpoint."""

    def test_get_circuit_breaker_status(self, client):
        """Test getting circuit breaker status."""
        with patch("backend.main.get_council_models") as mock_council:
            with patch("backend.main.get_chairman_model") as mock_chairman:
                mock_council.return_value = ["openai/gpt-4o"]
                mock_chairman.return_value = "openai/gpt-4o"

                with patch("backend.main.CircuitBreaker.get_status") as mock_status:
                    mock_status.return_value = "CLOSED"

                    response = client.get("/api/circuit-breaker/status")

                    assert response.status_code == 200
                    data = response.json()
                    assert "circuit_breaker_status" in data


class TestExportEndpoint:
    """Tests for conversation export endpoint."""

    def test_export_json(self, client):
        """Test exporting conversation as JSON."""
        with patch("backend.main.storage.get_conversation") as mock_get:
            mock_get.return_value = {
                "id": "123",
                "title": "Test",
                "created_at": "2024-01-01T00:00:00",
                "messages": []
            }

            response = client.post(
                "/api/conversations/123/export",
                json={"format": "json"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "123"

    def test_export_markdown(self, client):
        """Test exporting conversation as markdown."""
        with patch("backend.main.storage.get_conversation") as mock_get:
            mock_get.return_value = {
                "id": "123",
                "title": "Test Conversation",
                "created_at": "2024-01-01T00:00:00",
                "messages": []
            }

            response = client.post(
                "/api/conversations/123/export",
                json={"format": "markdown"}
            )

            assert response.status_code == 200
            assert "Test Conversation" in response.text

    def test_export_invalid_format(self, client):
        """Test exporting with invalid format returns 400."""
        with patch("backend.main.storage.get_conversation") as mock_get:
            mock_get.return_value = {
                "id": "123",
                "title": "Test",
                "created_at": "2024-01-01T00:00:00",
                "messages": []
            }

            response = client.post(
                "/api/conversations/123/export",
                json={"format": "pdf"}  # Invalid format
            )

            assert response.status_code == 400
