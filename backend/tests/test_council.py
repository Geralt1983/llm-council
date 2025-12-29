"""Tests for council orchestration."""

import pytest
from unittest.mock import patch, AsyncMock


class TestBuildMessagesWithHistory:
    """Tests for build_messages_with_history function."""

    def test_builds_messages_without_history(self):
        """Test building messages without conversation history."""
        from backend.council import build_messages_with_history

        messages = build_messages_with_history("What is 2+2?")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is 2+2?"

    def test_builds_messages_with_history(self):
        """Test building messages with conversation history."""
        from backend.council import build_messages_with_history

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        messages = build_messages_with_history("How are you?", history)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "How are you?"


class TestParseRankingFromText:
    """Tests for parse_ranking_from_text function."""

    def test_parses_standard_format(self):
        """Test parsing standard FINAL RANKING format."""
        from backend.council import parse_ranking_from_text

        text = """Response A is good but lacks depth.
Response B provides comprehensive coverage.

FINAL RANKING:
1. Response B
2. Response A
3. Response C"""

        result = parse_ranking_from_text(text)

        assert result == ["Response B", "Response A", "Response C"]

    def test_parses_without_numbers(self):
        """Test parsing when numbers are missing."""
        from backend.council import parse_ranking_from_text

        text = """Some evaluation text...

FINAL RANKING:
Response C
Response A
Response B"""

        result = parse_ranking_from_text(text)

        assert result == ["Response C", "Response A", "Response B"]

    def test_fallback_extracts_from_anywhere(self):
        """Test fallback extracts Response X patterns from anywhere."""
        from backend.council import parse_ranking_from_text

        text = """Based on my analysis, Response B is best,
followed by Response A, and then Response C."""

        result = parse_ranking_from_text(text)

        assert result == ["Response B", "Response A", "Response C"]

    def test_handles_empty_text(self):
        """Test handling empty or invalid text."""
        from backend.council import parse_ranking_from_text

        result = parse_ranking_from_text("")
        assert result == []

        result = parse_ranking_from_text("No responses mentioned here.")
        assert result == []


class TestCalculateAggregateRankings:
    """Tests for calculate_aggregate_rankings function."""

    def test_calculates_average_rankings(
        self, sample_stage2_results, sample_label_to_model
    ):
        """Test calculating average rankings across evaluations."""
        from backend.council import calculate_aggregate_rankings

        result = calculate_aggregate_rankings(
            sample_stage2_results, sample_label_to_model
        )

        # Response B (Claude) should have best average (1.0)
        # because it's ranked first by both evaluators
        assert len(result) == 3

        # Find Claude in results
        claude_result = next(
            r for r in result if r["model"] == "anthropic/claude-3.5-sonnet"
        )
        assert claude_result["average_rank"] == 1.0

    def test_handles_empty_results(self, sample_label_to_model):
        """Test handling empty stage2 results."""
        from backend.council import calculate_aggregate_rankings

        result = calculate_aggregate_rankings([], sample_label_to_model)
        assert result == []


class TestStage1CollectResponses:
    """Tests for stage1_collect_responses function."""

    @pytest.mark.asyncio
    async def test_collects_responses_from_models(self, mock_council_models):
        """Test collecting responses from all council models."""
        mock_response = {
            "content": "Test response content",
            "reasoning_details": None
        }

        with patch("backend.council.get_council_models") as mock_get_models:
            mock_get_models.return_value = mock_council_models

            with patch(
                "backend.council.query_models_parallel_with_circuit_breaker"
            ) as mock_query:
                mock_query.return_value = {
                    model: mock_response for model in mock_council_models
                }

                from backend.council import stage1_collect_responses

                results = await stage1_collect_responses("What is Python?")

                assert len(results) == 3
                for result in results:
                    assert "model" in result
                    assert "response" in result
                    assert result["response"] == "Test response content"

    @pytest.mark.asyncio
    async def test_includes_conversation_history(self, mock_council_models):
        """Test that conversation history is passed to models."""
        with patch("backend.council.get_council_models") as mock_get_models:
            mock_get_models.return_value = mock_council_models

            with patch(
                "backend.council.query_models_parallel_with_circuit_breaker"
            ) as mock_query:
                mock_query.return_value = {model: {"content": "Test"} for model in mock_council_models}

                from backend.council import stage1_collect_responses

                history = [
                    {"role": "user", "content": "Previous question"},
                    {"role": "assistant", "content": "Previous answer"},
                ]

                await stage1_collect_responses(
                    "Follow-up question",
                    conversation_history=history
                )

                # Check that query was called with proper messages
                call_args = mock_query.call_args
                messages = call_args[0][1]  # Second positional arg

                # Should have history + current question
                assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_filters_failed_responses(self, mock_council_models):
        """Test that failed responses are filtered out."""
        with patch("backend.council.get_council_models") as mock_get_models:
            mock_get_models.return_value = mock_council_models

            with patch(
                "backend.council.query_models_parallel_with_circuit_breaker"
            ) as mock_query:
                # One model fails
                mock_query.return_value = {
                    mock_council_models[0]: {"content": "Success"},
                    mock_council_models[1]: None,  # Failed
                    mock_council_models[2]: {"content": "Also success"},
                }

                from backend.council import stage1_collect_responses

                results = await stage1_collect_responses("Test query")

                assert len(results) == 2  # Only successful responses


class TestStage3SynthesizeFinal:
    """Tests for stage3_synthesize_final function."""

    @pytest.mark.asyncio
    async def test_synthesizes_from_stages(
        self, sample_stage1_results, sample_stage2_results
    ):
        """Test chairman synthesizes from stage results."""
        with patch("backend.council.get_chairman_model") as mock_get_chairman:
            mock_get_chairman.return_value = "google/gemini-2.5-flash"

            with patch(
                "backend.council.query_model_with_circuit_breaker"
            ) as mock_query:
                mock_query.return_value = {
                    "content": "Synthesized final answer based on council input."
                }

                from backend.council import stage3_synthesize_final

                result = await stage3_synthesize_final(
                    "What is AI?",
                    sample_stage1_results,
                    sample_stage2_results
                )

                assert result["model"] == "google/gemini-2.5-flash"
                assert "Synthesized final answer" in result["response"]

    @pytest.mark.asyncio
    async def test_handles_chairman_failure(
        self, sample_stage1_results, sample_stage2_results
    ):
        """Test fallback when chairman fails."""
        with patch("backend.council.get_chairman_model") as mock_get_chairman:
            mock_get_chairman.return_value = "google/gemini-2.5-flash"

            with patch(
                "backend.council.query_model_with_circuit_breaker"
            ) as mock_query:
                mock_query.return_value = None  # Chairman fails

                from backend.council import stage3_synthesize_final

                result = await stage3_synthesize_final(
                    "What is AI?",
                    sample_stage1_results,
                    sample_stage2_results
                )

                assert "Error" in result["response"]
