import os
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Unit tests for tools/llm_provider.py
# ---------------------------------------------------------------------------

class TestDetectProvider:
    """Test provider auto-detection logic."""

    def test_explicit_groq(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "groq"}, clear=False):
            from tools.llm_provider import _detect_provider
            assert _detect_provider() == "groq"

    def test_explicit_minimax(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}, clear=False):
            from tools.llm_provider import _detect_provider
            assert _detect_provider() == "minimax"

    def test_explicit_minimax_uppercase(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "MiniMax"}, clear=False):
            from tools.llm_provider import _detect_provider
            assert _detect_provider() == "minimax"

    def test_auto_detect_minimax_from_api_key(self):
        env = {"MINIMAX_API_KEY": "test-key-123"}
        remove = [k for k in ["LLM_PROVIDER"] if k in os.environ]
        with patch.dict(os.environ, env, clear=False):
            for k in remove:
                os.environ.pop(k, None)
            from tools.llm_provider import _detect_provider
            assert _detect_provider() == "minimax"

    def test_fallback_to_groq(self):
        saved = {k: os.environ.pop(k) for k in ["LLM_PROVIDER", "MINIMAX_API_KEY"] if k in os.environ}
        try:
            from tools.llm_provider import _detect_provider
            assert _detect_provider() == "groq"
        finally:
            os.environ.update(saved)


class TestClampTemperature:
    """Test MiniMax temperature clamping."""

    def test_minimax_clamp_zero(self):
        from tools.llm_provider import _clamp_temperature
        assert _clamp_temperature(0.0, "minimax") == 0.01

    def test_minimax_clamp_negative(self):
        from tools.llm_provider import _clamp_temperature
        assert _clamp_temperature(-0.5, "minimax") == 0.01

    def test_minimax_clamp_above_one(self):
        from tools.llm_provider import _clamp_temperature
        assert _clamp_temperature(1.5, "minimax") == 1.0

    def test_minimax_valid_temperature(self):
        from tools.llm_provider import _clamp_temperature
        assert _clamp_temperature(0.3, "minimax") == 0.3

    def test_groq_no_clamp(self):
        from tools.llm_provider import _clamp_temperature
        assert _clamp_temperature(0.0, "groq") == 0.0

    def test_groq_high_temp(self):
        from tools.llm_provider import _clamp_temperature
        assert _clamp_temperature(2.0, "groq") == 2.0


class TestCreateLLMGroq:
    """Test Groq LLM creation."""

    @patch("tools.llm_provider._detect_provider", return_value="groq")
    @patch("tools.llm_provider.ChatGroq", create=True)
    def test_creates_groq_instance(self, MockChatGroq, _mock_detect):
        # Patch the import inside create_llm
        mock_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_groq": MagicMock(ChatGroq=mock_cls)}):
            from importlib import reload
            import tools.llm_provider as mod
            reload(mod)
            with patch.object(mod, "_detect_provider", return_value="groq"):
                llm = mod.create_llm(temperature=0.5, max_tokens=256)
                mock_cls.assert_called_once()
                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs["model"] == "deepseek-r1-distill-llama-70b"
                assert call_kwargs["temperature"] == 0.5
                assert call_kwargs["max_tokens"] == 256


class TestCreateLLMMiniMax:
    """Test MiniMax LLM creation."""

    def test_creates_minimax_instance(self):
        mock_cls = MagicMock()
        mock_module = MagicMock(ChatOpenAI=mock_cls)
        env = {
            "LLM_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "test-key-abc",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch.dict("sys.modules", {"langchain_openai": mock_module}):
                from importlib import reload
                import tools.llm_provider as mod
                reload(mod)
                llm = mod.create_llm(temperature=0.3, max_tokens=512)
                mock_cls.assert_called_once()
                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs["model"] == "MiniMax-M2.7"
                assert call_kwargs["base_url"] == "https://api.minimax.io/v1"
                assert call_kwargs["api_key"] == "test-key-abc"
                assert call_kwargs["temperature"] == 0.3

    def test_minimax_missing_api_key_raises(self):
        env = {"LLM_PROVIDER": "minimax"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("MINIMAX_API_KEY", None)
            from importlib import reload
            import tools.llm_provider as mod
            reload(mod)
            with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                mod.create_llm()

    def test_minimax_temperature_clamped_in_create(self):
        mock_cls = MagicMock()
        mock_module = MagicMock(ChatOpenAI=mock_cls)
        env = {
            "LLM_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "test-key-abc",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch.dict("sys.modules", {"langchain_openai": mock_module}):
                from importlib import reload
                import tools.llm_provider as mod
                reload(mod)
                mod.create_llm(temperature=0.0)
                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs["temperature"] == 0.01

    def test_custom_model_override(self):
        mock_cls = MagicMock()
        mock_module = MagicMock(ChatOpenAI=mock_cls)
        env = {
            "LLM_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "test-key-abc",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch.dict("sys.modules", {"langchain_openai": mock_module}):
                from importlib import reload
                import tools.llm_provider as mod
                reload(mod)
                mod.create_llm(model="MiniMax-M2.7-highspeed")
                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs["model"] == "MiniMax-M2.7-highspeed"

    def test_env_model_override(self):
        mock_cls = MagicMock()
        mock_module = MagicMock(ChatOpenAI=mock_cls)
        env = {
            "LLM_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "test-key-abc",
            "LLM_MODEL": "MiniMax-M2.5",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch.dict("sys.modules", {"langchain_openai": mock_module}):
                from importlib import reload
                import tools.llm_provider as mod
                reload(mod)
                mod.create_llm()
                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs["model"] == "MiniMax-M2.5"
        os.environ.pop("LLM_MODEL", None)


class TestUnknownProvider:
    """Test unknown provider raises error."""

    def test_unknown_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_provider"}, clear=False):
            from importlib import reload
            import tools.llm_provider as mod
            reload(mod)
            with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
                mod.create_llm()


class TestDefaultModels:
    """Test default model mapping."""

    def test_groq_default(self):
        from tools.llm_provider import _DEFAULT_MODELS
        assert _DEFAULT_MODELS["groq"] == "deepseek-r1-distill-llama-70b"

    def test_minimax_default(self):
        from tools.llm_provider import _DEFAULT_MODELS
        assert _DEFAULT_MODELS["minimax"] == "MiniMax-M2.7"


class TestChatModuleIntegration:
    """Test that chat.py creates LLM via provider factory."""

    @patch("tools.llm_provider.create_llm")
    def test_chat_uses_provider_factory(self, mock_create):
        mock_create.return_value = MagicMock()
        from importlib import reload
        import tools.chat as chat_mod
        reload(chat_mod)
        mock_create.assert_any_call(
            temperature=0.3, max_tokens=512, max_retries=3
        )

    @patch("tools.llm_provider.create_llm")
    def test_decision_uses_provider_factory(self, mock_create):
        mock_create.return_value = MagicMock()
        from importlib import reload
        import tools.decision as decision_mod
        reload(decision_mod)
        mock_create.assert_any_call(
            temperature=0.3, max_tokens=512, max_retries=2
        )


# ---------------------------------------------------------------------------
# Integration tests (require actual API keys — skipped in CI)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set — skipping MiniMax integration tests",
)
class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_minimax_simple_invoke(self):
        api_key = os.environ["MINIMAX_API_KEY"]
        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": api_key}
        with patch.dict(os.environ, env, clear=False):
            from importlib import reload
            import tools.llm_provider as mod
            reload(mod)
            llm = mod.create_llm(temperature=0.3, max_tokens=64)
            from langchain_core.messages import HumanMessage
            resp = llm.invoke([HumanMessage(content="Say hello in one word.")])
            assert len(resp.content.strip()) > 0

    def test_minimax_chat_chain(self):
        api_key = os.environ["MINIMAX_API_KEY"]
        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": api_key}
        with patch.dict(os.environ, env, clear=False):
            from importlib import reload
            import tools.llm_provider as mod
            reload(mod)
            llm = mod.create_llm(temperature=0.3, max_tokens=64)
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                ("human", "{query}"),
            ])
            chain = prompt | llm
            resp = chain.invoke({"query": "What is 2+2? Reply with just the number."})
            assert "4" in resp.content

    def test_minimax_m27_highspeed(self):
        api_key = os.environ["MINIMAX_API_KEY"]
        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": api_key}
        with patch.dict(os.environ, env, clear=False):
            from importlib import reload
            import tools.llm_provider as mod
            reload(mod)
            llm = mod.create_llm(
                temperature=0.3, max_tokens=64,
                model="MiniMax-M2.7-highspeed",
            )
            from langchain_core.messages import HumanMessage
            resp = llm.invoke([HumanMessage(content="Say 'ok'.")])
            assert len(resp.content.strip()) > 0
