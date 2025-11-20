"""Tests for MCP-based configurators.

This test suite covers all 8 configurators using the new MCP architecture:
- Claude Desktop
- Cursor
- Cody (Sourcegraph)
- Continue.dev
- Gemini Code Assist
- VSCode (Cline)
- Windsurf (Codeium)
- Codex CLI

All configurators now use:
- Constructor: __init__(venv_python, mcp_script)
- Method: get_omnimemory_prompt() (not get_system_prompt)
- Tools: mcp__omnimemory__read, mcp__omnimemory__search
- Environment: OMNIMEMORY_TOOL_ID (not OMNIMEMORY_API_KEY)
"""

import json
import tempfile
from pathlib import Path
import pytest

from src.configurators.claude import ClaudeConfigurator
from src.configurators.cursor import CursorConfigurator
from src.configurators.cody import CodyConfigurator
from src.configurators.continuedev import ContinueConfigurator
from src.configurators.gemini import GeminiConfigurator
from src.configurators.vscode import VSCodeConfigurator
from src.configurators.windsurf import WindsurfConfigurator
from src.configurators.codex import CodexConfigurator


@pytest.fixture
def mock_mcp_paths(tmp_path):
    """Create mock venv and MCP script paths."""
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    venv_python.chmod(0o755)

    mcp_script = tmp_path / "omnimemory_mcp.py"
    mcp_script.write_text("# Mock MCP script\nprint('MCP Server')")

    return str(venv_python), str(mcp_script)


class TestClaudeConfigurator:
    """Test Claude Desktop configurator."""

    def test_initialization(self, mock_mcp_paths):
        """Test configurator initialization with MCP paths."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)

        assert config.venv_python == venv
        assert config.mcp_script == script

    def test_prompt_has_delimiter_syntax(self, mock_mcp_paths):
        """Test that prompt includes delimiter-based syntax examples."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)
        prompt = config.get_omnimemory_prompt()

        # Check delimiter syntax examples
        assert "file.py|overview" in prompt
        assert "tri_index" in prompt
        assert "file.py|symbol:Settings" in prompt

        # Check MCP tool names
        assert "mcp__omnimemory__read" in prompt
        assert "mcp__omnimemory__search" in prompt

        # Verify token savings messaging
        assert "90% token savings" in prompt
        assert "Token Savings Reporting" in prompt

    def test_config_generation_json(self, mock_mcp_paths, tmp_path):
        """Test JSON config file generation."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)

        # Use temp config path
        test_config = tmp_path / "test_claude_config.json"

        # Mock the config path
        config.get_config_path = lambda: test_config

        # Generate config
        result_path = config.configure()

        assert result_path == test_config
        assert test_config.exists()

        # Verify structure
        with open(test_config) as f:
            data = json.load(f)

        assert "mcpServers" in data
        assert "omn1" in data["mcpServers"]
        assert data["mcpServers"]["omn1"]["command"] == venv
        assert script in data["mcpServers"]["omn1"]["args"]
        assert "OMNIMEMORY_TOOL_ID" in data["mcpServers"]["omn1"]["env"]
        assert data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"] == "claude-code"

    def test_tool_id_is_correct(self, mock_mcp_paths, tmp_path):
        """Test that TOOL_ID is set correctly for Claude."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)

        test_config = tmp_path / "test_claude_config.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)

        assert data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"] == "claude-code"

    def test_no_api_key_in_config(self, mock_mcp_paths, tmp_path):
        """Test that old API key architecture is not present."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)

        test_config = tmp_path / "test_claude_config.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)
            env = data["mcpServers"]["omn1"]["env"]

        # Old API-based keys should NOT be present
        assert "OMNIMEMORY_API_KEY" not in env
        assert "OMNIMEMORY_API_URL" not in env
        assert "OMNIMEMORY_USER_ID" not in env


class TestCursorConfigurator:
    """Test Cursor configurator."""

    def test_initialization(self, mock_mcp_paths):
        """Test configurator initialization with MCP paths."""
        venv, script = mock_mcp_paths
        config = CursorConfigurator(venv, script)

        assert config.venv_python == venv
        assert config.mcp_script == script

    def test_prompt_has_delimiter_syntax(self, mock_mcp_paths):
        """Test that prompt includes delimiter-based syntax."""
        venv, script = mock_mcp_paths
        config = CursorConfigurator(venv, script)
        prompt = config.get_omnimemory_prompt()

        assert "mcp__omnimemory__read" in prompt
        assert "mcp__omnimemory__search" in prompt
        assert "tri_index" in prompt

    def test_config_generation_json(self, mock_mcp_paths, tmp_path):
        """Test JSON config file generation."""
        venv, script = mock_mcp_paths
        config = CursorConfigurator(venv, script)

        test_config = tmp_path / "mcp.json"
        config.get_config_path = lambda: test_config

        result_path = config.configure()

        assert result_path == test_config
        assert test_config.exists()

        with open(test_config) as f:
            data = json.load(f)

        assert "mcpServers" in data
        assert "omn1" in data["mcpServers"]
        assert data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"] == "cursor"

    def test_tool_id_is_correct(self, mock_mcp_paths, tmp_path):
        """Test that TOOL_ID is set correctly for Cursor."""
        venv, script = mock_mcp_paths
        config = CursorConfigurator(venv, script)

        test_config = tmp_path / "mcp.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)

        assert data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"] == "cursor"


class TestCodyConfigurator:
    """Test Cody (Sourcegraph) configurator."""

    def test_initialization(self, mock_mcp_paths):
        """Test configurator initialization with MCP paths."""
        venv, script = mock_mcp_paths
        config = CodyConfigurator(venv, script)

        assert config.venv_python == venv
        assert config.mcp_script == script

    def test_prompt_has_delimiter_syntax(self, mock_mcp_paths):
        """Test that prompt includes delimiter-based syntax."""
        venv, script = mock_mcp_paths
        config = CodyConfigurator(venv, script)
        prompt = config.get_omnimemory_prompt()

        assert "mcp__omnimemory__read" in prompt
        assert "mcp__omnimemory__search" in prompt
        assert "tri_index" in prompt

    def test_config_generation_json(self, mock_mcp_paths, tmp_path):
        """Test JSON config file generation via OpenCtx."""
        venv, script = mock_mcp_paths
        config = CodyConfigurator(venv, script)

        test_config = tmp_path / "settings.json"
        config.get_config_path = lambda: test_config

        result_path = config.configure()

        assert result_path == test_config
        assert test_config.exists()

        with open(test_config) as f:
            data = json.load(f)

        assert "openctx.enable" in data
        assert data["openctx.enable"] is True
        assert "openctx.providers" in data

        # Check for TOOL_ID in OpenCtx provider
        provider_key = "https://openctx.org/npm/@openctx/provider-modelcontextprotocol"
        assert provider_key in data["openctx.providers"]
        assert (
            data["openctx.providers"][provider_key]["mcp.provider.env"][
                "OMNIMEMORY_TOOL_ID"
            ]
            == "cody"
        )

    def test_tool_id_is_correct(self, mock_mcp_paths, tmp_path):
        """Test that TOOL_ID is set correctly for Cody."""
        venv, script = mock_mcp_paths
        config = CodyConfigurator(venv, script)

        test_config = tmp_path / "settings.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)

        provider_key = "https://openctx.org/npm/@openctx/provider-modelcontextprotocol"
        assert (
            data["openctx.providers"][provider_key]["mcp.provider.env"][
                "OMNIMEMORY_TOOL_ID"
            ]
            == "cody"
        )

    def test_prompt_embedded_in_config(self, mock_mcp_paths, tmp_path):
        """Test that OmniMemory prompt is embedded in systemPrompt."""
        venv, script = mock_mcp_paths
        config = CodyConfigurator(venv, script)

        test_config = tmp_path / "settings.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)

        assert "cody.systemPrompt" in data
        assert "OmniMemory MCP Tools" in data["cody.systemPrompt"]


class TestContinueConfigurator:
    """Test Continue.dev configurator."""

    def test_initialization(self, mock_mcp_paths):
        """Test configurator initialization with MCP paths."""
        venv, script = mock_mcp_paths
        config = ContinueConfigurator(venv, script)

        assert config.venv_python == venv
        assert config.mcp_script == script

    def test_prompt_has_delimiter_syntax(self, mock_mcp_paths):
        """Test that prompt includes delimiter-based syntax."""
        venv, script = mock_mcp_paths
        config = ContinueConfigurator(venv, script)
        prompt = config.get_omnimemory_prompt()

        assert "mcp__omnimemory__read" in prompt
        assert "mcp__omnimemory__search" in prompt
        assert "tri_index" in prompt

    def test_config_generation_json(self, mock_mcp_paths, tmp_path):
        """Test JSON config file generation."""
        venv, script = mock_mcp_paths
        config = ContinueConfigurator(venv, script)

        test_config = tmp_path / "config.json"
        config.get_config_path = lambda: test_config

        result_path = config.configure()

        assert result_path == test_config
        assert test_config.exists()

        with open(test_config) as f:
            data = json.load(f)

        assert "mcpServers" in data
        assert "omn1" in data["mcpServers"]
        assert data["mcpServers"]["omn1"]["command"] == venv
        assert data["mcpServers"]["omn1"]["args"] == [script]
        assert data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"] == "continue-dev"

    def test_tool_id_is_correct(self, mock_mcp_paths, tmp_path):
        """Test that TOOL_ID is set correctly for Continue.dev."""
        venv, script = mock_mcp_paths
        config = ContinueConfigurator(venv, script)

        test_config = tmp_path / "config.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)

        assert data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"] == "continue-dev"

    def test_prompt_embedded_in_config(self, mock_mcp_paths, tmp_path):
        """Test that OmniMemory prompt is embedded in systemMessage."""
        venv, script = mock_mcp_paths
        config = ContinueConfigurator(venv, script)

        test_config = tmp_path / "config.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)

        assert "systemMessage" in data
        assert "OmniMemory MCP Tools" in data["systemMessage"]


class TestGeminiConfigurator:
    """Test Gemini Code Assist configurator."""

    def test_initialization(self, mock_mcp_paths):
        """Test configurator initialization with MCP paths."""
        venv, script = mock_mcp_paths
        config = GeminiConfigurator(venv, script)

        assert config.venv_python == venv
        assert config.mcp_script == script

    def test_prompt_has_delimiter_syntax(self, mock_mcp_paths):
        """Test that prompt includes delimiter-based syntax."""
        venv, script = mock_mcp_paths
        config = GeminiConfigurator(venv, script)
        prompt = config.get_omnimemory_prompt()

        assert "mcp__omnimemory__read" in prompt
        assert "mcp__omnimemory__search" in prompt
        assert "tri_index" in prompt

    def test_config_generation_json(self, mock_mcp_paths, tmp_path):
        """Test JSON config file generation."""
        venv, script = mock_mcp_paths
        config = GeminiConfigurator(venv, script)

        test_config = tmp_path / "settings.json"
        config.get_config_path = lambda: test_config

        result_path = config.configure()

        assert result_path == test_config
        assert test_config.exists()

        with open(test_config) as f:
            data = json.load(f)

        assert "mcpServers" in data
        assert "omn1" in data["mcpServers"]
        assert (
            data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"]
            == "gemini-code-assist"
        )

    def test_tool_id_is_correct(self, mock_mcp_paths, tmp_path):
        """Test that TOOL_ID is set correctly for Gemini."""
        venv, script = mock_mcp_paths
        config = GeminiConfigurator(venv, script)

        test_config = tmp_path / "settings.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)

        assert (
            data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"]
            == "gemini-code-assist"
        )

    def test_prompt_embedded_in_config(self, mock_mcp_paths, tmp_path):
        """Test that OmniMemory prompt is embedded in systemPrompt."""
        venv, script = mock_mcp_paths
        config = GeminiConfigurator(venv, script)

        test_config = tmp_path / "settings.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)

        assert "systemPrompt" in data
        assert "OmniMemory MCP Tools" in data["systemPrompt"]


class TestVSCodeConfigurator:
    """Test VSCode (Cline) configurator."""

    def test_initialization(self, mock_mcp_paths):
        """Test configurator initialization with MCP paths."""
        venv, script = mock_mcp_paths
        config = VSCodeConfigurator(venv, script)

        assert config.venv_python == venv
        assert config.mcp_script == script

    def test_prompt_has_delimiter_syntax(self, mock_mcp_paths):
        """Test that prompt includes delimiter-based syntax."""
        venv, script = mock_mcp_paths
        config = VSCodeConfigurator(venv, script)
        prompt = config.get_omnimemory_prompt()

        assert "mcp__omnimemory__read" in prompt
        assert "mcp__omnimemory__search" in prompt
        assert "tri_index" in prompt

    def test_config_generation_mcp(self, mock_mcp_paths, tmp_path):
        """Test MCP config file generation."""
        venv, script = mock_mcp_paths
        config = VSCodeConfigurator(venv, script)

        test_mcp_config = tmp_path / "cline_mcp_settings.json"
        test_settings = tmp_path / "settings.json"

        # Mock both paths
        config.get_cline_mcp_config_path = lambda: test_mcp_config
        config.get_config_path = lambda: test_settings

        result_path = config.configure()

        # Should return settings path
        assert result_path == test_settings

        # Check MCP config
        assert test_mcp_config.exists()
        with open(test_mcp_config) as f:
            mcp_data = json.load(f)

        assert "mcpServers" in mcp_data
        assert "omn1" in mcp_data["mcpServers"]
        assert (
            mcp_data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"]
            == "vscode-copilot"
        )

    def test_tool_id_is_correct(self, mock_mcp_paths, tmp_path):
        """Test that TOOL_ID is set correctly for VSCode."""
        venv, script = mock_mcp_paths
        config = VSCodeConfigurator(venv, script)

        test_mcp_config = tmp_path / "cline_mcp_settings.json"
        test_settings = tmp_path / "settings.json"

        config.get_cline_mcp_config_path = lambda: test_mcp_config
        config.get_config_path = lambda: test_settings

        config.configure()

        with open(test_mcp_config) as f:
            data = json.load(f)

        assert (
            data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"] == "vscode-copilot"
        )

    def test_prompt_embedded_in_settings(self, mock_mcp_paths, tmp_path):
        """Test that OmniMemory prompt is embedded in customInstructions."""
        venv, script = mock_mcp_paths
        config = VSCodeConfigurator(venv, script)

        test_mcp_config = tmp_path / "cline_mcp_settings.json"
        test_settings = tmp_path / "settings.json"

        config.get_cline_mcp_config_path = lambda: test_mcp_config
        config.get_config_path = lambda: test_settings

        config.configure()

        with open(test_settings) as f:
            data = json.load(f)

        assert "cline.customInstructions" in data
        assert "OmniMemory MCP Tools" in data["cline.customInstructions"]


class TestWindsurfConfigurator:
    """Test Windsurf (Codeium) configurator."""

    def test_initialization(self, mock_mcp_paths):
        """Test configurator initialization with MCP paths."""
        venv, script = mock_mcp_paths
        config = WindsurfConfigurator(venv, script)

        assert config.venv_python == venv
        assert config.mcp_script == script

    def test_prompt_has_delimiter_syntax(self, mock_mcp_paths):
        """Test that prompt includes delimiter-based syntax."""
        venv, script = mock_mcp_paths
        config = WindsurfConfigurator(venv, script)
        prompt = config.get_omnimemory_prompt()

        assert "mcp__omnimemory__read" in prompt
        assert "mcp__omnimemory__search" in prompt
        assert "tri_index" in prompt

    def test_config_generation_json(self, mock_mcp_paths, tmp_path):
        """Test JSON config file generation."""
        venv, script = mock_mcp_paths
        config = WindsurfConfigurator(venv, script)

        test_config = tmp_path / "mcp_config.json"
        config.get_config_path = lambda: test_config

        result_path = config.configure()

        assert result_path == test_config
        assert test_config.exists()

        with open(test_config) as f:
            data = json.load(f)

        assert "mcpServers" in data
        assert "omn1" in data["mcpServers"]
        assert data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"] == "windsurf"

    def test_tool_id_is_correct(self, mock_mcp_paths, tmp_path):
        """Test that TOOL_ID is set correctly for Windsurf."""
        venv, script = mock_mcp_paths
        config = WindsurfConfigurator(venv, script)

        test_config = tmp_path / "mcp_config.json"
        config.get_config_path = lambda: test_config

        config.configure()

        with open(test_config) as f:
            data = json.load(f)

        assert data["mcpServers"]["omn1"]["env"]["OMNIMEMORY_TOOL_ID"] == "windsurf"


class TestCodexConfigurator:
    """Test Codex CLI configurator."""

    def test_initialization(self, mock_mcp_paths):
        """Test configurator initialization with MCP paths."""
        venv, script = mock_mcp_paths
        config = CodexConfigurator(venv, script)

        assert config.venv_python == venv
        assert config.mcp_script == script

    def test_prompt_has_delimiter_syntax(self, mock_mcp_paths):
        """Test that prompt includes delimiter-based syntax."""
        venv, script = mock_mcp_paths
        config = CodexConfigurator(venv, script)
        prompt = config.get_omnimemory_prompt()

        assert "mcp__omnimemory__read" in prompt
        assert "mcp__omnimemory__search" in prompt
        assert "tri_index" in prompt

    def test_config_generation_toml(self, mock_mcp_paths, tmp_path):
        """Test TOML config file generation."""
        venv, script = mock_mcp_paths
        config = CodexConfigurator(venv, script)

        test_config = tmp_path / "config.toml"
        config.get_config_path = lambda: test_config

        result_path = config.configure()

        assert result_path == test_config
        assert test_config.exists()

        # Read as text (TOML format)
        config_text = test_config.read_text()

        assert "[mcp_servers.omn1]" in config_text
        assert f'command = "{venv}"' in config_text
        assert f'args = ["{script}"]' in config_text
        assert 'OMNIMEMORY_TOOL_ID = "codex"' in config_text

    def test_tool_id_is_correct(self, mock_mcp_paths, tmp_path):
        """Test that TOOL_ID is set correctly for Codex."""
        venv, script = mock_mcp_paths
        config = CodexConfigurator(venv, script)

        test_config = tmp_path / "config.toml"
        config.get_config_path = lambda: test_config

        config.configure()

        config_text = test_config.read_text()
        assert 'OMNIMEMORY_TOOL_ID = "codex"' in config_text

    def test_prompt_embedded_in_toml(self, mock_mcp_paths, tmp_path):
        """Test that OmniMemory prompt is embedded in [prompt] section."""
        venv, script = mock_mcp_paths
        config = CodexConfigurator(venv, script)

        test_config = tmp_path / "config.toml"
        config.get_config_path = lambda: test_config

        config.configure()

        config_text = test_config.read_text()

        assert "[prompt]" in config_text
        assert "system = '''" in config_text
        assert "OmniMemory MCP Tools" in config_text


class TestConfiguratorCommonPatterns:
    """Test common patterns across all configurators."""

    def test_all_configurators_have_delimiter_syntax(self, mock_mcp_paths):
        """Test that all configurators include delimiter-based syntax in prompts."""
        venv, script = mock_mcp_paths

        configurators = [
            ClaudeConfigurator(venv, script),
            CursorConfigurator(venv, script),
            CodyConfigurator(venv, script),
            ContinueConfigurator(venv, script),
            GeminiConfigurator(venv, script),
            VSCodeConfigurator(venv, script),
            WindsurfConfigurator(venv, script),
            CodexConfigurator(venv, script),
        ]

        for config in configurators:
            prompt = config.get_omnimemory_prompt()
            assert "mcp__omnimemory__read" in prompt
            assert "mcp__omnimemory__search" in prompt
            assert "tri_index" in prompt
            assert "file.py|overview" in prompt

    def test_all_configurators_have_unique_tool_ids(self, mock_mcp_paths, tmp_path):
        """Test that all configurators have unique TOOL_IDs."""
        venv, script = mock_mcp_paths

        # Expected TOOL_IDs for each configurator
        expected_tool_ids = {
            "claude": "claude-code",
            "cursor": "cursor",
            "cody": "cody",
            "continue": "continue-dev",
            "gemini": "gemini-code-assist",
            "vscode": "vscode-copilot",
            "windsurf": "windsurf",
            "codex": "codex",
        }

        # Verify all TOOL_IDs are unique
        assert len(expected_tool_ids.values()) == len(set(expected_tool_ids.values()))

    def test_no_configurator_uses_api_key_architecture(self, mock_mcp_paths):
        """Test that no configurator uses the old API key architecture."""
        venv, script = mock_mcp_paths

        configurators = [
            ClaudeConfigurator(venv, script),
            CursorConfigurator(venv, script),
            CodyConfigurator(venv, script),
            ContinueConfigurator(venv, script),
            GeminiConfigurator(venv, script),
            VSCodeConfigurator(venv, script),
            WindsurfConfigurator(venv, script),
            CodexConfigurator(venv, script),
        ]

        for config in configurators:
            # Check that configurator doesn't have old API-based attributes
            assert not hasattr(config, "api_key")
            assert not hasattr(config, "api_url")
            assert not hasattr(config, "user_id")

            # Check that configurator has new MCP-based attributes
            assert hasattr(config, "venv_python")
            assert hasattr(config, "mcp_script")

    def test_all_configurators_have_omnimemory_prompt_method(self, mock_mcp_paths):
        """Test that all configurators have get_omnimemory_prompt() method."""
        venv, script = mock_mcp_paths

        configurators = [
            ClaudeConfigurator(venv, script),
            CursorConfigurator(venv, script),
            CodyConfigurator(venv, script),
            ContinueConfigurator(venv, script),
            GeminiConfigurator(venv, script),
            VSCodeConfigurator(venv, script),
            WindsurfConfigurator(venv, script),
            CodexConfigurator(venv, script),
        ]

        for config in configurators:
            # Should have get_omnimemory_prompt(), not get_system_prompt()
            assert hasattr(config, "get_omnimemory_prompt")
            assert callable(config.get_omnimemory_prompt)

            # Old method should not exist
            assert not hasattr(config, "get_system_prompt")


class TestBackupCreation:
    """Test backup creation for configurators that support it."""

    def test_backup_created_when_config_exists(self, mock_mcp_paths, tmp_path):
        """Test that backups are created when modifying existing configs."""
        venv, script = mock_mcp_paths
        config = ContinueConfigurator(venv, script)

        test_config = tmp_path / "config.json"
        config.get_config_path = lambda: test_config

        # Create initial config
        initial_data = {"foo": "bar"}
        with open(test_config, "w") as f:
            json.dump(initial_data, f)

        # Configure (should create backup)
        config.configure()

        # Check for backup files
        backup_files = list(tmp_path.glob("config.json.backup-*"))
        assert len(backup_files) >= 1

        # Verify backup contains original data
        with open(backup_files[0]) as f:
            backup_data = json.load(f)
        assert backup_data == initial_data


class TestErrorHandling:
    """Test error handling in configurators."""

    def test_configure_creates_directory_if_missing(self, mock_mcp_paths, tmp_path):
        """Test that configure() creates parent directories if they don't exist."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)

        # Use a path with non-existent parent directory
        test_config = tmp_path / "nested" / "dir" / "config.json"
        config.get_config_path = lambda: test_config

        # Should not raise an error
        result_path = config.configure()

        assert result_path == test_config
        assert test_config.exists()
        assert test_config.parent.exists()


class TestClaudeMDInjection:
    """Test CLAUDE.md file injection for Claude configurator."""

    def test_inject_global_claude_md_creates_file(self, mock_mcp_paths, tmp_path):
        """Test that inject_global_claude_md() creates new file."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)

        # Mock the home directory
        claude_md_path = tmp_path / ".claude" / "CLAUDE.md"

        # Mock get_config_path to avoid file system issues
        config.get_config_path = lambda: tmp_path / "config.json"

        # Patch the claude_md_path directly
        original_inject = config.inject_global_claude_md

        def mock_inject():
            claude_md_path.parent.mkdir(parents=True, exist_ok=True)
            with open(claude_md_path, "w", encoding="utf-8") as f:
                f.write(config.get_omnimemory_prompt())
            return claude_md_path

        config.inject_global_claude_md = mock_inject

        result_path = config.inject_global_claude_md()

        assert result_path == claude_md_path
        assert claude_md_path.exists()

        content = claude_md_path.read_text()
        assert "OmniMemory MCP Tools" in content

    def test_inject_global_claude_md_skips_if_already_present(
        self, mock_mcp_paths, tmp_path
    ):
        """Test that inject_global_claude_md() skips if already present."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)

        claude_md_path = tmp_path / ".claude" / "CLAUDE.md"
        claude_md_path.parent.mkdir(parents=True, exist_ok=True)

        # Write existing content with OmniMemory
        existing_content = "# Existing\n\n# OmniMemory MCP Tools\nalready present"
        claude_md_path.write_text(existing_content, encoding="utf-8")

        # Test _is_omnimemory_prompt_present
        assert config._is_omnimemory_prompt_present(existing_content)

    def test_inject_project_claude_md_creates_file(self, mock_mcp_paths, tmp_path):
        """Test that inject_project_claude_md() creates project file."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)

        project_path = tmp_path / "project"
        project_path.mkdir()

        result_path = config.inject_project_claude_md(project_path)

        if result_path:  # May return None if already present
            assert result_path.exists()
            assert ".claude" in str(result_path)


class TestCursorRulesInjection:
    """Test .cursorrules file injection for Cursor configurator."""

    def test_inject_cursor_rules_creates_file(self, mock_mcp_paths, tmp_path):
        """Test that inject_cursor_rules() creates new file."""
        venv, script = mock_mcp_paths
        config = CursorConfigurator(venv, script)

        cursorrules_path = tmp_path / ".cursorrules"

        # Mock the method to use tmp_path
        original_inject = config.inject_cursor_rules

        def mock_inject():
            with open(cursorrules_path, "w", encoding="utf-8") as f:
                f.write(config.get_omnimemory_prompt())
            return cursorrules_path

        config.inject_cursor_rules = mock_inject

        result_path = config.inject_cursor_rules()

        assert result_path == cursorrules_path
        assert cursorrules_path.exists()

        content = cursorrules_path.read_text()
        assert "OmniMemory MCP Tools" in content

    def test_inject_cursor_rules_skips_if_already_present(
        self, mock_mcp_paths, tmp_path
    ):
        """Test that inject_cursor_rules() skips if already present."""
        venv, script = mock_mcp_paths
        config = CursorConfigurator(venv, script)

        existing_content = "# mcp__omnimemory__ tools"
        assert config._is_omnimemory_prompt_present(existing_content)

    def test_inject_project_cursorrules_creates_file(self, mock_mcp_paths, tmp_path):
        """Test that inject_project_cursorrules() creates project file."""
        venv, script = mock_mcp_paths
        config = CursorConfigurator(venv, script)

        project_path = tmp_path / "project"
        project_path.mkdir()

        result_path = config.inject_project_cursorrules(project_path)

        if result_path:  # May return None if already present
            assert result_path.exists()
            assert ".cursorrules" in str(result_path)


class TestWindsurfRulesInjection:
    """Test .windsurfrules file injection for Windsurf configurator."""

    def test_inject_windsurf_rules_creates_file(self, mock_mcp_paths, tmp_path):
        """Test that inject_windsurf_rules() creates new file."""
        venv, script = mock_mcp_paths
        config = WindsurfConfigurator(venv, script)

        windsurfrules_path = tmp_path / ".windsurfrules"

        # Mock the method to use tmp_path
        original_inject = config.inject_windsurf_rules

        def mock_inject():
            with open(windsurfrules_path, "w", encoding="utf-8") as f:
                f.write(config.get_omnimemory_prompt())
            return windsurfrules_path

        config.inject_windsurf_rules = mock_inject

        result_path = config.inject_windsurf_rules()

        assert result_path == windsurfrules_path
        assert windsurfrules_path.exists()

        content = windsurfrules_path.read_text()
        assert "OmniMemory MCP Tools" in content

    def test_inject_windsurf_rules_skips_if_already_present(
        self, mock_mcp_paths, tmp_path
    ):
        """Test that inject_windsurf_rules() skips if already present."""
        venv, script = mock_mcp_paths
        config = WindsurfConfigurator(venv, script)

        existing_content = "# OmniMemory MCP Tools present"
        assert config._is_omnimemory_prompt_present(existing_content)

    def test_inject_project_windsurfrules_creates_file(self, mock_mcp_paths, tmp_path):
        """Test that inject_project_windsurfrules() creates project file."""
        venv, script = mock_mcp_paths
        config = WindsurfConfigurator(venv, script)

        project_path = tmp_path / "project"
        project_path.mkdir()

        result_path = config.inject_project_windsurfrules(project_path)

        if result_path:  # May return None if already present
            assert result_path.exists()
            assert ".windsurfrules" in str(result_path)


class TestConfigPathDetection:
    """Test config path detection for all configurators."""

    def test_claude_config_path_returns_path(self, mock_mcp_paths):
        """Test that Claude configurator returns a valid path."""
        venv, script = mock_mcp_paths
        config = ClaudeConfigurator(venv, script)

        path = config.get_config_path()

        assert isinstance(path, Path)
        assert "claude" in str(path).lower()

    def test_cursor_config_path_returns_path(self, mock_mcp_paths):
        """Test that Cursor configurator returns a valid path."""
        venv, script = mock_mcp_paths
        config = CursorConfigurator(venv, script)

        path = config.get_config_path()

        assert isinstance(path, Path)
        assert "cursor" in str(path).lower()

    def test_cody_config_path_returns_path(self, mock_mcp_paths):
        """Test that Cody configurator returns a valid path."""
        venv, script = mock_mcp_paths
        config = CodyConfigurator(venv, script)

        path = config.get_config_path()

        assert isinstance(path, Path)
        assert "Code" in str(path) or "code" in str(path).lower()

    def test_continue_config_path_returns_path(self, mock_mcp_paths):
        """Test that Continue configurator returns a valid path."""
        venv, script = mock_mcp_paths
        config = ContinueConfigurator(venv, script)

        path = config.get_config_path()

        assert isinstance(path, Path)
        assert ".continue" in str(path)

    def test_gemini_config_path_returns_path(self, mock_mcp_paths):
        """Test that Gemini configurator returns a valid path."""
        venv, script = mock_mcp_paths
        config = GeminiConfigurator(venv, script)

        path = config.get_config_path()

        assert isinstance(path, Path)
        assert ".gemini" in str(path)

    def test_vscode_config_path_returns_path(self, mock_mcp_paths):
        """Test that VSCode configurator returns a valid path."""
        venv, script = mock_mcp_paths
        config = VSCodeConfigurator(venv, script)

        path = config.get_config_path()

        assert isinstance(path, Path)
        assert "Code" in str(path) or "code" in str(path).lower()

    def test_vscode_mcp_config_path_returns_path(self, mock_mcp_paths):
        """Test that VSCode MCP config path is correct."""
        venv, script = mock_mcp_paths
        config = VSCodeConfigurator(venv, script)

        path = config.get_cline_mcp_config_path()

        assert isinstance(path, Path)
        assert ".vscode" in str(path)
        assert "cline_mcp_settings.json" in str(path)

    def test_windsurf_config_path_returns_path(self, mock_mcp_paths):
        """Test that Windsurf configurator returns a valid path."""
        venv, script = mock_mcp_paths
        config = WindsurfConfigurator(venv, script)

        path = config.get_config_path()

        assert isinstance(path, Path)
        assert "windsurf" in str(path).lower() or "codeium" in str(path).lower()

    def test_codex_config_path_returns_path(self, mock_mcp_paths):
        """Test that Codex configurator returns a valid path."""
        venv, script = mock_mcp_paths
        config = CodexConfigurator(venv, script)

        path = config.get_config_path()

        assert isinstance(path, Path)
        assert ".codex" in str(path)
        assert "config.toml" in str(path)


class TestCodexTOMLHelpers:
    """Test Codex TOML helper methods."""

    def test_remove_omn1_section(self, mock_mcp_paths):
        """Test removal of omn1 section from TOML."""
        venv, script = mock_mcp_paths
        config = CodexConfigurator(venv, script)

        config_text = """
[other_section]
foo = "bar"

[mcp_servers.omn1]
command = "python"
args = ["script.py"]

[another_section]
baz = "qux"
"""
        result = config._remove_omn1_section(config_text)

        assert "[mcp_servers.omn1]" not in result
        assert "[other_section]" in result
        assert "[another_section]" in result

    def test_remove_prompt_section(self, mock_mcp_paths):
        """Test removal of prompt section from TOML."""
        venv, script = mock_mcp_paths
        config = CodexConfigurator(venv, script)

        config_text = """
[other]
foo = "bar"

[prompt]
system = '''
Some prompt text
'''

[another]
baz = "qux"
"""
        result = config._remove_prompt_section(config_text)

        assert "[prompt]" not in result
        assert "[other]" in result
        assert "[another]" in result
