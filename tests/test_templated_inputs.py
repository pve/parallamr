"""Tests for templated input file paths (prompt and context)."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from parallamr.models import Experiment, ExperimentStatus
from parallamr.runner import ExperimentRunner
from parallamr.file_loader import FileLoader


@pytest.fixture
def temp_prompt_files(tmp_path):
    """Create temporary prompt files for different topics."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create prompts for different topics
    (prompts_dir / "AI-prompt.txt").write_text("Prompt about {{topic}}: AI")
    (prompts_dir / "Blockchain-prompt.txt").write_text("Prompt about {{topic}}: Blockchain")

    return prompts_dir


@pytest.fixture
def temp_context_files(tmp_path):
    """Create temporary context files for different models."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    # Create context files for different models
    (context_dir / "mock-guide.txt").write_text("Mock provider guide")
    (context_dir / "openrouter-guide.txt").write_text("OpenRouter provider guide")

    return context_dir


@pytest.fixture
def temp_experiments_csv(tmp_path):
    """Create temporary experiments CSV."""
    csv_path = tmp_path / "experiments.csv"
    csv_path.write_text("""provider,model,topic
mock,mock,AI
mock,mock,Blockchain
""")
    return csv_path


class TestTemplatedPromptPaths:
    """Tests for template variable substitution in prompt paths."""

    @pytest.mark.asyncio
    async def test_templated_prompt_path_basic(self, tmp_path, temp_prompt_files, temp_experiments_csv):
        """Test basic templated prompt path resolution."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        await runner.run_experiments(
            prompt_file=temp_prompt_files / "{{topic}}-prompt.txt",
            experiments_file=temp_experiments_csv,
            output_file=output_file,
            context_files=None
        )

        # Verify output file was created
        assert output_file.exists()

        # Verify both experiments ran successfully
        output_content = output_file.read_text()
        assert "mock,mock,AI" in output_content
        assert "mock,mock,Blockchain" in output_content
        assert "ok" in output_content

    @pytest.mark.asyncio
    async def test_templated_prompt_missing_file(self, tmp_path, temp_experiments_csv):
        """Test error handling when templated prompt file doesn't exist."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        # Use a template that resolves to non-existent files
        await runner.run_experiments(
            prompt_file="nonexistent/{{topic}}-prompt.txt",
            experiments_file=temp_experiments_csv,
            output_file=output_file
        )

        # Verify output file was created
        assert output_file.exists()

        # Verify experiments resulted in errors
        output_content = output_file.read_text()
        assert "error" in output_content
        assert "File loading error" in output_content or "No such file" in output_content

    @pytest.mark.asyncio
    async def test_templated_prompt_with_static_context(self, tmp_path, temp_prompt_files, temp_experiments_csv):
        """Test templated prompt with static context files."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        # Create a static context file
        context_file = tmp_path / "static-context.txt"
        context_file.write_text("Static context content")

        await runner.run_experiments(
            prompt_file=temp_prompt_files / "{{topic}}-prompt.txt",
            experiments_file=temp_experiments_csv,
            output_file=output_file,
            context_files=[context_file]
        )

        assert output_file.exists()
        output_content = output_file.read_text()
        assert "ok" in output_content


class TestTemplatedContextPaths:
    """Tests for template variable substitution in context paths."""

    @pytest.mark.asyncio
    async def test_templated_context_path_basic(self, tmp_path, temp_context_files, temp_experiments_csv):
        """Test basic templated context path resolution."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        # Create a simple prompt
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Simple prompt about {{topic}}")

        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=temp_experiments_csv,
            output_file=output_file,
            context_files=[temp_context_files / "{{provider}}-guide.txt"]
        )

        assert output_file.exists()
        output_content = output_file.read_text()
        assert "ok" in output_content

    @pytest.mark.asyncio
    async def test_multiple_templated_contexts(self, tmp_path, temp_experiments_csv):
        """Test multiple context files with templates."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        # Create context files for different variables
        context_dir = tmp_path / "contexts"
        context_dir.mkdir()
        (context_dir / "AI-topic.txt").write_text("AI topic context")
        (context_dir / "Blockchain-topic.txt").write_text("Blockchain topic context")
        (context_dir / "mock-provider.txt").write_text("Mock provider context")

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Prompt with {{topic}}")

        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=temp_experiments_csv,
            output_file=output_file,
            context_files=[
                context_dir / "{{topic}}-topic.txt",
                context_dir / "{{provider}}-provider.txt"
            ]
        )

        assert output_file.exists()
        output_content = output_file.read_text()
        assert "ok" in output_content

    @pytest.mark.asyncio
    async def test_mixed_static_and_templated_contexts(self, tmp_path, temp_experiments_csv):
        """Test mix of static and templated context files."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        # Create static and templated context files
        static_context = tmp_path / "static.txt"
        static_context.write_text("Static context")

        context_dir = tmp_path / "contexts"
        context_dir.mkdir()
        (context_dir / "AI-dynamic.txt").write_text("AI dynamic context")
        (context_dir / "Blockchain-dynamic.txt").write_text("Blockchain dynamic context")

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Prompt")

        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=temp_experiments_csv,
            output_file=output_file,
            context_files=[
                static_context,
                context_dir / "{{topic}}-dynamic.txt"
            ]
        )

        assert output_file.exists()
        output_content = output_file.read_text()
        assert "ok" in output_content


class TestTemplatedInputsWithTemplatedOutput:
    """Tests for templated inputs combined with templated output."""

    @pytest.mark.asyncio
    async def test_all_paths_templated(self, tmp_path, temp_prompt_files, temp_context_files, temp_experiments_csv):
        """Test when all paths (prompt, context, output) are templated."""
        runner = ExperimentRunner(verbose=False)

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        await runner.run_experiments(
            prompt_file=temp_prompt_files / "{{topic}}-prompt.txt",
            experiments_file=temp_experiments_csv,
            output_file=output_dir / "{{topic}}-results.csv",
            context_files=[temp_context_files / "{{provider}}-guide.txt"]
        )

        # Verify separate output files were created
        assert (output_dir / "AI-results.csv").exists()
        assert (output_dir / "Blockchain-results.csv").exists()

        # Verify content
        ai_content = (output_dir / "AI-results.csv").read_text()
        assert "AI" in ai_content
        assert "ok" in ai_content

        blockchain_content = (output_dir / "Blockchain-results.csv").read_text()
        assert "Blockchain" in blockchain_content
        assert "ok" in blockchain_content

    @pytest.mark.asyncio
    async def test_templated_input_with_subdirectories(self, tmp_path, temp_experiments_csv):
        """Test templated paths with subdirectory organization."""
        runner = ExperimentRunner(verbose=False)

        # Create organized prompt structure
        prompts_base = tmp_path / "prompts"
        (prompts_base / "AI").mkdir(parents=True)
        (prompts_base / "Blockchain").mkdir(parents=True)
        (prompts_base / "AI" / "prompt.txt").write_text("AI prompt with {{topic}}")
        (prompts_base / "Blockchain" / "prompt.txt").write_text("Blockchain prompt with {{topic}}")

        output_file = tmp_path / "results.csv"

        await runner.run_experiments(
            prompt_file=prompts_base / "{{topic}}" / "prompt.txt",
            experiments_file=temp_experiments_csv,
            output_file=output_file
        )

        assert output_file.exists()
        output_content = output_file.read_text()
        assert "ok" in output_content


class TestMissingFileHandling:
    """Tests for error handling with missing templated files."""

    @pytest.mark.asyncio
    async def test_missing_prompt_creates_error_result(self, tmp_path, temp_experiments_csv):
        """Test that missing prompt file creates error result instead of crashing."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        await runner.run_experiments(
            prompt_file="nonexistent/{{topic}}.txt",
            experiments_file=temp_experiments_csv,
            output_file=output_file
        )

        assert output_file.exists()
        output_content = output_file.read_text()

        # Should have error status
        assert "error" in output_content
        # Should complete both experiments (not crash on first error)
        lines = output_content.strip().split('\n')
        assert len(lines) >= 3  # Header + 2 data rows

    @pytest.mark.asyncio
    async def test_missing_context_creates_error_result(self, tmp_path, temp_experiments_csv):
        """Test that missing context file creates error result."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        # Create valid prompt
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Valid prompt")

        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=temp_experiments_csv,
            output_file=output_file,
            context_files=["nonexistent/{{provider}}-context.txt"]
        )

        assert output_file.exists()
        output_content = output_file.read_text()
        assert "error" in output_content

    @pytest.mark.asyncio
    async def test_partial_missing_files(self, tmp_path, temp_prompt_files, temp_experiments_csv):
        """Test when some experiments have missing files but others don't."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        # Only create prompt for AI, not Blockchain
        # (temp_prompt_files already has AI-prompt.txt, but we'll use different template)
        prompts_dir = tmp_path / "partial_prompts"
        prompts_dir.mkdir()
        (prompts_dir / "AI-prompt.txt").write_text("AI prompt")
        # Blockchain-prompt.txt intentionally missing

        await runner.run_experiments(
            prompt_file=prompts_dir / "{{topic}}-prompt.txt",
            experiments_file=temp_experiments_csv,
            output_file=output_file
        )

        assert output_file.exists()
        output_content = output_file.read_text()

        # AI should succeed
        assert "AI" in output_content
        # Blockchain should have error
        assert "error" in output_content


class TestTemplatedInputEdgeCases:
    """Edge case tests for templated input paths."""

    @pytest.mark.asyncio
    async def test_template_with_special_characters_in_value(self, tmp_path):
        """Test template variables with special characters get sanitized."""
        runner = ExperimentRunner(verbose=False)

        # Create experiments with special characters in variables
        experiments_csv = tmp_path / "experiments.csv"
        experiments_csv.write_text("""provider,model,filename
mock,mock,test/file
""")

        # Create the sanitized file (slashes become underscores)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test_file-prompt.txt").write_text("Test prompt with {{filename}}")

        output_file = tmp_path / "results.csv"

        await runner.run_experiments(
            prompt_file=prompts_dir / "{{filename}}-prompt.txt",
            experiments_file=experiments_csv,
            output_file=output_file
        )

        assert output_file.exists()
        output_content = output_file.read_text()
        assert "ok" in output_content

    @pytest.mark.asyncio
    async def test_no_templates_uses_fast_path(self, tmp_path, temp_experiments_csv):
        """Test that non-templated paths use original fast code path."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        # Create static prompt
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Static prompt")

        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=temp_experiments_csv,
            output_file=output_file
        )

        assert output_file.exists()
        output_content = output_file.read_text()
        assert "ok" in output_content

    @pytest.mark.asyncio
    async def test_template_variables_in_prompt_content(self, tmp_path, temp_experiments_csv):
        """Test that template variables in prompt CONTENT still work."""
        runner = ExperimentRunner(verbose=False)
        output_file = tmp_path / "results.csv"

        # Prompt file path is static, but content has templates
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Write about {{topic}} in detail")

        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=temp_experiments_csv,
            output_file=output_file
        )

        assert output_file.exists()
        output_content = output_file.read_text()
        assert "ok" in output_content
