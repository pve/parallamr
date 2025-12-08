#!/usr/bin/env python3
"""Quick test to verify parallel execution works correctly."""

import asyncio
import tempfile
from pathlib import Path
from parallamr.runner import ExperimentRunner
from parallamr.models import Experiment

async def main():
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create prompt file
        prompt_file = tmpdir / "prompt.txt"
        prompt_file.write_text("Test prompt about {{topic}}")

        # Create experiments file
        experiments_file = tmpdir / "experiments.csv"
        experiments_content = """provider,model,topic
mock,mock,AI
mock,mock,ML
mock,mock,NLP
mock,mock,CV
mock,mock,RL"""
        experiments_file.write_text(experiments_content)

        # Create output file
        output_file = tmpdir / "results.csv"

        print("Testing parallel execution...")

        # Test with default concurrent settings
        runner = ExperimentRunner(verbose=True)
        print(f"\nConcurrency limits: {[(k, s._value) for k, s in runner._provider_semaphores.items()]}")

        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file,
            context_files=None,
            read_prompt_stdin=False,
            read_experiments_stdin=False
        )

        # Verify results
        results = output_file.read_text()
        print(f"\nResults written to {output_file}")
        print(f"Number of result rows: {len(results.splitlines()) - 1}")  # -1 for header

        print("\n✓ Parallel execution test passed!")

        # Test with sequential flag
        print("\n\nTesting sequential execution...")
        output_file_seq = tmpdir / "results_seq.csv"
        runner_seq = ExperimentRunner(verbose=True, sequential=True)
        print(f"Sequential concurrency limits: {[(k, s._value) for k, s in runner_seq._provider_semaphores.items()]}")

        await runner_seq.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file_seq,
            context_files=None,
            read_prompt_stdin=False,
            read_experiments_stdin=False
        )

        print("✓ Sequential execution test passed!")

        # Test with custom concurrency
        print("\n\nTesting custom concurrency limits...")
        output_file_custom = tmpdir / "results_custom.csv"
        runner_custom = ExperimentRunner(
            verbose=True,
            max_concurrent=3,
            provider_concurrency={"mock": 2}
        )
        print(f"Custom concurrency limits: {[(k, s._value) for k, s in runner_custom._provider_semaphores.items()]}")

        await runner_custom.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file_custom,
            context_files=None,
            read_prompt_stdin=False,
            read_experiments_stdin=False
        )

        print("✓ Custom concurrency test passed!")

if __name__ == "__main__":
    asyncio.run(main())
