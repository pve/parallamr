"""File and stdin loading abstractions for testability."""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .models import Experiment
from .utils import load_context_files, load_experiments_from_csv, load_file_content


class FileLoader:
    """Abstraction for file/stdin loading operations."""

    def load_prompt(
        self,
        file_path: Optional[Path],
        use_stdin: bool
    ) -> str:
        """
        Load prompt from file or stdin.

        Args:
            file_path: Path to prompt file (ignored if use_stdin is True)
            use_stdin: Whether to read from stdin

        Returns:
            Prompt content as string
        """
        if use_stdin:
            return self._read_stdin()
        return load_file_content(file_path)

    def load_experiments(
        self,
        file_path: Optional[Path],
        use_stdin: bool
    ) -> List[Experiment]:
        """
        Load experiments from file or stdin.

        Args:
            file_path: Path to experiments CSV file (ignored if use_stdin is True)
            use_stdin: Whether to read from stdin

        Returns:
            List of Experiment objects
        """
        if use_stdin:
            content = self._read_stdin()
            return load_experiments_from_csv(csv_content=content)
        return load_experiments_from_csv(csv_path=file_path)

    def load_context(
        self,
        file_paths: Optional[List[Path]]
    ) -> List[Tuple[str, str]]:
        """
        Load context files.

        Args:
            file_paths: List of paths to context files

        Returns:
            List of (filename, content) tuples
        """
        if not file_paths:
            return []
        return load_context_files(file_paths)

    def _read_stdin(self) -> str:
        """
        Read from stdin (mockable).

        Returns:
            Content from stdin
        """
        return sys.stdin.read()
