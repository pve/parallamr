"""Tests for CSV writer parallel processing optimizations."""

import asyncio
import csv
import threading
import time
from pathlib import Path

import pytest

from parallamr.csv_writer import IncrementalCSVWriter
from parallamr.models import ExperimentResult, ExperimentStatus


def create_test_result(row_number: int, variables: dict = None, **kwargs) -> ExperimentResult:
    """Helper to create test ExperimentResult objects."""
    default_vars = {"topic": "test"}
    if variables:
        default_vars.update(variables)

    defaults = {
        "provider": "mock",
        "model": "test-model",
        "variables": default_vars,
        "row_number": row_number,
        "status": ExperimentStatus.OK,
        "input_tokens": 50,
        "context_window": 8192,
        "output_tokens": 10,
        "output": f"Output {row_number}",
        "error_message": None
    }
    defaults.update(kwargs)

    return ExperimentResult(**defaults)


class TestFileHandlePersistence:
    """Test file handle persistence across writes."""

    def test_file_handle_remains_open_across_writes(self, tmp_path):
        """Verify file handle is kept open across multiple write_result() calls."""
        output_file = tmp_path / "test_output.csv"

        writer = IncrementalCSVWriter(output_file)

        # Write 10 results
        for i in range(10):
            result = create_test_result(row_number=i)
            writer.write_result(result)

        # Verify file handle exists and is open
        assert hasattr(writer, '_file_handle')
        assert writer._file_handle is not None
        assert not writer._file_handle.closed

        writer.close()

    def test_file_handle_attribute_persists(self, tmp_path):
        """Verify _file_handle attribute is maintained across writes."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        # After first write, file handle should exist
        writer.write_result(create_test_result(row_number=1))
        assert hasattr(writer, '_file_handle')
        assert writer._file_handle is not None
        first_handle_id = id(writer._file_handle)

        # After second write, same handle should be used
        writer.write_result(create_test_result(row_number=2))
        assert id(writer._file_handle) == first_handle_id

        writer.close()

    def test_stdout_mode_no_file_handle(self, capsys):
        """Verify stdout mode doesn't maintain file handles."""
        writer = IncrementalCSVWriter(None)  # stdout mode

        writer.write_result(create_test_result(row_number=1))

        # Should not have file handle in stdout mode
        assert not hasattr(writer, '_file_handle') or writer._file_handle is None

        captured = capsys.readouterr()
        assert "provider" in captured.out  # Headers written


class TestThreadSafety:
    """Test thread-safe concurrent writes."""

    def test_concurrent_writes_with_threading(self, tmp_path):
        """Verify thread-safe concurrent writes using threading module."""
        output_file = tmp_path / "concurrent_output.csv"
        writer = IncrementalCSVWriter(output_file)

        num_threads = 10
        writes_per_thread = 20

        def write_results(thread_id):
            for i in range(writes_per_thread):
                result = create_test_result(
                    row_number=thread_id * writes_per_thread + i,
                    variables={"thread": str(thread_id), "write": str(i)}
                )
                writer.write_result(result)

        threads = [threading.Thread(target=write_results, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        writer.close()

        # Verify all rows written without corruption
        with open(output_file, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == num_threads * writes_per_thread

        # Verify no partial/corrupted rows
        for row in rows:
            assert row['provider'] == 'mock'
            assert row['thread'].isdigit()
            assert row['write'].isdigit()

    def test_lock_prevents_interleaved_writes(self, tmp_path):
        """Verify locking prevents corrupted/interleaved CSV rows."""
        output_file = tmp_path / "locked_output.csv"
        writer = IncrementalCSVWriter(output_file)

        # Verify writer has a lock
        assert hasattr(writer, '_lock')

        barrier = threading.Barrier(5)

        def synchronized_write(thread_id):
            barrier.wait()  # All threads start simultaneously
            for i in range(10):
                result = create_test_result(
                    row_number=thread_id * 10 + i,
                    output=f"Thread {thread_id} write {i}" * 100  # Long output
                )
                writer.write_result(result)

        threads = [threading.Thread(target=synchronized_write, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        writer.close()

        # Read raw file to check for interleaving
        with open(output_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        # Should have header + 50 data rows = 51 lines
        assert len(lines) == 51

        # Each line should be complete (no partial rows)
        for i, line in enumerate(lines):
            if i == 0:  # header
                assert 'provider' in line
            else:
                # Data rows should have proper CSV structure
                row = next(csv.reader([line]))
                assert len(row) > 0  # Should parse as valid CSV

    def test_lock_is_reentrant_safe(self, tmp_path):
        """Verify lock supports reentrant calls (threading.RLock)."""
        output_file = tmp_path / "reentrant_output.csv"
        writer = IncrementalCSVWriter(output_file)

        # Should use RLock, not Lock
        # Check by type name since RLock is not directly comparable
        assert type(writer._lock).__name__ == 'RLock'

        writer.close()


class TestContextManager:
    """Test context manager support."""

    def test_context_manager_basic_usage(self, tmp_path):
        """Verify context manager __enter__ and __exit__ work correctly."""
        output_file = tmp_path / "context_output.csv"

        with IncrementalCSVWriter(output_file) as writer:
            writer.write_result(create_test_result(row_number=1))
            writer.write_result(create_test_result(row_number=2))

            # File handle should be open inside context
            assert hasattr(writer, '_file_handle')
            assert writer._file_handle is not None
            assert not writer._file_handle.closed

        # File handle should be closed after context exit
        assert writer._file_handle.closed

    def test_context_manager_cleanup_on_exception(self, tmp_path):
        """Verify file handle is closed even if exception occurs."""
        output_file = tmp_path / "exception_output.csv"

        writer = None
        try:
            with IncrementalCSVWriter(output_file) as writer:
                writer.write_result(create_test_result(row_number=1))
                raise ValueError("Test exception")
        except ValueError:
            pass

        # File should still be closed
        assert writer._file_handle.closed

        # File should still be readable and contain the row
        with open(output_file, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_context_manager_returns_self(self, tmp_path):
        """Verify context manager __enter__ returns self."""
        output_file = tmp_path / "self_output.csv"

        writer = IncrementalCSVWriter(output_file)
        returned = writer.__enter__()

        assert returned is writer

        writer.__exit__(None, None, None)

    def test_context_manager_stdout_mode(self, capsys):
        """Verify context manager works with stdout mode."""
        with IncrementalCSVWriter(None) as writer:
            writer.write_result(create_test_result(row_number=1))

        # Should not raise any errors
        captured = capsys.readouterr()
        assert "provider" in captured.out


class TestExplicitClose:
    """Test explicit close() method."""

    def test_explicit_close_closes_file_handle(self, tmp_path):
        """Verify close() method closes file handle."""
        output_file = tmp_path / "close_output.csv"
        writer = IncrementalCSVWriter(output_file)

        writer.write_result(create_test_result(row_number=1))

        assert not writer._file_handle.closed

        writer.close()

        assert writer._file_handle.closed

    def test_close_idempotent(self, tmp_path):
        """Verify close() can be called multiple times safely."""
        output_file = tmp_path / "idempotent_output.csv"
        writer = IncrementalCSVWriter(output_file)

        writer.write_result(create_test_result(row_number=1))

        writer.close()
        writer.close()  # Should not raise
        writer.close()  # Should not raise

    def test_close_without_writes(self, tmp_path):
        """Verify close() works when no writes occurred."""
        output_file = tmp_path / "no_writes_output.csv"
        writer = IncrementalCSVWriter(output_file)

        # Close without writing
        writer.close()  # Should not raise

    def test_close_stdout_mode(self):
        """Verify close() works in stdout mode."""
        writer = IncrementalCSVWriter(None)

        writer.close()  # Should not raise

    def test_write_after_close_raises_error(self, tmp_path):
        """Verify writing after close() raises ValueError."""
        output_file = tmp_path / "closed_output.csv"
        writer = IncrementalCSVWriter(output_file)

        writer.write_result(create_test_result(row_number=1))
        writer.close()

        with pytest.raises(ValueError, match="closed|cannot write"):
            writer.write_result(create_test_result(row_number=2))


class TestStdoutMode:
    """Test stdout mode functionality."""

    def test_stdout_mode_still_works(self, capsys):
        """Verify stdout mode works with optimized implementation."""
        writer = IncrementalCSVWriter(None)

        result1 = create_test_result(row_number=1, variables={"topic": "AI"})
        result2 = create_test_result(row_number=2, variables={"topic": "ML"})

        writer.write_result(result1)
        writer.write_result(result2)
        writer.close()

        captured = capsys.readouterr()

        # Verify headers and data in stdout
        lines = captured.out.strip().split('\n')
        assert len(lines) == 3  # header + 2 rows
        assert 'provider' in lines[0]
        assert 'AI' in lines[1]
        assert 'ML' in lines[2]

    def test_stdout_buffer_flushed(self, capsys):
        """Verify stdout is flushed after each write."""
        writer = IncrementalCSVWriter(None)
        writer.write_result(create_test_result(row_number=1))

        # Verify output appears (flush is working)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_api_unchanged(self, tmp_path):
        """Verify existing API remains unchanged."""
        output_file = tmp_path / "compat_output.csv"
        writer = IncrementalCSVWriter(output_file)

        # All existing methods should still exist
        assert hasattr(writer, 'write_result')
        assert hasattr(writer, 'write_results')
        assert hasattr(writer, 'reset')
        assert hasattr(writer, 'exists')
        assert hasattr(writer, 'headers_written')
        assert hasattr(writer, 'get_existing_fieldnames')
        assert hasattr(writer, 'validate_compatibility')

        # Should work as before
        results = [create_test_result(row_number=i) for i in range(3)]
        writer.write_results(results)
        writer.close()

        with open(output_file, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_reset_closes_file_handle(self, tmp_path):
        """Verify reset() closes file handle before resetting state."""
        output_file = tmp_path / "reset_output.csv"
        writer = IncrementalCSVWriter(output_file)

        writer.write_result(create_test_result(row_number=1))

        old_handle = writer._file_handle
        writer.reset()

        # Old handle should be closed
        assert old_handle.closed

        # State should be reset
        assert not writer._headers_written
        assert writer._fieldnames is None

        # Should be able to write again
        writer.write_result(create_test_result(row_number=2))
        writer.close()

    def test_validate_compatibility_still_works(self, tmp_path):
        """Verify validate_compatibility works with new implementation."""
        output_file = tmp_path / "validate_output.csv"

        result1 = create_test_result(row_number=1, variables={"topic": "AI"})

        writer = IncrementalCSVWriter(output_file)
        writer.write_result(result1)

        # Compatible result
        result2 = create_test_result(row_number=2, variables={"topic": "ML"})
        is_compat, error = writer.validate_compatibility(result2)
        assert is_compat is True

        # Incompatible result
        result3 = create_test_result(row_number=3, variables={"topic": "AI", "source": "Wiki"})
        is_compat, error = writer.validate_compatibility(result3)
        assert is_compat is False

        writer.close()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_large_concurrent_write_load(self, tmp_path):
        """Stress test with many threads writing many rows."""
        output_file = tmp_path / "stress_output.csv"
        writer = IncrementalCSVWriter(output_file)

        num_threads = 20
        writes_per_thread = 50

        def write_batch(thread_id):
            for i in range(writes_per_thread):
                result = create_test_result(row_number=thread_id * writes_per_thread + i)
                writer.write_result(result)

        threads = [threading.Thread(target=write_batch, args=(i,)) for i in range(num_threads)]

        start = time.time()

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        elapsed = time.time() - start
        writer.close()

        # Verify all rows written
        with open(output_file, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == num_threads * writes_per_thread

        # Should complete in reasonable time (< 10 seconds)
        assert elapsed < 10.0

    def test_unicode_handling_in_concurrent_writes(self, tmp_path):
        """Verify Unicode characters handled correctly with concurrent writes."""
        output_file = tmp_path / "unicode_output.csv"
        writer = IncrementalCSVWriter(output_file)

        unicode_strings = [
            "Hello 世界",
            "Émoji 😀",
            "Русский текст",
            "العربية",
            "עברית"
        ]

        def write_unicode(index):
            result = create_test_result(
                row_number=index,
                output=unicode_strings[index % len(unicode_strings)]
            )
            writer.write_result(result)

        threads = [threading.Thread(target=write_unicode, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        writer.close()

        # Verify Unicode preserved
        with open(output_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            assert "世界" in content
            assert "😀" in content
            assert "Русский" in content
