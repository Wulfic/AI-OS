"""Tests for optimization cleanup utility."""

import pytest
import tempfile
from pathlib import Path
from aios.utils.optimization_cleanup import (
    extract_session_id,
    group_files_by_session,
    cleanup_old_optimization_runs,
    get_session_timestamp
)


class TestSessionIdExtraction:
    """Test session ID extraction from various filename patterns."""
    
    def test_gpu_metrics_simple(self):
        assert extract_session_id("gpu_metrics_abc12345.jsonl") == "abc12345"
    
    def test_gpu_metrics_gen(self):
        assert extract_session_id("gpu_metrics_gen_def67890.jsonl") == "def67890"
    
    def test_gpu_metrics_train(self):
        assert extract_session_id("gpu_metrics_train_12345678.jsonl") == "12345678"
    
    def test_progressive_results(self):
        assert extract_session_id("progressive_results_aabbccdd.json") == "aabbccdd"
    
    def test_results(self):
        assert extract_session_id("results_11223344.json") == "11223344"
    
    def test_gen(self):
        assert extract_session_id("gen_55667788.jsonl") == "55667788"
    
    def test_stop_flag(self):
        assert extract_session_id("stop_99aabbcc.flag") == "99aabbcc"
    
    def test_train(self):
        assert extract_session_id("train_ddeeff00.jsonl") == "ddeeff00"
    
    def test_no_match(self):
        assert extract_session_id("random_file.txt") is None
        assert extract_session_id("config.yaml") is None
        assert extract_session_id("gpu_metrics.jsonl") is None  # Missing session ID


class TestFileGrouping:
    """Test grouping files by session ID."""
    
    def test_group_files_by_session(self, tmp_path):
        """Test that files are correctly grouped by session ID."""
        # Create test files for 3 sessions
        session1_files = [
            "gpu_metrics_aaa11111.jsonl",
            "results_aaa11111.json",
            "train_aaa11111.jsonl"
        ]
        session2_files = [
            "gpu_metrics_bbb22222.jsonl",
            "progressive_results_bbb22222.json"
        ]
        session3_files = [
            "stop_ccc33333.flag",
            "gen_ccc33333.jsonl"
        ]
        
        all_files = session1_files + session2_files + session3_files
        
        # Create the files
        for filename in all_files:
            (tmp_path / filename).touch()
        
        # Group them
        grouped = group_files_by_session(tmp_path)
        
        # Verify grouping
        assert len(grouped) == 3
        assert "aaa11111" in grouped
        assert "bbb22222" in grouped
        assert "ccc33333" in grouped
        
        # Verify file counts
        assert len(grouped["aaa11111"]) == 3
        assert len(grouped["bbb22222"]) == 2
        assert len(grouped["ccc33333"]) == 2


class TestCleanup:
    """Test the cleanup functionality."""
    
    def test_cleanup_keeps_recent(self, tmp_path):
        """Test that cleanup keeps the most recent N sessions."""
        import time
        
        # Create 5 sessions with different timestamps
        sessions = ["aaa11111", "bbb22222", "ccc33333", "ddd44444", "eee55555"]
        
        for i, session in enumerate(sessions):
            file1 = tmp_path / f"gpu_metrics_{session}.jsonl"
            file2 = tmp_path / f"results_{session}.json"
            
            file1.write_text(f"metrics for {session}")
            file2.write_text(f"results for {session}")
            
            # Sleep briefly to ensure different timestamps
            time.sleep(0.01)
        
        # Run cleanup keeping last 3
        stats = cleanup_old_optimization_runs(tmp_path, keep_last_n=3, dry_run=False)
        
        # Verify statistics
        assert stats["success"] is True
        assert stats["sessions_found"] == 5
        assert stats["sessions_kept"] == 3
        assert stats["sessions_deleted"] == 2
        assert stats["files_deleted"] == 4  # 2 sessions * 2 files each
        
        # Verify the oldest 2 sessions were deleted
        remaining_files = list(tmp_path.iterdir())
        remaining_sessions = set()
        for f in remaining_files:
            session_id = extract_session_id(f.name)
            if session_id:
                remaining_sessions.add(session_id)
        
        # Should have the 3 most recent sessions
        assert len(remaining_sessions) == 3
        assert "aaa11111" not in remaining_sessions  # Oldest
        assert "bbb22222" not in remaining_sessions  # Second oldest
        assert "ccc33333" in remaining_sessions
        assert "ddd44444" in remaining_sessions
        assert "eee55555" in remaining_sessions  # Most recent
    
    def test_cleanup_dry_run(self, tmp_path):
        """Test that dry run doesn't actually delete files."""
        # Create 5 sessions
        sessions = ["aaa11111", "bbb22222", "ccc33333", "ddd44444", "eee55555"]
        
        for session in sessions:
            (tmp_path / f"gpu_metrics_{session}.jsonl").touch()
            (tmp_path / f"results_{session}.json").touch()
        
        # Run cleanup in dry run mode
        stats = cleanup_old_optimization_runs(tmp_path, keep_last_n=2, dry_run=True)
        
        # Verify statistics show what would be deleted
        assert stats["success"] is True
        assert stats["sessions_found"] == 5
        assert stats["sessions_deleted"] == 3
        assert stats["files_deleted"] == 6
        assert stats["dry_run"] is True
        
        # Verify no files were actually deleted
        remaining_files = list(tmp_path.iterdir())
        assert len(remaining_files) == 10  # All files still there
    
    def test_cleanup_with_no_files(self, tmp_path):
        """Test cleanup on empty directory."""
        stats = cleanup_old_optimization_runs(tmp_path, keep_last_n=3, dry_run=False)
        
        assert stats["success"] is True
        assert stats["sessions_found"] == 0
        assert stats["sessions_deleted"] == 0
    
    def test_cleanup_with_fewer_than_keep(self, tmp_path):
        """Test cleanup when there are fewer sessions than keep_last_n."""
        # Create only 2 sessions
        for session in ["aaa11111", "bbb22222"]:
            (tmp_path / f"gpu_metrics_{session}.jsonl").touch()
        
        # Try to keep 5
        stats = cleanup_old_optimization_runs(tmp_path, keep_last_n=5, dry_run=False)
        
        assert stats["success"] is True
        assert stats["sessions_found"] == 2
        assert stats["sessions_kept"] == 2
        assert stats["sessions_deleted"] == 0
        
        # All files should still exist
        assert len(list(tmp_path.iterdir())) == 2


class TestSessionTimestamp:
    """Test getting session timestamp."""
    
    def test_get_most_recent_timestamp(self, tmp_path):
        """Test that we get the most recent timestamp from a session's files."""
        import time
        
        # Create 3 files with different timestamps
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file3 = tmp_path / "file3.txt"
        
        file1.touch()
        time.sleep(0.01)
        file2.touch()
        time.sleep(0.01)
        file3.touch()
        
        files = [file1, file2, file3]
        timestamp = get_session_timestamp(files)
        
        # Should be the timestamp of file3 (most recent)
        assert timestamp == file3.stat().st_mtime
    
    def test_empty_file_list(self):
        """Test handling of empty file list."""
        assert get_session_timestamp([]) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
