"""Evaluation history database for tracking past evaluations."""

from __future__ import annotations

import json
import sqlite3
import time
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from aios.core.evaluation import EvaluationResult

logger = logging.getLogger(__name__)


class EvaluationHistory:
    """Database manager for evaluation history."""
    
    def __init__(self, db_path: str = "artifacts/evaluation/history.db") -> None:
        """Initialize the history database.
        
        Args:
            db_path: Path to SQLite database file
        """
        logger.info(f"Initializing evaluation history database: {db_path}")
        self.db_path = db_path
        
        # Ensure directory exists
        db_dir = Path(db_path).parent
        if not db_dir.exists():
            logger.info(f"Creating database directory: {db_dir}")
            db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        logger.info("Initializing database schema")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_source TEXT,
                    model_args TEXT,
                    overall_score REAL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration REAL,
                    output_path TEXT,
                    tasks TEXT NOT NULL,
                    config TEXT,
                    raw_results TEXT,
                    created_at REAL NOT NULL,
                    notes TEXT,
                    samples_path TEXT
                )
            """)
            
            # Add samples_path column if it doesn't exist (for existing databases)
            try:
                cursor.execute("ALTER TABLE evaluations ADD COLUMN samples_path TEXT")
                logger.debug("Added missing column: samples_path")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Evaluation scores table (per-benchmark scores)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id INTEGER NOT NULL,
                    benchmark_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    stderr REAL,
                    raw_data TEXT,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
                        ON DELETE CASCADE
                )
            """)
            
            # Indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluations_model 
                ON evaluations (model_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluations_status 
                ON evaluations (status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluations_created 
                ON evaluations (created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scores_evaluation 
                ON evaluation_scores (evaluation_id)
            """)
            
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    def save_evaluation(
        self,
        result: EvaluationResult,
        model_name: str,
        model_source: str = "",
        model_args: str = "",
        tasks: list[str] | None = None,
        config: dict[str, Any] | None = None,
        notes: str = "",
        samples_path: str = "",
    ) -> int:
        """Save an evaluation to the database.
        
        Args:
            result: EvaluationResult to save
            model_name: Name of evaluated model
            model_source: Source type (huggingface, local, brain)
            model_args: Model arguments used
            tasks: List of task names evaluated
            config: Configuration dict used
            notes: Optional notes about this evaluation
            samples_path: Path to samples directory (if log_samples was enabled)
            
        Returns:
            ID of saved evaluation
        """
        logger.info(f"Saving evaluation for model: {model_name}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Serialize data
                logger.debug(f"Serializing tasks and config for model: {model_name}")
                tasks_str = ",".join(tasks) if tasks else ""
                config_str = json.dumps(config) if config else ""
                raw_results_str = json.dumps(result.raw_results) if result.raw_results else ""
                
                # Insert evaluation
                cursor.execute("""
                    INSERT INTO evaluations (
                        model_name, model_source, model_args,
                        overall_score, status, error_message,
                        start_time, end_time, duration,
                        output_path, tasks, config, raw_results,
                        created_at, notes, samples_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    model_source,
                    model_args,
                    result.overall_score,
                    result.status,
                    result.error_message,
                    result.start_time,
                    result.end_time,
                    result.duration,
                    result.output_path,
                    tasks_str,
                    config_str,
                    raw_results_str,
                    time.time(),
                    notes,
                    samples_path,
                ))
                
                evaluation_id = cursor.lastrowid
                
                if evaluation_id is None:
                    raise RuntimeError("Failed to insert evaluation")
                
                logger.info(f"Evaluation saved with ID: {evaluation_id}")
                
                # Insert benchmark scores
                score_count = 0
                for benchmark_name, benchmark_data in result.benchmark_scores.items():
                    scores_dict = benchmark_data.get("scores", {})
                    raw_data = benchmark_data.get("raw", {})
                    
                    for metric_name, score_value in scores_dict.items():
                        # Try to get stderr
                        stderr = None
                        if f"{metric_name}_stderr" in raw_data:
                            stderr = raw_data[f"{metric_name}_stderr"]
                        
                        cursor.execute("""
                            INSERT INTO evaluation_scores (
                                evaluation_id, benchmark_name, metric_name,
                                score, stderr, raw_data
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            evaluation_id,
                            benchmark_name,
                            metric_name,
                            float(score_value),
                            stderr,
                            json.dumps(raw_data),
                        ))
                        score_count += 1
                
                logger.debug(f"Inserted {score_count} benchmark scores")
                conn.commit()
                return evaluation_id
        except Exception as e:
            logger.error(f"Failed to save evaluation for {model_name}: {e}")
            raise
    
    def get_evaluation(self, evaluation_id: int) -> Optional[dict[str, Any]]:
        """Get a single evaluation by ID.
        
        Args:
            evaluation_id: ID of evaluation
            
        Returns:
            Evaluation dict or None if not found
        """
        logger.debug(f"Fetching evaluation ID: {evaluation_id}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM evaluations WHERE id = ?
            """, (evaluation_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Evaluation not found: ID {evaluation_id}")
                return None
            
            eval_dict = dict(row)
            
            # Parse JSON fields
            if eval_dict.get("config"):
                eval_dict["config"] = json.loads(eval_dict["config"])
            if eval_dict.get("raw_results"):
                eval_dict["raw_results"] = json.loads(eval_dict["raw_results"])
            if eval_dict.get("tasks"):
                eval_dict["tasks"] = eval_dict["tasks"].split(",")
            
            # Get scores
            cursor.execute("""
                SELECT benchmark_name, metric_name, score, stderr
                FROM evaluation_scores
                WHERE evaluation_id = ?
                ORDER BY benchmark_name, metric_name
            """, (evaluation_id,))
            
            scores = []
            for score_row in cursor.fetchall():
                scores.append(dict(score_row))
            
            eval_dict["scores"] = scores
            logger.debug(f"Retrieved {len(scores)} scores for evaluation {evaluation_id}")
            
            return eval_dict
    
    def get_recent_evaluations(
        self,
        limit: int = 50,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get recent evaluations.
        
        Args:
            limit: Maximum number of results
            model_name: Filter by model name (partial match)
            status: Filter by status
            
        Returns:
            List of evaluation dicts
        """
        logger.debug(f"Query recent evaluations: limit={limit}, model={model_name}, status={status}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM evaluations WHERE 1=1"
            params: list[Any] = []
            
            if model_name:
                query += " AND model_name LIKE ?"
                params.append(f"%{model_name}%")
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                eval_dict = dict(row)
                
                # Parse tasks
                if eval_dict.get("tasks"):
                    eval_dict["tasks"] = eval_dict["tasks"].split(",")
                
                results.append(eval_dict)
            
            logger.info(f"Retrieved {len(results)} evaluations")
            return results
    
    def get_model_history(
        self,
        model_name: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get evaluation history for a specific model.
        
        Args:
            model_name: Model name (exact match)
            limit: Maximum number of results
            
        Returns:
            List of evaluation dicts ordered by date
        """
        return self.get_recent_evaluations(limit=limit, model_name=model_name)
    
    def get_benchmark_history(
        self,
        benchmark_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get evaluation history for a specific benchmark.
        
        Args:
            benchmark_name: Benchmark name
            limit: Maximum number of results
            
        Returns:
            List of evaluations that included this benchmark
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT e.*
                FROM evaluations e
                JOIN evaluation_scores es ON e.id = es.evaluation_id
                WHERE es.benchmark_name = ?
                ORDER BY e.created_at DESC
                LIMIT ?
            """, (benchmark_name, limit))
            
            results = []
            for row in cursor.fetchall():
                eval_dict = dict(row)
                if eval_dict.get("tasks"):
                    eval_dict["tasks"] = eval_dict["tasks"].split(",")
                results.append(eval_dict)
            
            return results
    
    def compare_evaluations(
        self,
        evaluation_ids: list[int],
    ) -> dict[str, Any]:
        """Compare multiple evaluations.
        
        Args:
            evaluation_ids: List of evaluation IDs to compare
            
        Returns:
            Comparison dict with aligned scores
        """
        evaluations = []
        for eval_id in evaluation_ids:
            eval_data = self.get_evaluation(eval_id)
            if eval_data:
                evaluations.append(eval_data)
        
        if not evaluations:
            return {}
        
        # Collect all unique benchmarks
        all_benchmarks = set()
        for eval_data in evaluations:
            for score in eval_data.get("scores", []):
                all_benchmarks.add(score["benchmark_name"])
        
        # Build comparison structure
        comparison = {
            "evaluations": [
                {
                    "id": e["id"],
                    "model_name": e["model_name"],
                    "overall_score": e["overall_score"],
                    "created_at": e["created_at"],
                }
                for e in evaluations
            ],
            "benchmarks": {},
        }
        
        for benchmark in sorted(all_benchmarks):
            comparison["benchmarks"][benchmark] = []
            
            for eval_data in evaluations:
                # Find score for this benchmark
                score_data = None
                for score in eval_data.get("scores", []):
                    if score["benchmark_name"] == benchmark:
                        score_data = {
                            "metric": score["metric_name"],
                            "score": score["score"],
                            "stderr": score["stderr"],
                        }
                        break
                
                comparison["benchmarks"][benchmark].append(score_data)
        
        return comparison
    
    def delete_evaluation(self, evaluation_id: int) -> bool:
        """Delete an evaluation.
        
        Args:
            evaluation_id: ID of evaluation to delete
            
        Returns:
            True if deleted, False if not found
        """
        logger.info(f"Deleting evaluation ID: {evaluation_id}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM evaluations WHERE id = ?
            """, (evaluation_id,))
            
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Evaluation {evaluation_id} deleted successfully")
                return True
            else:
                logger.warning(f"Evaluation not found for deletion: ID {evaluation_id}")
                return False
    
    def get_statistics(self) -> dict[str, Any]:
        """Get overall statistics.
        
        Returns:
            Statistics dict
        """
        logger.debug("Computing evaluation statistics")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total evaluations
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_evaluations = cursor.fetchone()[0]
            
            # By status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM evaluations
                GROUP BY status
            """)
            by_status = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Unique models
            cursor.execute("SELECT COUNT(DISTINCT model_name) FROM evaluations")
            unique_models = cursor.fetchone()[0]
            
            # Unique benchmarks
            cursor.execute("SELECT COUNT(DISTINCT benchmark_name) FROM evaluation_scores")
            unique_benchmarks = cursor.fetchone()[0]
            
            # Average scores
            cursor.execute("""
                SELECT AVG(overall_score) FROM evaluations
                WHERE status = 'completed'
            """)
            avg_score = cursor.fetchone()[0] or 0.0
            
            # Recent activity (last 7 days)
            week_ago = time.time() - (7 * 24 * 60 * 60)
            cursor.execute("""
                SELECT COUNT(*) FROM evaluations
                WHERE created_at > ?
            """, (week_ago,))
            recent_count = cursor.fetchone()[0]
            
            logger.info(f"Statistics: {total_evaluations} evaluations, {unique_models} models")
            
            return {
                "total_evaluations": total_evaluations,
                "by_status": by_status,
                "unique_models": unique_models,
                "unique_benchmarks": unique_benchmarks,
                "average_score": avg_score,
                "recent_count": recent_count,
            }
    
    def search(
        self,
        query: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search evaluations by model name or notes.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching evaluations
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM evaluations
                WHERE model_name LIKE ? OR notes LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))
            
            results = []
            for row in cursor.fetchall():
                eval_dict = dict(row)
                if eval_dict.get("tasks"):
                    eval_dict["tasks"] = eval_dict["tasks"].split(",")
                results.append(eval_dict)
            
            return results
