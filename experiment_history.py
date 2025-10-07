"""
Experiment History Tracking
Persistent storage and querying of chaos experiments using SQLite
"""

import sqlite3
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import asdict
from chaos_controller import ChaosExperiment, ExperimentStatus, ExperimentMetrics
import logging


class ExperimentHistory:
    """
    Tracks and persists chaos experiment history using SQLite
    Enables historical analysis, trend detection, and comparison
    """

    def __init__(self, db_path: str = "chaos_experiments.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    fault_type TEXT NOT NULL,
                    parameters TEXT,
                    duration INTEGER,
                    status TEXT,
                    rollback_strategy TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    recovery_time REAL,
                    accuracy_degradation REAL,
                    pipeline_downtime REAL,
                    error_count INTEGER,
                    custom_metrics TEXT,
                    resilience_score REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

            # Tags table for flexible metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

            # Indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment_name
                ON experiments(name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment_status
                ON experiments(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment_created
                ON experiments(created_at)
            """)

            conn.commit()
            self.logger.info(f"Database initialized: {self.db_path}")

    def save_experiment(
        self,
        experiment: ChaosExperiment,
        resilience_score: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> int:
        """Save experiment and its metrics to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert experiment
            cursor.execute("""
                INSERT INTO experiments
                (name, layer, fault_type, parameters, duration, status, rollback_strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.name,
                experiment.layer.value,
                experiment.fault_type,
                json.dumps(experiment.parameters),
                experiment.duration,
                experiment.status.value,
                experiment.rollback_strategy
            ))

            experiment_id = cursor.lastrowid

            # Insert metrics if available
            if experiment.metrics:
                metrics = experiment.metrics
                cursor.execute("""
                    INSERT INTO experiment_metrics
                    (experiment_id, start_time, end_time, recovery_time,
                     accuracy_degradation, pipeline_downtime, error_count,
                     custom_metrics, resilience_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    metrics.start_time.isoformat(),
                    metrics.end_time.isoformat() if metrics.end_time else None,
                    metrics.recovery_time,
                    metrics.accuracy_degradation,
                    metrics.pipeline_downtime,
                    metrics.error_count,
                    json.dumps(metrics.custom_metrics),
                    resilience_score
                ))

            # Insert tags
            if tags:
                for key, value in tags.items():
                    cursor.execute("""
                        INSERT INTO experiment_tags (experiment_id, key, value)
                        VALUES (?, ?, ?)
                    """, (experiment_id, key, value))

            conn.commit()
            self.logger.info(f"Saved experiment: {experiment.name} (ID: {experiment_id})")

            return experiment_id

    def get_experiment_by_id(self, experiment_id: int) -> Optional[Dict]:
        """Retrieve experiment by ID with all metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT e.*, m.start_time, m.end_time, m.recovery_time,
                       m.accuracy_degradation, m.pipeline_downtime, m.error_count,
                       m.custom_metrics, m.resilience_score
                FROM experiments e
                LEFT JOIN experiment_metrics m ON e.id = m.experiment_id
                WHERE e.id = ?
            """, (experiment_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def query_experiments(
        self,
        name_pattern: Optional[str] = None,
        layer: Optional[str] = None,
        fault_type: Optional[str] = None,
        status: Optional[str] = None,
        min_resilience_score: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query experiments with filters"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT e.*, m.resilience_score
                FROM experiments e
                LEFT JOIN experiment_metrics m ON e.id = m.experiment_id
                WHERE 1=1
            """
            params = []

            if name_pattern:
                query += " AND e.name LIKE ?"
                params.append(f"%{name_pattern}%")

            if layer:
                query += " AND e.layer = ?"
                params.append(layer)

            if fault_type:
                query += " AND e.fault_type = ?"
                params.append(fault_type)

            if status:
                query += " AND e.status = ?"
                params.append(status)

            if min_resilience_score is not None:
                query += " AND m.resilience_score >= ?"
                params.append(min_resilience_score)

            query += " ORDER BY e.created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_experiment_trends(
        self,
        experiment_name: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze trends for a specific experiment over time"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    AVG(m.resilience_score) as avg_score,
                    MIN(m.resilience_score) as min_score,
                    MAX(m.resilience_score) as max_score,
                    AVG(m.recovery_time) as avg_recovery_time,
                    SUM(CASE WHEN e.status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN e.status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM experiments e
                JOIN experiment_metrics m ON e.id = m.experiment_id
                WHERE e.name = ?
                AND e.created_at >= datetime('now', '-' || ? || ' days')
            """, (experiment_name, days))

            row = cursor.fetchone()

            if row and row[0] > 0:  # total_runs > 0
                return {
                    "experiment_name": experiment_name,
                    "period_days": days,
                    "total_runs": row[0],
                    "avg_resilience_score": round(row[1], 2) if row[1] else 0,
                    "min_resilience_score": round(row[2], 2) if row[2] else 0,
                    "max_resilience_score": round(row[3], 2) if row[3] else 0,
                    "avg_recovery_time": round(row[4], 2) if row[4] else 0,
                    "completed": row[5],
                    "failed": row[6],
                    "success_rate": round((row[5] / row[0] * 100), 2) if row[0] > 0 else 0
                }
            return {"experiment_name": experiment_name, "status": "No data available"}

    def compare_experiments(
        self,
        experiment_ids: List[int]
    ) -> Dict[str, Any]:
        """Compare multiple experiments side-by-side"""
        results = []

        for exp_id in experiment_ids:
            exp_data = self.get_experiment_by_id(exp_id)
            if exp_data:
                results.append({
                    "id": exp_id,
                    "name": exp_data["name"],
                    "fault_type": exp_data["fault_type"],
                    "resilience_score": exp_data.get("resilience_score", 0),
                    "recovery_time": exp_data.get("recovery_time", 0),
                    "status": exp_data["status"]
                })

        return {
            "comparison": results,
            "best_resilience": max(results, key=lambda x: x["resilience_score"]) if results else None,
            "worst_resilience": min(results, key=lambda x: x["resilience_score"]) if results else None
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get overall summary statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_experiments,
                    COUNT(DISTINCT name) as unique_experiments,
                    AVG(m.resilience_score) as avg_score,
                    SUM(CASE WHEN e.status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN e.status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM experiments e
                LEFT JOIN experiment_metrics m ON e.id = m.experiment_id
            """)

            row = cursor.fetchone()

            # Get fault type distribution
            cursor.execute("""
                SELECT fault_type, COUNT(*) as count
                FROM experiments
                GROUP BY fault_type
                ORDER BY count DESC
            """)
            fault_types = dict(cursor.fetchall())

            return {
                "total_experiments": row[0],
                "unique_experiments": row[1],
                "avg_resilience_score": round(row[2], 2) if row[2] else 0,
                "completed": row[3],
                "failed": row[4],
                "success_rate": round((row[3] / row[0] * 100), 2) if row[0] > 0 else 0,
                "fault_type_distribution": fault_types
            }

    def export_to_csv(self, filepath: str, query_params: Optional[Dict] = None):
        """Export experiments to CSV"""
        import csv

        experiments = self.query_experiments(**(query_params or {}))

        if not experiments:
            self.logger.warning("No experiments to export")
            return

        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = experiments[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for exp in experiments:
                writer.writerow(exp)

        self.logger.info(f"Exported {len(experiments)} experiments to {filepath}")

    def delete_old_experiments(self, days: int = 90) -> int:
        """Delete experiments older than specified days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM experiments
                WHERE created_at < datetime('now', '-' || ? || ' days')
            """, (days,))

            deleted_count = cursor.rowcount
            conn.commit()

            self.logger.info(f"Deleted {deleted_count} experiments older than {days} days")
            return deleted_count


# Example usage
if __name__ == "__main__":
    from chaos_controller import ChaosLayer

    history = ExperimentHistory()

    # Create sample experiment
    experiment = ChaosExperiment(
        name="test_data_corruption",
        layer=ChaosLayer.DATA,
        fault_type="data_corruption",
        parameters={"corruption_rate": 0.1},
        duration=60,
        status=ExperimentStatus.COMPLETED
    )

    experiment.metrics = ExperimentMetrics(
        start_time=datetime.now(),
        end_time=datetime.now(),
        recovery_time=45.2,
        accuracy_degradation=5.3,
        error_count=2
    )

    # Save experiment
    exp_id = history.save_experiment(
        experiment,
        resilience_score=85.5,
        tags={"env": "staging", "version": "v1.2.3"}
    )

    print(f"Saved experiment ID: {exp_id}")

    # Query experiments
    results = history.query_experiments(layer="data", limit=10)
    print(f"Found {len(results)} data layer experiments")

    # Get summary
    summary = history.get_summary_statistics()
    print(f"\n📊 Summary Statistics:")
    print(f"  Total Experiments: {summary['total_experiments']}")
    print(f"  Avg Resilience Score: {summary['avg_resilience_score']}")
    print(f"  Success Rate: {summary['success_rate']}%")
