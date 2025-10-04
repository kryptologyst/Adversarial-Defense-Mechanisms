"""
Database utilities for storing and retrieving experiment results.
"""
import sqlite3
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import datetime
from contextlib import contextmanager


class ExperimentDatabase:
    """SQLite database for storing adversarial defense experiment results."""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    model_type TEXT NOT NULL,
                    attack_types TEXT NOT NULL,
                    defense_types TEXT NOT NULL,
                    epsilon_values TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config TEXT
                )
            """)
            
            # Results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    attack_type TEXT NOT NULL,
                    epsilon REAL NOT NULL,
                    defense_type TEXT,
                    accuracy REAL NOT NULL,
                    loss REAL,
                    sample_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL,
                    train_accuracy REAL,
                    val_loss REAL,
                    val_accuracy REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def create_experiment(self, name: str, description: str, model_type: str,
                         attack_types: List[str], defense_types: List[str],
                         epsilon_values: List[float], config: Dict[str, Any] = None) -> int:
        """
        Create a new experiment record.
        
        Args:
            name: Experiment name
            description: Experiment description
            model_type: Type of model used
            attack_types: List of attack types tested
            defense_types: List of defense types tested
            epsilon_values: List of epsilon values tested
            config: Additional configuration
        
        Returns:
            Experiment ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO experiments (name, description, model_type, attack_types, 
                                      defense_types, epsilon_values, config)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                description,
                model_type,
                json.dumps(attack_types),
                json.dumps(defense_types),
                json.dumps(epsilon_values),
                json.dumps(config) if config else None
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def add_result(self, experiment_id: int, attack_type: str, epsilon: float,
                   defense_type: Optional[str], accuracy: float, 
                   loss: Optional[float] = None, sample_count: Optional[int] = None):
        """
        Add a result to the database.
        
        Args:
            experiment_id: ID of the experiment
            attack_type: Type of attack
            epsilon: Epsilon value used
            defense_type: Type of defense (None for no defense)
            accuracy: Accuracy achieved
            loss: Loss value (optional)
            sample_count: Number of samples tested (optional)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO results (experiment_id, attack_type, epsilon, defense_type,
                                  accuracy, loss, sample_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, attack_type, epsilon, defense_type, 
                  accuracy, loss, sample_count))
            
            conn.commit()
    
    def add_model_performance(self, experiment_id: int, epoch: int,
                             train_loss: Optional[float] = None,
                             train_accuracy: Optional[float] = None,
                             val_loss: Optional[float] = None,
                             val_accuracy: Optional[float] = None):
        """
        Add model performance metrics for an epoch.
        
        Args:
            experiment_id: ID of the experiment
            epoch: Epoch number
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_loss: Validation loss
            val_accuracy: Validation accuracy
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_performance (experiment_id, epoch, train_loss,
                                             train_accuracy, val_loss, val_accuracy)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (experiment_id, epoch, train_loss, train_accuracy, val_loss, val_accuracy))
            
            conn.commit()
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment details by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM experiments WHERE id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'model_type': row[3],
                    'attack_types': json.loads(row[4]),
                    'defense_types': json.loads(row[5]),
                    'epsilon_values': json.loads(row[6]),
                    'created_at': row[7],
                    'config': json.loads(row[8]) if row[8] else None
                }
            return None
    
    def get_experiment_results(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Get all results for an experiment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM results WHERE experiment_id = ? ORDER BY attack_type, epsilon, defense_type
            """, (experiment_id,))
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    'id': row[0],
                    'experiment_id': row[1],
                    'attack_type': row[2],
                    'epsilon': row[3],
                    'defense_type': row[4],
                    'accuracy': row[5],
                    'loss': row[6],
                    'sample_count': row[7],
                    'created_at': row[8]
                })
            return results
    
    def get_model_performance(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Get model performance metrics for an experiment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM model_performance WHERE experiment_id = ? ORDER BY epoch
            """, (experiment_id,))
            
            rows = cursor.fetchall()
            performance = []
            for row in rows:
                performance.append({
                    'id': row[0],
                    'experiment_id': row[1],
                    'epoch': row[2],
                    'train_loss': row[3],
                    'train_accuracy': row[4],
                    'val_loss': row[5],
                    'val_accuracy': row[6],
                    'created_at': row[7]
                })
            return performance
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM experiments ORDER BY created_at DESC
            """)
            
            rows = cursor.fetchall()
            experiments = []
            for row in rows:
                experiments.append({
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'model_type': row[3],
                    'attack_types': json.loads(row[4]),
                    'defense_types': json.loads(row[5]),
                    'epsilon_values': json.loads(row[6]),
                    'created_at': row[7],
                    'config': json.loads(row[8]) if row[8] else None
                })
            return experiments
    
    def get_best_defenses(self, experiment_id: int) -> Dict[str, Dict[str, Any]]:
        """Get the best defense for each attack type and epsilon combination."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT attack_type, epsilon, defense_type, accuracy
                FROM results 
                WHERE experiment_id = ? AND defense_type IS NOT NULL
                ORDER BY attack_type, epsilon, accuracy DESC
            """, (experiment_id,))
            
            rows = cursor.fetchall()
            best_defenses = {}
            
            for row in rows:
                attack_type, epsilon, defense_type, accuracy = row
                key = f"{attack_type}_{epsilon}"
                
                if key not in best_defenses:
                    best_defenses[key] = {
                        'attack_type': attack_type,
                        'epsilon': epsilon,
                        'defense_type': defense_type,
                        'accuracy': accuracy
                    }
            
            return best_defenses
    
    def export_experiment_data(self, experiment_id: int, export_path: str):
        """Export all experiment data to JSON file."""
        experiment = self.get_experiment(experiment_id)
        results = self.get_experiment_results(experiment_id)
        performance = self.get_model_performance(experiment_id)
        
        export_data = {
            'experiment': experiment,
            'results': results,
            'performance': performance,
            'exported_at': datetime.datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Experiment data exported to {export_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count experiments
            cursor.execute("SELECT COUNT(*) FROM experiments")
            experiment_count = cursor.fetchone()[0]
            
            # Count results
            cursor.execute("SELECT COUNT(*) FROM results")
            result_count = cursor.fetchone()[0]
            
            # Count unique attack types
            cursor.execute("SELECT COUNT(DISTINCT attack_type) FROM results")
            attack_types_count = cursor.fetchone()[0]
            
            # Count unique defense types
            cursor.execute("SELECT COUNT(DISTINCT defense_type) FROM results WHERE defense_type IS NOT NULL")
            defense_types_count = cursor.fetchone()[0]
            
            # Average accuracy by attack type
            cursor.execute("""
                SELECT attack_type, AVG(accuracy) as avg_accuracy, COUNT(*) as count
                FROM results 
                GROUP BY attack_type
            """)
            attack_stats = cursor.fetchall()
            
            return {
                'experiment_count': experiment_count,
                'result_count': result_count,
                'attack_types_count': attack_types_count,
                'defense_types_count': defense_types_count,
                'attack_statistics': [
                    {'attack_type': row[0], 'avg_accuracy': row[1], 'count': row[2]}
                    for row in attack_stats
                ]
            }
