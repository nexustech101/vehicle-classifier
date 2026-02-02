"""Persistent database for reports and data."""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Database:
    """SQLite database for persistent storage."""
    
    def __init__(self, db_path: str = "reports.db"):
        """Initialize database."""
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    vehicle_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    user_id TEXT
                )
            """)
            
            # Create indices for reports
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_vehicle_id ON reports(vehicle_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_user_id ON reports(user_id)")
            
            # Classification results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    predictions TEXT NOT NULL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_time_ms REAL,
                    user_id TEXT
                )
            """)
            
            # Create indices for classifications
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_classifications_image_path ON classifications(image_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_classifications_created_at ON classifications(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_classifications_user_id ON classifications(user_id)")
            
            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for audit_log
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at)")
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def save_report(self, vehicle_id: str, report_data: Dict[str, Any], 
                   user_id: Optional[str] = None) -> str:
        """Save report to database."""
        import uuid
        report_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reports (id, vehicle_id, data, user_id)
                VALUES (?, ?, ?, ?)
            """, (report_id, vehicle_id, json.dumps(report_data), user_id))
            conn.commit()
        
        logger.info(f"Report saved: {report_id}")
        return report_id
    
    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve report from database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, vehicle_id, data, created_at
                FROM reports
                WHERE id = ? AND status = 'active'
            """, (report_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'vehicle_id': row['vehicle_id'],
                    'data': json.loads(row['data']),
                    'created_at': row['created_at']
                }
        
        return None
    
    def save_classification(self, image_path: str, predictions: Dict[str, Any],
                           confidence: float, processing_time_ms: float,
                           user_id: Optional[str] = None):
        """Save classification result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO classifications 
                (image_path, predictions, confidence, processing_time_ms, user_id)
                VALUES (?, ?, ?, ?, ?)
            """, (image_path, json.dumps(predictions), confidence, processing_time_ms, user_id))
            conn.commit()
    
    def get_classifications(self, limit: int = 100, 
                          user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent classifications."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("""
                    SELECT id, image_path, predictions, confidence, created_at, processing_time_ms
                    FROM classifications
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
            else:
                cursor.execute("""
                    SELECT id, image_path, predictions, confidence, created_at, processing_time_ms
                    FROM classifications
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def log_audit(self, user_id: str, action: str, resource: str, 
                 details: Optional[str] = None, ip_address: Optional[str] = None):
        """Log audit event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (user_id, action, resource, details, ip_address)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, action, resource, details, ip_address))
            conn.commit()
        
        logger.info(f"Audit log: {user_id} {action} {resource}")
    
    def get_audit_logs(self, user_id: Optional[str] = None,
                      action: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve audit logs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def delete_report(self, report_id: str):
        """Soft delete report."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE reports
                SET status = 'deleted'
                WHERE id = ?
            """, (report_id,))
            conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) as count FROM reports WHERE status = 'active'")
            report_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM classifications")
            classification_count = cursor.fetchone()['count']
            
            # Get average processing time
            cursor.execute("SELECT AVG(processing_time_ms) as avg_time FROM classifications")
            avg_time = cursor.fetchone()['avg_time'] or 0
            
            return {
                'total_reports': report_count,
                'total_classifications': classification_count,
                'average_processing_time_ms': avg_time
            }
    
    def health_check(self) -> bool:
        """Check database health by executing a simple query."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM reports")
                cursor.fetchone()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
