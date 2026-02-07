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
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)
            
            # Create indices for users
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
            
            # Reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    vehicle_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    user_id TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(username)
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
    
    def create_user(self, username: str, email: str, password: str, 
                   role: str = 'user') -> bool:
        """Create a new user account."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (username, email, password, role, is_active)
                    VALUES (?, ?, ?, ?, 1)
                """, (username, email, password, role))
                conn.commit()
                logger.info(f"User created: {username} with role {role}")
                return True
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed - duplicate username or email: {e}")
            return False
        except Exception as e:
            logger.error(f"User creation error: {e}")
            return False
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by username."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, email, password, role, is_active, created_at, last_login
                FROM users
                WHERE username = ?
            """, (username,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by email."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, email, password, role, is_active, created_at, last_login
                FROM users
                WHERE email = ?
            """, (email,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
    
    def update_user_role(self, username: str, role: str) -> bool:
        """Update user role."""
        valid_roles = {'user', 'admin'}
        if role not in valid_roles:
            logger.error(f"Invalid role: {role}")
            return False
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users
                    SET role = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE username = ?
                """, (role, username))
                conn.commit()
                logger.info(f"User role updated: {username} -> {role}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"User role update failed: {e}")
            return False
    
    def update_user_status(self, username: str, is_active: bool) -> bool:
        """Activate or deactivate user."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users
                    SET is_active = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE username = ?
                """, (int(is_active), username))
                conn.commit()
                logger.info(f"User status updated: {username} -> active={is_active}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"User status update failed: {e}")
            return False
    
    def update_last_login(self, username: str) -> bool:
        """Update user's last login timestamp."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users
                    SET last_login = CURRENT_TIMESTAMP
                    WHERE username = ?
                """, (username,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Last login update failed: {e}")
            return False
    
    def delete_user(self, username: str) -> bool:
        """Delete user account."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM users WHERE username = ?", (username,))
                conn.commit()
                logger.info(f"User deleted: {username}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"User deletion failed: {e}")
            return False
    
    def list_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List all users (admin only)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, email, role, is_active, created_at, last_login
                FROM users
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def user_exists(self, username: str) -> bool:
        """Check if user exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
            return cursor.fetchone() is not None

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
                 details: Optional[Dict[str, Any]] = None, ip_address: Optional[str] = None):
        """Log audit event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Convert details dict to JSON string if provided
            details_json = json.dumps(details) if details else None
            cursor.execute("""
                INSERT INTO audit_log (user_id, action, resource, details, ip_address)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, action, resource, details_json, ip_address))
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
    
    def close(self):
        """Close the database (no-op for SQLite per-call connections)."""
        logger.info(f"Database connection to {self.db_path} closed")
