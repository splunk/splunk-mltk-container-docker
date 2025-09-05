import os
import json
import gzip
import datetime
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union
import time


class RotatingJSONLLogger:
    """Logger for raw HTTP request and response data using JSONL format with rotation."""
    
    def __init__(self):
        # Configuration from environment
        self.log_dir = Path(os.getenv("LOG_DIR", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Rotation settings
        self.max_file_size_mb = int(os.getenv("RAW_LOGGING_ROTATION_SIZE", "100"))
        self.max_file_size = self.max_file_size_mb * 1024 * 1024
        self.retention_days = int(os.getenv("RAW_LOGGING_RETENTION_DAYS", "30"))
        self.compress_old_files = os.getenv("RAW_LOGGING_COMPRESS", "true").lower() == "true"
        
        # Logging settings
        self.max_body_log_size = int(os.getenv("MAX_BODY_LOG_SIZE", "1048576"))  # 1MB default
        self.include_headers = os.getenv("RAW_LOGGING_INCLUDE_HEADERS", "false").lower() == "true"
        self.redact_sensitive = os.getenv("RAW_LOGGING_REDACT_SENSITIVE", "true").lower() == "true"
        
        # File paths
        self.current_log_file = self.log_dir / "api_logs.jsonl"
        self.lock = threading.Lock()
        
        # Clean up old files on startup
        self._cleanup_old_files()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format with milliseconds."""
        return datetime.datetime.now().isoformat(timespec='milliseconds')
    
    def _truncate_body(self, body: bytes) -> tuple[Union[str, Dict[str, Any]], bool]:
        """Truncate and decode body if it exceeds max size."""
        truncated = False
        
        if len(body) > self.max_body_log_size:
            body = body[:self.max_body_log_size]
            truncated = True
        
        # Try to decode body
        try:
            text = body.decode('utf-8')
            # Try to parse as JSON
            try:
                return json.loads(text), truncated
            except json.JSONDecodeError:
                # Return as string if not valid JSON
                return text, truncated
        except UnicodeDecodeError:
            # Return as hex string if binary
            return f"<binary data: {body[:100].hex()}...>", truncated
    
    def _redact_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Redact sensitive information from headers."""
        if not self.redact_sensitive:
            return headers
        
        redacted = headers.copy()
        sensitive_keys = ['authorization', 'api-key', 'x-api-key', 'cookie', 'x-auth-token']
        
        for key in redacted:
            if key.lower() in sensitive_keys:
                # Keep first few characters for debugging
                value = redacted[key]
                if len(value) > 8:
                    redacted[key] = value[:8] + "***REDACTED***"
                else:
                    redacted[key] = "***REDACTED***"
        
        return redacted
    
    def log_request(self, request_id: str, method: str, path: str, 
                   headers: Dict[str, str], body: bytes, 
                   query_params: Optional[Dict[str, str]] = None,
                   client_ip: Optional[str] = None):
        """Log HTTP request data."""
        body_content = None
        body_truncated = False
        
        if body:
            body_content, body_truncated = self._truncate_body(body)
        
        entry = {
            "timestamp": self._get_timestamp(),
            "type": "request",
            "request_id": request_id,
            "client_ip": client_ip,
            "method": method,
            "path": path,
            "query_params": query_params or {},
            "body_size": len(body) if body else 0,
            "body_truncated": body_truncated,
            "body": body_content
        }
        
        # Add headers if configured
        if self.include_headers:
            entry["headers"] = self._redact_headers(headers)
        
        self._write_log(entry)
    
    def log_response(self, request_id: str, status_code: int,
                    headers: Dict[str, str], body: bytes,
                    duration_ms: Optional[float] = None,
                    is_streaming: bool = False):
        """Log HTTP response data."""
        body_content = None
        body_truncated = False
        
        if body:
            body_content, body_truncated = self._truncate_body(body)
        
        entry = {
            "timestamp": self._get_timestamp(),
            "type": "response",
            "request_id": request_id,
            "status_code": status_code,
            "body_size": len(body) if body else 0,
            "body_truncated": body_truncated,
            "is_streaming": is_streaming,
            "body": body_content
        }
        
        # Add duration if available
        if duration_ms is not None:
            entry["duration_ms"] = round(duration_ms, 2)
        
        # Add headers if configured
        if self.include_headers:
            entry["headers"] = self._redact_headers(headers)
        
        self._write_log(entry)
    
    def log_error(self, request_id: str, error_type: str, error_message: str):
        """Log error events."""
        entry = {
            "timestamp": self._get_timestamp(),
            "type": "error",
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message
        }
        
        self._write_log(entry)
    
    def _write_log(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to the JSONL file with rotation."""
        # Convert to single-line JSON (newlines are automatically escaped)
        json_line = json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + "\n"
        
        with self.lock:
            # Check if rotation needed
            if self.current_log_file.exists():
                current_size = self.current_log_file.stat().st_size
                if current_size >= self.max_file_size:
                    self._rotate()
            
            # Append to log file
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(json_line)
    
    def _rotate(self) -> None:
        """Rotate log files."""
        if not self.current_log_file.exists():
            return
        
        # Generate timestamp for rotated file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        rotated_file = self.log_dir / f"api_logs_{timestamp}.jsonl"
        
        # Rename current file
        self.current_log_file.rename(rotated_file)
        
        # Compress if enabled
        if self.compress_old_files:
            self._compress_file(rotated_file)
    
    def _compress_file(self, file_path: Path) -> None:
        """Compress a log file using gzip."""
        compressed_path = file_path.with_suffix(".jsonl.gz")
        
        with open(file_path, "rb") as f_in:
            with gzip.open(compressed_path, "wb", compresslevel=6) as f_out:
                # Stream copy to handle large files
                while chunk := f_in.read(1024 * 1024):  # 1MB chunks
                    f_out.write(chunk)
        
        # Remove uncompressed file
        file_path.unlink()
    
    def _cleanup_old_files(self) -> None:
        """Remove log files older than retention_days."""
        if self.retention_days <= 0:
            return
        
        cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
        
        # Get all log files (both compressed and uncompressed)
        for pattern in ["api_logs_*.jsonl", "api_logs_*.jsonl.gz"]:
            for log_file in self.log_dir.glob(pattern):
                try:
                    if log_file.stat().st_mtime < cutoff_time:
                        log_file.unlink()
                        print(f"Deleted old log file: {log_file}")
                except Exception as e:
                    print(f"Error deleting {log_file}: {e}")


# Create a global logger instance
logger = RotatingJSONLLogger()