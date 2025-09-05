"""Splunk REST API Client for async operations."""

import httpx
import json
from typing import Optional, Dict, Any, List
import xml.etree.ElementTree as ET


class SplunkAPIError(Exception):
    """Custom exception for Splunk API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class SplunkClient:
    """Async client for Splunk REST API operations."""
    
    def __init__(self, config: dict):
        """Initialize Splunk client with configuration.
        
        Args:
            config: Dictionary containing:
                - splunk_host: Splunk server hostname
                - splunk_port: Splunk management port (default: 8089)
                - splunk_username: Username for basic auth (optional)
                - splunk_password: Password for basic auth (optional)
                - splunk_token: Token for token auth (optional)
                - verify_ssl: Whether to verify SSL certificates
        """
        self.config = config
        self.base_url = f"https://{config['splunk_host']}:{config['splunk_port']}"
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        
    async def connect(self):
        """Create and configure the HTTP client."""
        # Setup authentication - prefer token over basic auth
        auth = None
        headers = {}
        
        if self.config.get("splunk_token"):
            headers["Authorization"] = f"Splunk {self.config['splunk_token']}"
        elif self.config.get("splunk_username") and self.config.get("splunk_password"):
            auth = httpx.BasicAuth(self.config["splunk_username"], self.config["splunk_password"])
        else:
            raise SplunkAPIError("No valid authentication configured. Set either SPLUNK_TOKEN or SPLUNK_USERNAME/SPLUNK_PASSWORD.")
        
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=auth,
            headers=headers,
            verify=self.config.get("verify_ssl", False),
            timeout=30.0
        )
        
    async def disconnect(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            
    def _ensure_connected(self):
        """Ensure client is connected."""
        if not self._client:
            raise SplunkAPIError("Client not connected. Call connect() first or use async context manager.")
            
    def _parse_response(self, response_text: str, output_mode: str = "json") -> List[Dict[str, Any]]:
        """Parse Splunk response based on output mode."""
        if output_mode == "json":
            try:
                # Try to parse as a single JSON object first (oneshot format)
                data = json.loads(response_text)
                if "results" in data:
                    return data["results"]
                elif "result" in data:
                    return [data["result"]]
            except json.JSONDecodeError:
                # Fall back to line-by-line parsing (export format)
                events = []
                for line in response_text.strip().split('\n'):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "result" in data:
                                events.append(data["result"])
                            elif "results" in data:
                                events.extend(data["results"])
                        except json.JSONDecodeError:
                            continue
                return events
        else:
            # Simple XML parsing for other formats
            events = []
            try:
                root = ET.fromstring(response_text)
                for result in root.findall(".//result"):
                    event = {}
                    for field in result.findall("field"):
                        key = field.get("k")
                        value = field.find("value/text").text if field.find("value/text") is not None else ""
                        event[key] = value
                    events.append(event)
            except ET.ParseError:
                pass
            return events
            
    async def search_oneshot(self, query: str, earliest_time: str = "-24h", 
                           latest_time: str = "now", max_count: int = 100) -> List[Dict[str, Any]]:
        """Execute a oneshot search and return results immediately.
        
        Args:
            query: SPL search query
            earliest_time: Start time for search
            latest_time: End time for search  
            max_count: Maximum number of results
            
        Returns:
            List of event dictionaries
        """
        self._ensure_connected()
        
        # Don't prepend "search" if query starts with a pipe (|)
        if query.strip().startswith("|"):
            search_query = query
        else:
            search_query = f"search {query}"
            
        params = {
            "search": search_query,
            "earliest_time": earliest_time,
            "latest_time": latest_time,
            "count": max_count,
            "output_mode": "json"
        }
        
        try:
            response = await self._client.post("/services/search/jobs/oneshot", data=params)
            response.raise_for_status()
            return self._parse_response(response.text, "json")
        except httpx.HTTPStatusError as e:
            raise SplunkAPIError(f"Search failed", status_code=e.response.status_code, 
                               details={"error": e.response.text})
        except Exception as e:
            raise SplunkAPIError(f"Search failed: {str(e)}")
            
    async def search_export(self, query: str, earliest_time: str = "-24h",
                          latest_time: str = "now", max_count: int = 100) -> List[Dict[str, Any]]:
        """Execute an export search that streams results.
        
        Args:
            query: SPL search query
            earliest_time: Start time for search
            latest_time: End time for search
            max_count: Maximum number of results
            
        Returns:
            List of event dictionaries
        """
        self._ensure_connected()
        
        # Don't prepend "search" if query starts with a pipe (|)
        if query.strip().startswith("|"):
            search_query = query
        else:
            search_query = f"search {query}"
            
        params = {
            "search": search_query,
            "earliest_time": earliest_time,
            "latest_time": latest_time,
            "count": max_count,
            "output_mode": "json",
            "search_mode": "normal"
        }
        
        try:
            response = await self._client.post("/services/search/jobs/export", data=params)
            response.raise_for_status()
            events = self._parse_response(response.text, "json")
            
            # Limit results if needed
            if max_count > 0:
                return events[:max_count]
            return events
        except httpx.HTTPStatusError as e:
            raise SplunkAPIError(f"Export search failed", status_code=e.response.status_code,
                               details={"error": e.response.text})
        except Exception as e:
            raise SplunkAPIError(f"Export search failed: {str(e)}")
            
    async def get_indexes(self) -> List[Dict[str, Any]]:
        """Get list of all indexes with detailed information.
        
        Returns:
            List of index dictionaries with properties
        """
        self._ensure_connected()
        
        try:
            response = await self._client.get("/services/data/indexes", params={"output_mode": "json"})
            response.raise_for_status()
            
            data = response.json()
            indexes = []
            
            for entry in data.get("entry", []):
                content = entry.get("content", {})
                indexes.append({
                    "name": entry.get("name", ""),
                    "datatype": content.get("datatype", "event"),
                    "totalEventCount": int(content.get("totalEventCount", "0")),
                    "currentDBSizeMB": float(content.get("currentDBSizeMB", "0")),
                    "maxDataSize": content.get("maxDataSize", "auto"),
                    "maxTotalDataSizeMB": content.get("maxTotalDataSizeMB", "unknown"),
                    "minTime": content.get("minTime", ""),
                    "maxTime": content.get("maxTime", ""),
                    "disabled": content.get("disabled", False),
                    "frozenTimePeriodInSecs": content.get("frozenTimePeriodInSecs", "")
                })
            
            return indexes
        except httpx.HTTPStatusError as e:
            raise SplunkAPIError(f"Failed to get indexes", status_code=e.response.status_code,
                               details={"error": e.response.text})
        except Exception as e:
            raise SplunkAPIError(f"Failed to get indexes: {str(e)}")
            
