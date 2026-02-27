import os
import json
import time
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from logger import logger  # Import the global logger instance

app = FastAPI(title="Raw Logger Proxy")

# Configuration from environment variables
LITELLM_HOST = os.getenv("LITELLM_HOST", "localhost")
LITELLM_PORT = int(os.getenv("LITELLM_PORT", "4000"))
LITELLM_URL = f"http://{LITELLM_HOST}:{LITELLM_PORT}"
PROXY_PORT = int(os.getenv("PROXY_PORT", "4001"))

# Create async HTTP client
client = httpx.AsyncClient(timeout=300.0, limits=httpx.Limits(max_keepalive_connections=5))

async def stream_response(response: httpx.Response, request_id: str, start_time: float) -> AsyncGenerator[bytes, None]:
    """Stream response chunks and log them."""
    chunks = []
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        yield chunk
    
    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Log complete streamed response
    complete_response = b''.join(chunks)
    logger.log_response(
        request_id=request_id,
        status_code=response.status_code,
        headers=dict(response.headers),
        body=complete_response,
        duration_ms=duration_ms,
        is_streaming=True
    )

@app.get("/about")
async def about():
    """Health check and info endpoint."""
    return {
        "service": "Raw Logger Proxy",
        "version": "2.0.0",
        "status": "healthy",
        "description": "Transparent proxy for LiteLLM with JSONL request/response logging",
        "endpoints": {
            "proxy": f"http://localhost:{PROXY_PORT}/",
            "target": LITELLM_URL,
            "logs_dir": str(logger.log_dir),
            "log_file": str(logger.current_log_file)
        },
        "configuration": {
            "max_file_size_mb": logger.max_file_size_mb,
            "retention_days": logger.retention_days,
            "compress_old_files": logger.compress_old_files,
            "include_headers": logger.include_headers,
            "redact_sensitive": logger.redact_sensitive
        }
    }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(request: Request, path: str):
    """Transparent proxy for all HTTP methods."""
    start_time = time.time()
    request_id = f"{start_time}_{request.client.host}"
    
    # Read request body
    body = await request.body()
    
    # Log incoming request
    logger.log_request(
        request_id=request_id,
        method=request.method,
        path=path,
        headers=dict(request.headers),
        body=body,
        query_params=dict(request.query_params),
        client_ip=request.client.host
    )
    
    # Prepare forwarded request
    url = f"{LITELLM_URL}/{path}"
    
    # Forward headers, excluding host
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        # Make request to LiteLLM
        response = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            params=request.query_params,
            follow_redirects=True
        )
        
        # Check if response is streaming
        is_streaming = "text/event-stream" in response.headers.get("content-type", "")
        
        if is_streaming:
            # Return streaming response
            return StreamingResponse(
                stream_response(response, request_id, start_time),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type")
            )
        else:
            # Read full response
            response_body = await response.aread()
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.log_response(
                request_id=request_id,
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response_body,
                duration_ms=duration_ms,
                is_streaming=False
            )
            
            # Return response
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
    except httpx.TimeoutException:
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log error
        logger.log_error(
            request_id=request_id,
            error_type="TimeoutException",
            error_message="Request to LiteLLM timed out"
        )
        
        error_response = {"error": "Request to LiteLLM timed out"}
        logger.log_response(
            request_id=request_id,
            status_code=504,
            headers={"content-type": "application/json"},
            body=json.dumps(error_response).encode(),
            duration_ms=duration_ms,
            is_streaming=False
        )
        return Response(
            content=json.dumps(error_response),
            status_code=504,
            media_type="application/json"
        )
    except Exception as e:
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log error
        logger.log_error(
            request_id=request_id,
            error_type=type(e).__name__,
            error_message=str(e)
        )
        
        error_response = {"error": f"Proxy error: {str(e)}"}
        logger.log_response(
            request_id=request_id,
            status_code=500,
            headers={"content-type": "application/json"},
            body=json.dumps(error_response).encode(),
            duration_ms=duration_ms,
            is_streaming=False
        )
        return Response(
            content=json.dumps(error_response),
            status_code=500,
            media_type="application/json"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "proxy_target": LITELLM_URL}

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    print(f"Raw Logger Proxy starting on port {PROXY_PORT}")
    print(f"Forwarding requests to: {LITELLM_URL}")
    print(f"Log directory: {logger.log_dir}")
    print(f"Current log file: {logger.current_log_file}")
    print(f"Max file size: {logger.max_file_size_mb}MB")
    print(f"Retention: {logger.retention_days} days")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources."""
    await client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)