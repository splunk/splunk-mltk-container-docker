#---------------------------------------------------------------------------------
# FastAPI routes and handlers
#---------------------------------------------------------------------------------

import os
import re
import math
import json
import time
import traceback
import aiohttp
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logfire
from typing import Optional, List, Dict, Any
from pydantic import ValidationError
from utils import safe_get

from models import QueryRequest, DataDescriptionEntry
from query_engine import find_anomalous_ip_addresses, find_suspicious_user_sessions, select_tool, generate_fallback_query
from vector_store import VectorStore, parse_query_example, parse_datastore_description

# Initialize FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize vector store
vector_store = None

@app.get("/")
async def home(request: Request):
    """Render the home page."""
    logfire.info("Home page accessed", client_ip=request.client.host)
    return templates.TemplateResponse("home.html", {
        "request": request,
        "timestamp": datetime.now().timestamp()
    })

#----------------------------------------------------------------------------
# curl http://localhost:2993/generate_query -d '{"query":"Show me all errors in the last day", "llm_provider":"ollama_3.1_70b", "q_samples":5}'
#   {"spl_query":"index=web_traffic earliest=-1d latest=now | where status>=400 | table _time src_ip status"}

# | makeresults
# | ai provider=ai2spl;model::ollama_3.3_70b;q_samples::5  prompt="Show unusually high downloads from server in last 2 weeks"
#
# ai_spl = "index=web_traffic ..."

@app.post("/generate_query")
async def generate_query(request: Request):
    """Generate a Splunk SPL query based on user's request using tool or fallback."""
    try:
        body = await request.json()
        query_request = QueryRequest(**body)
        user_query = query_request.query.replace("### User:", "").replace("### Response:", "").strip()

        logfire.info(f"===> Query: {user_query}",
                     query=user_query,
                     llm_provider=query_request.llm_provider)

        # Select the appropriate tool based on the user query
        tool_decision = await select_tool(user_query, query_request.llm_provider)
        selected_tool = tool_decision.tool_name
        earliest = tool_decision.earliest
        latest = tool_decision.latest

        if selected_tool in ["find_anomalous_ip_addresses", "find_suspicious_user_sessions"]:
            # DIRECTLY EXECUTE THE TOOL WITHOUT LLM INVOLVEMENT
            if selected_tool == "find_anomalous_ip_addresses":
                tool_result = find_anomalous_ip_addresses(earliest=earliest, latest=latest)
            else:
                tool_result = find_suspicious_user_sessions(earliest=earliest, latest=latest)
            
            return JSONResponse(content={
                "spl_query": tool_result.spl_query,
                "viz_type": tool_result.viz_type,
                "tool_used": selected_tool,
                "tool_selection_reason": tool_decision.reason,
                "earliest": earliest,
                "latest": latest
            })

        # # 1. Find N top matching query samples to user query
        # initial_query_samples = vector_store.similarity_search_with_score(
        #     user_query,
        #     k=3,  # Always get top N matches
        #     filter={"entry_type": "query_example"}
        # )

        # # Extract SPL query fragments from initial samples
        query_samples_fragments = []
        # for doc, _ in initial_query_samples:
        #     if "SPL answer:" in doc["page_content"]:
        #         fragment = doc["page_content"].split("SPL answer:", 1)[-1].strip()
        #     else:
        #         fragment = doc["page_content"].strip()
        #     query_samples_fragments.append(fragment)

        # 2. Find top matching dataset using user query + SPL fragments
        query_for_matching = user_query + "\n" + "\n".join(query_samples_fragments)
        matching_datasets = vector_store.similarity_search_with_score(
            query_for_matching,
            k=2,
            filter={"entry_type": "data_description"}
        )
        
        # Log matching datasets and their scores
        for idx, (dataset, score) in enumerate(matching_datasets):
            logfire.info(f"Matching dataset {idx+1}",
                        dataset_name=dataset['metadata'].get('data_store_name', 'Unknown'),
                        data_store=dataset['metadata'].get('data_store', 'Unknown'),
                        match_score=score)

        # 3. Find more query samples using user query + dataset metadata
        dataset_metadata = ""
        dataset_name = ""
        if matching_datasets:
            dataset = matching_datasets[0][0]  # Get the top matching dataset
            dataset_type = dataset['metadata'].get('data_store', '')
            dataset_name = dataset['metadata'].get('data_store_name', '')
            # dataset_metadata = (
            #     f"{dataset['metadata'].get('data_store', '')}={dataset_name}\n"
            #     f"Description: {dataset['metadata'].get('description', '')}\n"
            #     f"{dataset['page_content']}"
            # )

        # Get additional query samples using combined context
        final_query_samples = vector_store.similarity_search_with_score(
            f"{user_query} (use: {dataset_type} = {dataset_name})",
            k=int(query_request.q_samples) * 2, # retrieve more as further filtering may reduce number of elements
            filter={"entry_type": "query_example"}
        )

        # Filter query samples to only include those containing the dataset name in SPL
        filtered_samples = []
        for doc, score in final_query_samples:
            # Extract SPL part
            if "SPL answer:" in doc["page_content"]:
                spl_part = doc["page_content"].split("SPL answer:", 1)[1].strip()
            else:
                spl_part = doc["page_content"].strip()
            
            # Only include if SPL contains dataset name
            if dataset_name.lower() in spl_part.lower():
                filtered_samples.append((doc, score))
        final_query_samples = filtered_samples
        
        # Use different number of samples based on data source
        samples_limit = 5 if dataset_name.lower() == "fraud_cms" else int(query_request.q_samples)
        final_query_samples = final_query_samples[:samples_limit]

        # 4. Prepare query samples for context
        docs__query_examples_list = '\n\n'.join([
            '\n'.join(line for line in doc["page_content"].split('\n') 
                     if not line.lower().startswith('explanation:'))
            for doc, _ in final_query_samples
        ])

        try:
            data_description1 = f"Data_Store_Type: {matching_datasets[0][0]['metadata']['data_store']}\nData_Store_Name: {matching_datasets[0][0]['metadata']['data_store_name']}"
        except Exception:
            data_description1 = ""

        try:
            data_description2 = ""
            if matching_datasets[0][0]['metadata'].get('description'):
                data_description2 = f"\nData_Store_Description: \n{matching_datasets[0][0]['metadata']['description']}"
        except Exception:
            data_description2 = ""

        try:
            data_fields = "\nData_Store_Content:\n" + matching_datasets[0][0]["page_content"].split("Data_Store_Content:")[-1].strip()
        except Exception:
            data_fields = ""

        data_source_summary = (data_description1 + data_description2).strip()

        # NOTE: for some reason healthcare use cases arent working well when data description is present  - possibly because of lots of extra text.
        #       Although it DOES work when issuing raw |ai ... prompt. Problem with API. Need to investigate.
        #       However quite often data description - context - exact values of fields, their meaning is often needed to RAG into the prompt.
        # Need to investigate.
        #
        
        # Check if the data source is fraud_cms to use simplified prompt
        # if matching_datasets and matching_datasets[0][0]['metadata'].get('data_store_name') == "fraud_cms":
        if True or safe_get(matching_datasets, [0, 0, 'metadata', 'data_store_name'], None) == "fraud_cms":
            # Simplified prompt for fraud_cms data source
            fallback_context_prompt = f"""
User Request: {user_query}

Relevant Information:

Query examples:

{docs__query_examples_list}

Based on the user request and the relevant information provided, generate a valid Splunk SPL query. Just reply with the query, no explanation needed.
"""
        else:
            # Original prompt for all other data sources
            fallback_context_prompt = f"""
User Request: {user_query}

Relevant Information:

Following data source is available:
{data_source_summary}
{data_fields}

Query examples:

{docs__query_examples_list}

Time extraction rules:
- Convert relative times ("last hour") to earliest="-1h"
- Absolute times must use Splunk format in double quotes (e.g., "02/17/2021:00:00:00")
- For date-only requests (e.g. "December 17, 2024"):
    - Set earliest to "MM/DD/YYYY:00:00:00"
    - Set latest to "MM/DD/YYYY:23:59:59"
- Default: earliest="-1d", latest="now"

Based on the user request and the relevant information provided, generate a valid Splunk SPL query. Just reply with the query, no explanation needed.
"""

        # Generate fallback query
        fallback_result = await generate_fallback_query(
            user_query, 
            query_request.llm_provider, 
            fallback_context_prompt
        )

        return JSONResponse(content={
            "spl_query": fallback_result.spl_query,
            "viz_type": fallback_result.viz_type,
            "tool_used": "general",
            "tool_selection_reason": "No tool matched",
            "earliest": earliest,
            "latest": latest,
            # Provide the fallback system prompt and the user query
            "system_prompt": "",
            "llm_query": fallback_context_prompt.strip()
        })

    except ValidationError as ve:
        logfire.error("Validation error in generate_query",
                     error=str(ve),
                     error_type="ValidationError")
        print(f"ValidationError in generate_query: {str(ve)}")
        return JSONResponse(status_code=422, content={"error": str(ve)})
    except Exception as e:
        error_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logfire.error("Unexpected error in generate_query",
                     error=str(e),
                     error_type=type(e).__name__,
                     stack_trace=error_trace)
        print(f"Exception in generate_query: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Exception: {str(e)}"})
#----------------------------------------------------------------------------

@app.get("/manage")
async def manage(
    request: Request,
    message: str = Query(None),
    page: int = Query(1, ge=1),
    search: str = Query(None)
):
    """Render the management page with pagination and search."""
    page_size = 10
    total_entries = await get_total_entries_count(search)
    total_pages = math.ceil(total_entries / page_size)
    entries = await get_paginated_entries(page, page_size, search)
    return templates.TemplateResponse("manage.html", {
        "request": request,
        "entries": entries,
        "message": message,
        "current_page": page,
        "total_pages": total_pages,
        "search": search
    })

@app.post("/add_content")
async def add_content(
    upload_type: str = Form(...),
    content: str = Form(...),
    data_store: Optional[str] = Form(None),
    data_store_name: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """Add new content or update existing entries in the vector store."""
    try:
        documents_to_upsert = []

        if upload_type == "data_description":
            # Validate the input using DataDescriptionEntry
            entry = DataDescriptionEntry(
                data_store=data_store,
                data_store_name=data_store_name,
                description=description,
                content=content
            )
            
            # Create document directly from the validated entry
            documents_to_upsert.append({
                "page_content": entry.content.strip(),
                "metadata": {
                    "entry_type": "data_description",
                    "data_store": entry.data_store,
                    "data_store_name": entry.data_store_name,
                    "description": entry.description,
                    "id": f"{entry.data_store}_{entry.data_store_name}"
                }
            })

        elif upload_type == "query_example":
            documents_to_upsert.extend(parse_query_example(content))
        else:
            raise ValueError(f"Invalid upload type: {upload_type}")
        
        if documents_to_upsert:
            vector_store.add_documents(
                documents_to_upsert,
                ids=[doc["metadata"]["id"] for doc in documents_to_upsert]
            )
        
        message = f"Successfully processed content: {len(documents_to_upsert)} entries added or updated."
        return JSONResponse(content={"message": message, "redirect": "/manage"})

    except ValidationError as ve:
        return JSONResponse(
            status_code=422,
            content={"message": f"Validation error: {str(ve)}"}
        )
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={"message": str(ve)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An unexpected error occurred: {str(e)}"}
        )

async def get_total_entries_count(search: str = None) -> int:
    """Get the total count of entries, optionally filtered by search term."""
    try:
        collection = vector_store.get()
        if search:
            total_count = 0
            for entry_type in ["data_description", "query_example"]:
                results = vector_store.similarity_search_with_score(
                    search,
                    k=10000,
                    filter={"entry_type": entry_type}
                )
                total_count += len(results)
            return total_count
        else:
            total_count = 0
            for entry_type in collection['ids']:
                total_count += len(collection['ids'][entry_type])
            return total_count
    except Exception as e:
        print(f"Error in get_total_entries_count: {str(e)}")
        return 0

async def get_paginated_entries(page: int, page_size: int, search: str = None) -> List[Dict[str, Any]]:
    """Get a paginated list of entries, optionally filtered by search term."""
    try:
        start_index = (page - 1) * page_size
        end_index = start_index + page_size

        if search:
            all_results = []
            for entry_type in ["data_description", "query_example"]:
                results = vector_store.similarity_search_with_score(
                    search,
                    k=10000,
                    filter={"entry_type": entry_type}
                )
                all_results.extend(results)
            all_results.sort(key=lambda x: x[1], reverse=True)
            paginated_results = all_results[start_index:end_index]
            entries = []
            for doc, score in paginated_results:
                entries.append({
                    'id': doc['metadata'].get('id', 'Unknown'),
                    'type': doc['metadata'].get('entry_type', 'Unknown'),
                    'content': doc['page_content'],
                    'metadata': doc['metadata']
                })
        else:
            collection = vector_store.get()
            all_entries = []
            for entry_type in collection['ids']:
                for idx, doc_id in enumerate(collection['ids'][entry_type]):
                    all_entries.append({
                        'id': doc_id,
                        'type': entry_type,
                        'content': collection['documents'][entry_type][idx],
                        'metadata': collection['metadatas'][entry_type][idx]
                    })
            entries = all_entries[start_index:end_index]

        return entries
    except Exception as e:
        print(f"Error in get_paginated_entries: {str(e)}")
        return []

@app.get("/static/{filename:path}")
async def serve_static(filename: str):
    response = FileResponse(f"static/{filename}")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.delete("/remove_entry/{entry_id}")
async def remove_entry(entry_id: str):
    """Remove an entry from the vector database."""
    try:
        vector_store.delete([entry_id])
        return {"message": f"Entry {entry_id} removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while removing the entry: {str(e)}")

@app.get("/list_entries")
async def list_entries():
    """List all entries in the vector database."""
    try:
        collection = vector_store.get()
        entries = []
        for entry_type in collection['ids']:
            for i, doc_id in enumerate(collection['ids'][entry_type]):
                entries.append({
                    'id': doc_id,
                    'content': collection['documents'][entry_type][i],
                    'metadata': collection['metadatas'][entry_type][i],
                })
        return {"entries": entries}
    except Exception as e:
        print(f"Error in list_entries: {str(e)}")
        return {"entries": [], "message": f"An error occurred while fetching entries: {str(e)}"}

# Add a decorator to support both /api/tags and //api/tags
@app.route("/api/tags", methods=["GET", "POST"])
@app.route("//api/tags", methods=["GET", "POST"])
async def tags(request: Request):
    """Forward /api/tags requests to Ollama server"""
    try:
        # Get the base URL from the Ollama config
        ollama_base_url = "http://localhost:11438"
        
        # Forward the request to Ollama
        async with aiohttp.ClientSession() as session:
            # Forward the request with same method and body
            async with session.request(
                method=request.method,
                url=f"{ollama_base_url}/api/tags",
                headers={"Content-Type": "application/json"},
                data=await request.body() if request.method == "POST" else None
            ) as response:
                # Get response data
                response_data = await response.json()
                
                # Return response with same status code
                return JSONResponse(
                    content=response_data,
                    status_code=response.status
                )
                
    except Exception as e:
        logfire.error("Error forwarding request to Ollama tags API", 
                     error=str(e),
                     error_type=type(e).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to forward request: {str(e)}"}
        )

# Add this new endpoint after your other endpoints
# curl http://localhost:2993/api/generate -d '{"model": "llama3.1:8b-instruct-q8_0", "prompt": "What top most active IP addresses in the last hour?", "stream":false}'
@app.post("/api/generate")
async def api_generate(request: Request):
    """Handle generation requests in Ollama-compatible format"""
    try:
        # Get start time for duration tracking
        start_time = time.time()
        
        # Parse request body
        body = await request.json()
        model = body.get("model")

        # Make best effort to retrieve user prompt
        prompt = safe_get(body, ['prompt'], None)
        if not prompt:
            messages = safe_get(body, ['messages'], [])
            for msg_dict in messages:
                if safe_get(msg_dict, ['role'], None) == 'user':
                    prompt = safe_get(msg_dict, ['content'], None)
                    break
        
        if not model or not prompt:
            raise HTTPException(
                status_code=400,
                content={"error": "Missing required fields 'model' or 'prompt'"}
            )
        
        # Map the model name to our provider format
        # Simply use the model name as the provider
        # If it contains '/', it will be routed to LiteLLM
        # Otherwise, it will be routed to Ollama
        provider = model
        
        # Create query request
        query_request = QueryRequest(
            query=prompt,
            llm_provider=provider,
            q_samples=10
        )
        
        # Build a fake receive function to simulate the request body for generate_query
        async def fake_receive():
            return {"type": "http.request", "body": json.dumps(query_request.model_dump()).encode("utf-8")}
        fake_scope = {"type": "http", "method": "POST", "path": "/generate_query", "headers": []}
        fake_request = Request(fake_scope, receive=fake_receive)
        response = await generate_query(fake_request)
        
        # Calculate duration
        total_duration = int((time.time() - start_time) * 1000000000)  # Convert to nanoseconds
        
        # Format response in Ollama-compatible format
        response_body = json.loads(response.body.decode("utf-8"))
        # <|spl_query>...<|viz_type><|>
        spl_query = safe_get(response_body, ['spl_query'], "| makeresults | eval info=\"No SPL query generated\"")
        viz_type = safe_get(response_body, ['viz_type'], "table")
        response_body = f"<|spl_query>{spl_query}<|viz_type>{viz_type}<|>"
        # response_body.pop('system_prompt', None)
        # response_body.pop('llm_query', None)
        # response_body = json.dumps(response_body)
        formatted_response = {
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": response_body,
            "done": True,
            "done_reason": "stop",
            "context": [0],
            "total_duration": total_duration,
            "load_duration": 0,
            "prompt_eval_count": 20,
            "prompt_eval_duration": total_duration // 2,
            "eval_count": 200,
            "eval_duration": total_duration // 2
        }
        formatted_response['message'] = {'role':'assistant', 'content':response_body}
        formatted_response['choices'] = [{"message":{"content":response_body}}]
        formatted_response['content'] = [{"text":response_body}]
        
        return JSONResponse(content=formatted_response)
        
    except ValidationError as ve:
        logfire.error("Validation error in api_generate",
                     error=str(ve),
                     error_type="ValidationError")
        return JSONResponse(
            status_code=422,
            content={"error": f"Validation error: {str(ve)}"}
        )
    except Exception as e:
        error_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logfire.error("Unexpected error in api_generate",
                     error=str(e),
                     error_type=type(e).__name__,
                     stack_trace=error_trace)
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )

# Add new /api/chat endpoint using the same logic as /api/generate
@app.post("/api/chat")
async def api_chat(request: Request):
    # Reuse the same logic as the /api/generate endpoint
    return await api_generate(request)
