#---------------------------------------------------------------------------------
# Main application entry point
#---------------------------------------------------------------------------------

import os
import sys
import uvicorn
import logfire
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from api import app
from vector_store import load_vector_db_content, test_faiss_gpu, VectorStore

# Configure logfire
logfire.configure()
logfire.instrument_pydantic()  # For Pydantic model validation tracking
logfire.instrument_openai()    # For OpenAI API calls
# logfire.instrument_asyncpg()   # For any async operations - commented out as asyncpg is not used

# Load environment variables from root .env
load_dotenv()

# Load LiteLLM environment variables
litellm_env_file = os.path.join(os.path.dirname(__file__), 'LiteLLM', '.env')
if os.path.exists(litellm_env_file):
    load_dotenv(litellm_env_file)

# Test if FAISS is using GPU
gpu_available = test_faiss_gpu()
print(f"GPU Available: {gpu_available}")

# Initialize vector store
vector_store = None

def initialize_app():
    """Initialize the application and load vector DB content."""
    global vector_store
    
    # Check for loaddata parameter
    if len(sys.argv) > 1 and sys.argv[1] == 'loaddata':
        print("Loading vector DB content from files...")
        vector_store = load_vector_db_content()
        print("Loading complete!")
    else:
        # Initialize empty vector store
        vector_store = VectorStore(embedding_model="local")
    
    # Make vector_store available to the API module
    import api
    api.vector_store = vector_store
    
    return app

if __name__ == "__main__":
    app = initialize_app()
    uvicorn.run(app, host="0.0.0.0", port=2993)
