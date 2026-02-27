#---------------------------------------------------------------------------------
# Query generation logic and tool functions
#---------------------------------------------------------------------------------

from typing import Dict, Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
import logfire

from models import (
    ToolSelector, AnomalousIPsResult, SuspiciousUserSessionsResult, 
    GeneratedSPLQuery
)
from constants import g_IP_Anomaly_Detection_SPL, g_User_Session_Anomaly_Detection_SPL

#----------------------------------------------------------------------------
# Function tools definitions
def find_anomalous_ip_addresses(earliest: str = "-1d", latest: str = "now") -> AnomalousIPsResult:
    """Find anomalously or suspiciously behaving IP addresses using multiple behavior characteristics and unsupervised machine learning algorithms.
    
    Args:
        earliest: Start time for the query in Splunk format (e.g. "-1d", "-4h")
        latest: End time for the query in Splunk format (e.g. "now", "+1d")
    """
    spl_query = g_IP_Anomaly_Detection_SPL.replace("{{TIMEFRAME}}", f"earliest={earliest} latest={latest}")
    return AnomalousIPsResult(spl_query=spl_query.strip(), viz_type="3D")
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def find_suspicious_user_sessions(earliest: str = "-1d", latest: str = "now") -> SuspiciousUserSessionsResult:
    """Find user sessions and users exhibiting suspicious behavior patterns that may signify account takeover attacks.
    
    Args:
        earliest: Start time for the query in Splunk format (e.g. "-1d", "-4h")
        latest: End time for the query in Splunk format (e.g. "now", "+1d")
    """
    spl_query = g_User_Session_Anomaly_Detection_SPL.replace("{{TIMEFRAME}}", f"earliest={earliest} latest={latest}")
    return SuspiciousUserSessionsResult(spl_query=spl_query.strip(), viz_type="table")
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def get_model_for_provider(provider_key: str):
    """Create and return a PydanticAI model (OpenAIModel with appropriate provider) based on the user's selection."""
    import os
    
    # Check if provider contains '/' - indicates LiteLLM routing
    if '/' in provider_key:
        # Use LiteLLM proxy
        raw_logging_port = os.getenv('RAW_LOGGING_PORT', '7011')
        litellm_master_key = os.getenv('LITELLM_MASTER_KEY', 'sk-21056a9b6f5910fa')
        
        return OpenAIModel(
            provider_key,  # Pass the full provider/model string
            provider=OpenAIProvider(
                base_url=f"http://localhost:{raw_logging_port}/v1",
                api_key=litellm_master_key
            )
        )
    else:
        # Direct Ollama model (e.g., "llama3.3:70b-instruct-q5_K_M")
        # Route to Ollama using configurable base URL
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11438/v1')
        
        return OpenAIModel(
            provider_key,  # Use the provider_key as the model name
            provider=OpenAIProvider(
                base_url=ollama_base_url
            )
        )
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
async def select_tool(user_query: str, llm_provider: str) -> ToolSelector:
    """Select the appropriate tool based on the user query."""
    model = get_model_for_provider(llm_provider)
    
    # Use default temperature for all providers
    temperature = 0.0
    model_settings = ModelSettings(temperature=temperature)

    # Create tool selection agent with more specific guidance
    tool_selection_agent = Agent(
        model,
        result_type=ToolSelector,
        system_prompt="""You are a tool selector for Splunk query generation. Based on the user's query, 
        select the most appropriate tool or choose 'general' for custom query generation.
        
        Available tools:
        1. find_anomalous_ip_addresses: for detecting anomalously behaving IP addresses using unsupervised machine learning.
           
        2. find_suspicious_user_sessions: ONLY for detecting general anomalous user behavior patterns that may indicate account takeover. 
           This tool should ONLY be selected when the request is specifically about detecting suspicious patterns or anomalies in user sessions.
           
           Examples where this tool should be selected:
           - "Find suspicious user session patterns that might indicate account takeover"
           - "Detect anomalous user behavior patterns in sessions"
           - "Identify potentially compromised accounts based on session behavior"
           
           Examples where this tool should NOT be selected (use 'general' instead):
           - "Show users logged in from multiple countries"
           - "Find failed login attempts by country"
           - "List user accounts with successful logins"
           - Any specific query about login patterns, countries, or user activities that isn't explicitly about anomaly detection
        
        3. general: For ALL other requests, including specific queries about user logins, authentication patterns, or any other data analysis needs.
           
        Time extraction rules:
        - Convert relative times ("last hour") to earliest="-1h"
        - Absolute times must use Splunk format in double quotes (e.g., "02/17/2021:00:00:00")
        - For date-only requests (e.g. "December 17, 2024"):
          - Set earliest to "MM/DD/YYYY:00:00:00"
          - Set latest to "MM/DD/YYYY:23:59:59"
        - Default: earliest="-1d", latest="now"  """,
        retries=3,
        model_settings=model_settings
    )

    # Get tool selection decision
    tool_decision = await tool_selection_agent.run(
        user_query, 
        model_settings=ModelSettings(temperature=temperature)
    )
    
    return tool_decision.data
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
async def generate_fallback_query(user_query: str, llm_provider: str, context_prompt: str) -> GeneratedSPLQuery:
    """Generate a fallback query when no specific tool matches."""
    system_prompt = "You are a Splunk SPL query generator. Given the context of available data sources and query samples produce a valid SPL query that matches the user's request."
    fallback_model = get_model_for_provider(llm_provider)
    
    # Use default temperature for all providers
    temperature = 0.0
    
    fallback_agent = Agent(
        fallback_model,
        result_type=GeneratedSPLQuery,
        system_prompt=system_prompt,
        retries=8,
        model_settings=ModelSettings(temperature=temperature)
    )

    with logfire.span("fallback_generation_prompt") as span:
        fallback_result = await fallback_agent.run(context_prompt)

    return fallback_result.data
#----------------------------------------------------------------------------
