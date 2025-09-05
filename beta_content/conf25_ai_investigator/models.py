#---------------------------------------------------------------------------------
# Pydantic models for data validation and configuration
#---------------------------------------------------------------------------------

import os
import re
from typing import List, Dict, Any, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ValidationError, field_validator

#----------------------------------------------------------------------------
class DataDescriptionEntry(BaseModel):
    data_store: str
    data_store_name: str
    description: Optional[str] = None
    content: str
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
class QueryExampleEntry(BaseModel):
    content: str
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
class GeneratedSPLQuery(BaseModel):
    """Model for generated SPL queries"""
    spl_query: str = Field(
        ...,
        description=(
            "A valid Splunk SPL (Search Processing Language) query string. "
            "The query should follow Splunk's syntax and include proper "
            "keywords, fields, and clauses. Examples include search commands "
            "with filters, data aggregation, or transformation steps."
        )
    )
    viz_type: str = Field(default="table", description="Recommended visualization type")

    @field_validator('spl_query')
    def validate_spl_query(cls, v, info):
        # Check for single quotes
        if "'" in v:
            raise ValueError("SPL query must not contain single quotes - use double quotes instead")
            
        # Basic validation logic
        if not (v.startswith('index=') or v.startswith('| tstats') or v.startswith('| from datamodel')):
            raise ValueError("SPL query must start with either 'index=' or '| tstats' or '| from datamodel'")
        
        if len(v) < 50:
            raise ValueError("SPL query must be at least 50 characters long")
        
        # For queries starting with '| tstats', ensure there is at least one additional '|' character.
        if v.strip().startswith('| tstats') and v.count('|') < 2:
            raise ValueError("SPL query starting with '| tstats' must contain at least one additional '|' character")
        
        # Add validation for tstats requiring datamodel
        if v.strip().startswith('| tstats'):
            # Find the position of the second pipe or end of string
            first_part = v.split('|', 2)[1] if len(v.split('|', 2)) > 1 else v
            if 'from datamodel=' not in first_part.lower():
                raise ValueError("tstats queries must include 'from datamodel=' before the next pipe character")
                
            # Check for illegal math expressions in tstats WHERE clause
            def contains_math_expression(spl_query):
                # Step 1: Match 'from' ... 'where' ... 'by' (whole words, no '|' between, across multiple lines)
                pattern = re.compile(
                    r'\bfrom\b[^\|]*?\bwhere\b([^\|]*?)\bby\b',
                    re.IGNORECASE | re.DOTALL  # DOTALL: '.' matches across newlines
                )
                
                match = pattern.search(spl_query)
                if not match:
                    return False, None, None

                # Step 2: Extract content between WHERE and BY
                between_where_by = match.group(1).strip()

                # Step 3: Check for simple math expression: digit followed or preceded by + - * /
                # math_expr_pattern = re.compile(r'(\d\s*[\+\-\*/]\s*\w+|\w+\s*[\+\-\*/]\s*\d)')
                # Eliminated '/' check temporarily to accomodate for latest="31/12/2024:23:59:59"
                math_expr_pattern = re.compile(r'(\d\s*[\+\-\*]\s*\w+|\w+\s*[\+\-\*]\s*\d)')
                math_found = bool(math_expr_pattern.search(between_where_by))

                return math_found, between_where_by, match.group(0)
            
            math_found, where_clause, _ = contains_math_expression(v)
            if math_found:
                raise ValueError(f"'WHERE' clause within '| tstats' queries cannot contain numerical math expressions. Found in: {where_clause}. Move them outside, after 'rename' operator.")
                
        head_match = re.search(r'\|\s*head\s+(\d+)', v)
        if head_match:
            head_count = int(head_match.group(1))
            if head_count < 25 and False:
                raise ValueError("When using '| head NNN', modify NNN to be at least 25 or higher")
        
        # New validation for tstats field prefixing
        if v.strip().startswith('| tstats'):
            # Find the position of the second pipe or end of string
            parts = v.split('|', 2)
            tstats_part = parts[1] if len(parts) > 1 else ''
            
            # List of fields that need to be prefixed (based on the data model)
            fields_to_check = [
                'src_ip', 'Country', 'Region', 'City', 'username', 'logged_in',
                'http_user_agent', 'http_method', 'status', 'uri_path',
                'bytes_in', 'bytes_out'
            ]
            
            # Check for datamodel name
            datamodel_match = re.search(r'FROM\s+datamodel\s*=\s*(\w+)', tstats_part, re.IGNORECASE)
            if datamodel_match:
                datamodel_name = datamodel_match.group(1)
                
                # Check each field usage
                for field in fields_to_check:
                    # Look for field usage before the second pipe
                    # Exclude cases where field is already properly prefixed
                    field_pattern = rf'(?<![\w.])({field})(?![\w.])'
                    if re.search(field_pattern, tstats_part, re.IGNORECASE):
                        # Check if the field is properly prefixed with datamodel name
                        prefixed_pattern = rf'{datamodel_name}\.{field}'
                        if not re.search(prefixed_pattern, tstats_part, re.IGNORECASE):
                            raise ValueError(
                                f"In tstats queries, field '{field}' must be prefixed with "
                                f"the datamodel name '{datamodel_name}' (e.g., '{datamodel_name}.{field}')"
                            )

        # After all validation passes, replace escaped double quotes
        v = v.replace('\\"', '"')
        return v

    @field_validator('viz_type', mode='after')
    def determine_viz_type(cls, v, info):
        spl_query = info.data.get('spl_query', '').lower()
        if '| table' in spl_query:
            return 'table'
        elif any(cmd in spl_query for cmd in ['| chart', '| timechart']):
            return 'chart'
        return 'table'  # default to chart if none matched
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
class LLMConfig(BaseModel):
    """Model for LLM configuration"""
    provider: str  # Changed from Literal to str to support dynamic providers
    base_url: str
    model: str
    api_key: str = "not_used"  # For providers that don't need real API keys
    temperature: float = 0.0
#----------------------------------------------------------------------------

# LLM_CONFIGS removed - routing is now based on presence of "/" in provider name
# Models without "/" are routed to OLLAMA_BASE_URL
# Models with "/" are routed to LiteLLM proxy

#----------------------------------------------------------------------------
# Tool models
from typing import Literal

class ToolSelector(BaseModel):
    """Model for tool selection decision"""
    tool_name: Literal["find_anomalous_ip_addresses", "find_suspicious_user_sessions", "general"] = Field(
        ..., 
        description="Name of the tool to use or 'general' for default query generation"
    )
    reason: str = Field(..., description="Reason for selecting this tool")
    earliest: str = Field(default="-1d", description="Earliest time for the query")
    latest: str = Field(default="now", description="Latest time for the query")
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
class AnomalousIPsResult(BaseModel):
    """Result from anomalous IPs detection"""
    spl_query: str = Field(..., description="Generated SPL query for anomalous users detection")
    viz_type: str = Field(default="3D", description="Recommended visualization type")
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
class SuspiciousUserSessionsResult(BaseModel):
    """Result from suspicious User Sessions detection"""
    spl_query: str = Field(..., description="Generated SPL query for failed logins detection")
    viz_type: str = Field(default="table", description="Recommended visualization type")
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    llm_provider: str = "llama3.1:8b-instruct-q4_K_M"
    q_samples: int = 10  # Updated default, though actual limit is determined by data source
#----------------------------------------------------------------------------
