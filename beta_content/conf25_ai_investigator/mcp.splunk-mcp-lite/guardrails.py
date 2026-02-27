"""SPL query validation and output sanitization functions."""

import re
from typing import Any, Dict, List, Union, Tuple, Callable

# Helper functions for complex validation rules
def check_collect_params(query: str, context: dict, base_risk: int) -> int:
    """Check for risky collect parameters."""
    if re.search(r'\|\s*collect\b', context['query_lower']):
        if 'override=true' in context['query_lower'] or 'addtime=false' in context['query_lower']:
            return base_risk
    return 0


def check_outputlookup_params(query: str, context: dict, base_risk: int) -> int:
    """Check for risky outputlookup parameters."""
    if re.search(r'\|\s*outputlookup\b', context['query_lower']):
        if 'override=true' in context['query_lower']:
            return base_risk
    return 0


def parse_time_to_hours(time_str: str) -> float:
    """Convert Splunk time string to hours."""
    time_str = time_str.strip().lower()
    
    # Remove leading minus sign if present
    if time_str.startswith('-'):
        time_str = time_str[1:]
    
    # Handle relative time modifiers
    match = re.match(r'^(\d+)([smhdwmonqy]+)?(@[smhdwmonqy]+)?$', time_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2) if match.group(2) else 's'
        
        # Convert to hours
        multipliers = {
            's': 1/3600,      # seconds to hours
            'm': 1/60,        # minutes to hours  
            'h': 1,           # hours
            'd': 24,          # days to hours
            'w': 24*7,        # weeks to hours
            'mon': 24*30,     # months to hours (approximate)
            'q': 24*90,       # quarters to hours (approximate)
            'y': 24*365       # years to hours
        }
        
        return value * multipliers.get(unit, 1)
    
    # Handle special keywords  
    if time_str in ['0', 'all', 'alltime'] or time_str == '0':
        return float('inf')  # All time
    
    # Default to 24 hours if unparseable
    return 24


def check_time_range(query: str, context: dict, base_risk: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, str]]:
    """Check time range issues.
    
    base_risk can be:
    - int: single risk score
    - tuple: (no_risk, exceeds_safe_range, all_time/no_time)
    
    Returns:
    - int: risk score
    - OR tuple: (risk_score, time_range_type) where time_range_type is 'all_time', 'exceeds_safe', or 'no_time'
    """
    if isinstance(base_risk, tuple):
        no_risk, exceeds_safe, all_time = base_risk
    else:
        # Backward compatibility
        no_risk = 0
        exceeds_safe = int(base_risk * 0.5)
        all_time = base_risk
    
    query_lower = context['query_lower']
    safe_timerange_str = context.get('safe_timerange', '24h')
    safe_hours = parse_time_to_hours(safe_timerange_str)
    
    has_earliest = 'earliest' in query_lower or 'earliest_time' in query_lower
    has_latest = 'latest' in query_lower or 'latest_time' in query_lower
    has_time_range = has_earliest or has_latest
    
    if not has_time_range:
        # Check if it's an all-time search
        if re.search(r'all\s*time|alltime', query_lower):
            return (all_time, 'all_time')  # All time search
        else:
            # No time range specified, could default to all-time
            return (all_time, 'no_time')
    else:
        # Extract time range from query
        # Look for patterns like earliest=-30d or earliest=0
        earliest_match = re.search(r'earliest(?:_time)?\s*=\s*([^\s,]+)', query_lower)
        
        if earliest_match:
            time_value = earliest_match.group(1)
            query_hours = parse_time_to_hours(time_value)
            
            # Check if it's all time (0 or inf)
            if query_hours == float('inf') or time_value == '0':
                return (all_time, 'all_time')
            
            # Check if time range exceeds safe range
            if query_hours > safe_hours:
                return (exceeds_safe, 'exceeds_safe')
        
    return no_risk


def check_index_usage(query: str, context: dict, base_risk: Union[int, Tuple[int, ...]]) -> int:
    """Check for index usage patterns.
    
    base_risk can be:
    - int: single risk score
    - tuple: (no_risk, no_index_with_constraints, index_star_unconstrained)
    """
    if isinstance(base_risk, tuple):
        no_risk, no_index_constrained, index_star = base_risk
    else:
        # Backward compatibility
        no_risk = 0
        no_index_constrained = int(base_risk * 0.57)  # ~20/35
        index_star = base_risk
    
    query_lower = context['query_lower']
    
    if 'index=*' in query_lower:
        # Check if there are constraining source/sourcetype
        if not (re.search(r'source\s*=', query_lower) or re.search(r'sourcetype\s*=', query_lower)):
            return index_star  # Full risk for unconstrained index=*
    elif not re.search(r'index\s*=', query_lower):
        # No index specified
        if re.search(r'source\s*=|sourcetype\s*=', query_lower):
            return no_index_constrained
    return no_risk


def check_subsearch_limits(query: str, context: dict, base_risk: int) -> int:
    """Check for subsearches without limits."""
    if '[' in query and ']' in query:
        subsearch = query[query.find('['):query.find(']')+1]
        if 'maxout' not in subsearch.lower() and 'maxresults' not in subsearch.lower():
            return base_risk
    return 0


def check_expensive_commands(query: str, context: dict, base_risk: int) -> int:
    """Check for expensive commands and return appropriate score."""
    query_lower = context['query_lower']
    multiplier = 0
    
    # Check each expensive command (each adds to the multiplier)
    if re.search(r'\|\s*transaction\b', query_lower):
        multiplier += 1
    if re.search(r'\|\s*map\b', query_lower):
        multiplier += 1
    if re.search(r'\|\s*join\b', query_lower):
        multiplier += 1
    
    return int(base_risk * multiplier)


def check_append_operations(query: str, context: dict, base_risk: int) -> int:
    """Check for append operations."""
    if re.search(r'\|\s*(append|appendcols)\b', context['query_lower']):
        return base_risk
    return 0


# ===========================================================================
def validate_spl_query(query: str, safe_timerange: str) -> Tuple[int, str]:
    """
    Validate SPL query and calculate risk score using rule-based system.
    
    Args:
        query: The SPL query to validate
        safe_timerange: Safe time range from configuration
        
    Returns:
        Tuple of (risk_score, risk_message)
    """
    # Import here to avoid circular dependency
    from spl_risk_rules import SPL_RISK_RULES
    
    risk_score = 0
    issues = []
    query_lower = query.lower()
    
    # Context for function-based rules
    context = {
        'safe_timerange': safe_timerange,
        'query_lower': query_lower
    }
    
    # Process all rules
    for rule in SPL_RISK_RULES:
        pattern_or_func, base_score, message = rule
        
        if callable(pattern_or_func):
            # It's a function - call it with base_score
            result = pattern_or_func(query, context, base_score)
            
            # Handle special case where function returns (score, type) tuple
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], str):
                score, time_type = result
                if score > 0:
                    risk_score += score
                    # Special handling for time range messages
                    if pattern_or_func.__name__ == 'check_time_range':
                        if time_type == 'all_time':
                            formatted_message = f"All-time search detected (+{score}). This can be very resource intensive. Add time constraints like earliest=-24h latest=now to limit search scope."
                        elif time_type == 'exceeds_safe':
                            formatted_message = f"Time range exceeds safe limit (+{score}). Consider narrowing your search window for better performance."
                        elif time_type == 'no_time':
                            formatted_message = f"No time range specified (+{score}). Query may default to all-time. Add explicit time constraints like earliest=-24h latest=now."
                        else:
                            formatted_message = message.format(score=score)
                    else:
                        formatted_message = message.format(score=score)
                    issues.append(formatted_message)
            else:
                # Regular integer score
                score = result if isinstance(result, int) else 0
                if score > 0:
                    risk_score += score
                    # Format message with actual score
                    formatted_message = message.format(score=score)
                    issues.append(formatted_message)
        else:
            # It's a regex pattern
            if re.search(pattern_or_func, query_lower):
                risk_score += base_score
                # Format message with base score
                formatted_message = message.format(score=base_score)
                issues.append(formatted_message)
    
    # Cap risk score at 100
    risk_score = min(risk_score, 100)
    
    # Build final message
    if not issues:
        return risk_score, "Query appears safe."
    else:
        risk_message = "Risk factors found:\n" + "\n".join(f"- {issue}" for issue in issues)
        
        # Add high-risk warning if needed
        if risk_score >= 50:
            risk_message += "\n\nConsider reviewing this query with your Splunk administrator."
        
        return risk_score, risk_message
# ===========================================================================

# ===========================================================================
def sanitize_output(data: Any) -> Any:
    """
    Recursively sanitize sensitive data in output.
    
    Masks:
    - Credit card numbers (showing only last 4 digits)
    - Social Security Numbers (complete masking)
    
    Args:
        data: Data to sanitize (can be dict, list, string, or other)
        
    Returns:
        Sanitized data with same structure
    """
    # Credit card pattern - matches 13-19 digit sequences with optional separators
    cc_pattern = re.compile(r'\b(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})[-\s]?(\d{3,6})\b')
    
    # SSN pattern - matches XXX-XX-XXXX format
    ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    
    def sanitize_string(text: str) -> str:
        """Sanitize a single string value."""
        if not isinstance(text, str):
            return text
            
        # Replace credit cards, keeping last 4 digits
        def cc_replacer(match):
            last_four = match.group(4)
            # Determine separator from original
            separator = '-' if '-' in match.group(0) else ' ' if ' ' in match.group(0) else ''
            masked = f"****{separator}****{separator}****{separator}{last_four}"
            return masked
            
        text = cc_pattern.sub(cc_replacer, text)
        
        # Replace SSNs completely
        text = ssn_pattern.sub('***-**-****', text)
        
        return text
    
    # Handle different data types
    if isinstance(data, dict):
        return {key: sanitize_output(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_output(item) for item in data]
    elif isinstance(data, str):
        return sanitize_string(data)
    else:
        # For other types (int, float, bool, None), return as-is
        return data
# ===========================================================================
