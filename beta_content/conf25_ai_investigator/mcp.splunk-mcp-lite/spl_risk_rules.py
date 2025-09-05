"""SPL risk rules configuration for query validation.

This file contains all SPL risk validation rules in a single location
for easy adjustment and maintenance. Each rule can be:
- A regex pattern with a fixed risk score
- A function with a single score or tuple of scores for different outcomes

Modify scores and messages here to adjust SPL validation behavior.
"""

from guardrails import (
    check_collect_params,
    check_outputlookup_params,
    check_index_usage,
    check_time_range,
    check_expensive_commands,
    check_append_operations,
    check_subsearch_limits
)

# ===========================================================================
# Risk rules definition: (pattern_or_function, score, message_with_suggestion)
SPL_RISK_RULES = [
    # Simple regex rules
    (r'\|\s*delete\b', 80, 
     "Uses 'delete' command (+{score}). This permanently removes data. Ensure you have backups and proper authorization before using delete."),
    
    (r'\|\s*(script|external)\b', 40,
     "Uses external script execution (+{score}). Ensure scripts are trusted and review security implications."),
    
    # Function-based rules
    (check_collect_params, 25,
     "Uses 'collect' with risky parameters (+{score}). Consider using addtime=true and avoid override=true for safer data collection."),
    
    (check_outputlookup_params, 20,
     "Uses 'outputlookup' with override=true (+{score}). This will overwrite existing lookup. Consider append=true or create a new lookup file."),
    
    (check_index_usage, (0, 20, 35),  # (no_risk, no_index_with_constraints, index_star_unconstrained)
     "Index usage issue detected (+{score}). Specify exact indexes or add source/sourcetype constraints for better performance."),
    
    (check_time_range, (0, 20, 30),  # (no_risk, exceeds_safe_range, all_time/no_time)
     "Time range issue detected (+{score}). Add time constraints like earliest=-24h latest=now to limit search scope."),
    
    (check_expensive_commands, 20,  # Base score per command
     "Uses expensive command(s) (+{score}). Consider using stats commands instead of transaction/map/join for better performance."),
    
    (check_append_operations, 15,
     "Uses append operations (+{score}). These can be memory intensive. Consider using OR conditions or union command."),
    
    (check_subsearch_limits, 20,
     "Subsearch without explicit limits (+{score}). Add maxout= or maxresults= to subsearches to prevent timeout issues."),
]
# ===========================================================================