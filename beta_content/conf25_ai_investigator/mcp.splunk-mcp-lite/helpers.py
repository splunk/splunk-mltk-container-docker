"""Helper functions for formatting Splunk search results."""

from typing import List, Dict, Any


def format_events_as_markdown(events: List[Dict[str, Any]], query: str) -> str:
    """Convert events to markdown table format."""
    if not events:
        return f"Query: {query}\nNo events found."
    
    # Get all unique keys from events
    all_keys = []
    seen_keys = set()
    for event in events:
        for key in event.keys():
            if key not in seen_keys:
                all_keys.append(key)
                seen_keys.add(key)
    
    # Build markdown table
    lines = [f"Query: {query}", f"Found: {len(events)} events", ""]
    
    # Header
    header = "| " + " | ".join(all_keys) + " |"
    separator = "|" + "|".join(["-" * (len(key) + 2) for key in all_keys]) + "|"
    lines.extend([header, separator])
    
    # Rows
    for event in events:
        row_values = []
        for key in all_keys:
            value = str(event.get(key, ""))
            # Escape pipe characters in values
            value = value.replace("|", "\\|")
            row_values.append(value)
        row = "| " + " | ".join(row_values) + " |"
        lines.append(row)
    
    return "\n".join(lines)


def format_events_as_csv(events: List[Dict[str, Any]], query: str) -> str:
    """Convert events to CSV format."""
    if not events:
        return f"# Query: {query}\n# No events found"
    
    # Get all unique keys
    all_keys = []
    seen_keys = set()
    for event in events:
        for key in event.keys():
            if key not in seen_keys:
                all_keys.append(key)
                seen_keys.add(key)
    
    lines = [f"# Query: {query}", f"# Events: {len(events)}", ""]
    
    # Header
    lines.append(",".join(all_keys))
    
    # Rows
    for event in events:
        row_values = []
        for key in all_keys:
            value = str(event.get(key, ""))
            # Escape quotes and handle commas
            if "," in value or '"' in value or "\n" in value:
                value = '"' + value.replace('"', '""') + '"'
            row_values.append(value)
        lines.append(",".join(row_values))
    
    return "\n".join(lines)


def format_events_as_summary(events: List[Dict[str, Any]], query: str, event_count: int) -> str:
    """Create a natural language summary of events."""
    lines = [f"Query: {query}", f"Total events: {event_count}"]
    
    if not events:
        lines.append("No events found.")
        return "\n".join(lines)
    
    # Analyze events
    if len(events) < event_count:
        lines.append(f"Showing: First {len(events)} events")
    
    # Time range analysis if _time exists
    if events and "_time" in events[0]:
        times = [e.get("_time", "") for e in events if e.get("_time")]
        if times:
            lines.append(f"Time range: {times[-1]} to {times[0]}")
    
    # Field analysis
    all_fields = set()
    for event in events:
        all_fields.update(event.keys())
    
    lines.append(f"Fields: {', '.join(sorted(all_fields))}")
    
    # Value frequency analysis for common fields
    for field in ["status", "sourcetype", "host", "source"]:
        if field in all_fields:
            values = [str(e.get(field, "")) for e in events if field in e]
            if values:
                value_counts = {}
                for v in values:
                    value_counts[v] = value_counts.get(v, 0) + 1
                top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                summary = ", ".join([f"{v[0]} ({v[1]})" for v in top_values])
                lines.append(f"{field.capitalize()} distribution: {summary}")
    
    # Sample events
    lines.append("\nFirst 3 events:")
    for i, event in enumerate(events[:3], 1):
        lines.append(f"Event {i}: " + " | ".join([f"{k}={v}" for k, v in event.items()]))
    
    return "\n".join(lines)