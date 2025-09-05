import re

#============================================================================
# Improved v.4 to support objects with attributes.
def safe_get(data_structure, keys, default=None):
    """
    Safely retrieves a value from a nested structure that may include
    dictionaries, lists, tuples, objects with attributes, or dictionary-like objects.

    Args:
        data_structure: The starting object (dict, list, tuple, custom object, or dict-like).
        keys: A sequence of keys (for dicts), indices (for lists/tuples), or attribute names (for objects).
              Dict keys and object attributes must be strings; list/tuple indices can be any valid index type.
        default: The value to return if access fails at any level (default: None).

    Returns:
        The value if all keys/indices/attributes are found, otherwise the default.
    """
    current = data_structure
    for key in keys:
        try:
            # Try dictionary-like access first (covers dicts and objects with __getitem__)
            if hasattr(current, '__getitem__'):
                current = current[key]
            # Then try attribute access if dictionary-like access isn't applicable
            else:
                current = getattr(current, key)
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    return current
#============================================================================

#============================================================================
# Safe-retrieve first key of inner dictionary
def safe_get_first_key(xdict, default_value):
    
    try:
        retval=next(iter(xdict))
    except:
        retval=default_value
    return retval
#============================================================================

#============================================================================
def parse_exception_info(e):
    x_text=""
    try:    
        x_text=f"Exception: {e.reason}({e.code})" 
    except: 
        pass

    try:    
        x_extra_raw_text = e.read().decode('utf-8')
        x_extra_raw_text = re.sub(r'\s+', ' ', x_extra_raw_text).strip()
        if x_extra_raw_text:
            x_extra_raw_text = ". Extra: " + x_extra_raw_text
    except: 
        x_extra_raw_text = ""

    return x_text + x_extra_raw_text
#============================================================================

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# For debugging only
# Set to 1 to enable debugging
debug_mode = 0
if debug_mode==1:
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))           # Allow other computers to attach to debugpy at this IP address and port.
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()   # Pause the program until a remote debugger is attached
    print("Debugger attached")
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
