import os, sys
import re
import inspect

#----------------------------------------------------------------------------
# Add the 'lib*' directories to the Python path
# lib_path = os.path.join(os.path.dirname(__file__), 'lib')
# sys.path.insert(0, lib_path)
# lib_path = os.path.join(os.path.dirname(__file__), 'lib.debug')
# sys.path.insert(0, lib_path)
lib_path = os.path.join(os.path.dirname(__file__), 'config')
sys.path.insert(0, lib_path)
#----------------------------------------------------------------------------

#from dotenv import load_dotenv
# load_dotenv('api_keys.txt')
# from api_keys import *

#############################################################################
# For debugging only
debug=0
if debug:
    import debugpy
    debugpy.listen(('0.0.0.0', 5681))           # Allow other computers to attach to debugpy at this IP address and port.
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()   # Pause the program until a remote debugger is attached
#############################################################################

# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # Pydantic setup
# import logfire

# from dotenv import load_dotenv
# load_dotenv()

# #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# # For debugging only
# debug=1
# if debug:
#     import debugpy
#     debugpy.listen(('0.0.0.0', 5681))           # Allow other computers to attach to debugpy at this IP address and port.
#     print("Waiting for debugger attach...")
#     debugpy.wait_for_client()   # Pause the program until a remote debugger is attached
# #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

# # 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
# logfire.configure(send_to_logfire='if-token-present')
# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


#============================================================================
g_do_print=1
def PRINT(message):
    if g_do_print == 1:
        # Get the caller's frame
        caller_frame = inspect.currentframe().f_back
        # Get the filename and line number
        filename = caller_frame.f_code.co_filename
        line_number = caller_frame.f_lineno
        # Format the output
        output = f"{filename}({line_number}): {message}"
        print(output)
        sys.stdout.flush()
#============================================================================

#============================================================================
def format_to_html(input_text):

    modified_text = re.sub(r'User query:', r'       User Query:', input_text)
    return modified_text
#============================================================================

#============================================================================
# safe-retrieve value from nested mix of dicts or lists.
def safe_get(xdict, keys, default):
    curr_value=xdict
    for key in keys:
        try:
            curr_value=curr_value.get(key, {})
        except:
            try:
                curr_value=curr_value[key]
            except:
                curr_value={}

    if curr_value == {}:
        curr_value=default

    return curr_value
#============================================================================
