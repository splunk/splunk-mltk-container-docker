"""Init for SplunkSPLMagic"""
__version__ = '1.0.0'

import SplunkSPLMagic

def load_ipython_extension(ipython):
    ipython.register_magics(SplunkSPLMagic)