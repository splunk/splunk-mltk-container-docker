from IPython.core.magic import (Magics, magics_class, cell_magic)

@magics_class
class SplunkSPLMagic(Magics):

    @cell_magic
    def spl(self, line, cell):
        return line, cell