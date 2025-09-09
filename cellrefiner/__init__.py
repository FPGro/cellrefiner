from . import preprocessing as pp
from . import plotting as pl
from . import tools as tl
__version__ = "0.0.1"

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp','tl','pl']})