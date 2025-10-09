# Avoid circular imports by delaying imports
def _lazy_import():
    from tpgmm.tpgmm.tpgmm import TPGMM
    from tpgmm.gmr.gmr import GaussianMixtureRegression
    return TPGMM, GaussianMixtureRegression

# Make classes available at module level
import sys
def __getattr__(name):
    if name in ('TPGMM', 'GaussianMixtureRegression'):
        TPGMM, GaussianMixtureRegression = _lazy_import()
        # Cache the imports in the module
        sys.modules[__name__].TPGMM = TPGMM
        sys.modules[__name__].GaussianMixtureRegression = GaussianMixtureRegression
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from tpgmm.utils import plot
