"""Legacy optimizer wrapper for backward compatibility.

This module wraps the unified optimizer to maintain the existing
optimize_settings interface used by the GUI panel.
"""

from .optimizer_unified import optimize_from_gui


def optimize_settings(panel):
    """Run optimization from GUI panel.
    
    This is a wrapper around optimize_from_gui for backward compatibility.
    
    Args:
        panel: HRMTrainingPanel instance with model/teacher vars and resources panel
    
    Returns:
        Tuple of (results_dict, optimizer_instance)
    """
    return optimize_from_gui(panel)
