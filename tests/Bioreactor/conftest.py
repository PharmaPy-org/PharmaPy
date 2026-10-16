"""Run notebook-backed tests without requiring a graphical session."""

import matplotlib

matplotlib.use('Agg', force=True)
