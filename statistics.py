"""Compatibility wrapper for the former local ``statistics`` module.

New code should import :mod:`statistical_analysis` to avoid colliding with
Python's standard-library module of the same name.
"""

from statistical_analysis import *  # noqa: F401,F403
