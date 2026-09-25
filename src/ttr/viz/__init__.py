"""Pygame viewer (PLAN.md Phase 3). Needs the optional dependency: pip install -e ".[viz]".

The engine never imports this package.

Deliberately re-exports nothing: `from ttr.viz.screen import Screen` here would
import pygame on `import ttr.viz`, and the engine's tests would need the extra.
"""
