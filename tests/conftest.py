"""Shared test setup.

SDL is put in dummy mode before any test module imports pygame, and pygame is
started once for the whole session: a module that quit it left every cached font
in `ttr.viz.theme` dead for the modules that ran after it.

Engine tests must still run without pygame installed, so the fixture yields None
rather than skipping; the viz modules skip themselves with `importorskip`.
"""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="session", autouse=True)
def pygame_session():
    try:
        import pygame
    except ImportError:
        yield None
        return
    pygame.init()
    yield pygame
    pygame.quit()
