"""Shared pytest fixtures for stick_fighter tests."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import pytest


@pytest.fixture
def screen():
    """Provide a pygame screen surface for rendering tests."""
    pygame.init()
    surface = pygame.display.set_mode((1280, 720))
    yield surface
    pygame.quit()
