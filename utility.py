"""Miscellaneous utility functions for this project."""
import random


def coin_flip(probability=0.5):
    """Randomly return True with the given probability, False otherwise."""
    return random.random() < probability
