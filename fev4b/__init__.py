"""Fev-4B public inference interface."""
from .model import Fev4B, route_with_predict
from .integrity import verify_package
from .decisions import decide_with_predict

__all__ = ['Fev4B', 'route_with_predict', 'decide_with_predict', 'verify_package']
__version__ = '0.2.0'
