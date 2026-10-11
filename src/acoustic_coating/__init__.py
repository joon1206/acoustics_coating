"""Acoustic coating models based on JCA equivalent-fluid theory and 1D TMM."""

from .air import AirProperties, DEFAULT_AIR
from .jca import JCAMaterial
from .viscoelastic import ViscoelasticLayer

__all__ = ["AirProperties", "DEFAULT_AIR", "JCAMaterial", "ViscoelasticLayer"]

