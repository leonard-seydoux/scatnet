#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is the scatnet_learnable.py from Randall modified."""

print("""
ScatNet – a deep scattering network with learnable wavelets.
(C) Randall Balestriero and Leonard Seydoux
""") # noqa

from . import io
from . import data
from . import display
# from . import models
from . import logtable
from .layer import Scattering


__all__ = [
    # 'models',
    'display',
    'io',
    'logtable',
    'data',
    'Scattering']
