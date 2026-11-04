"""Inspect the installed DeepFace API and preprocessing implementation.

Prints the installed DeepFace version, representation signature,
and relevant library source code for diagnostic review.

Run: python audit_deepface.py
Does not execute the complete facial-recognition experiment.
"""

import inspect
from importlib.metadata import version

from deepface import DeepFace
from deepface.modules import preprocessing, representation

print('Deepface version: ', version('deepface'))

print('\n---Represent signature---')
print(inspect.signature(DeepFace.represent))

print('\n---Resize image---')
print(inspect.getsource(preprocessing.resize_image))

print('\n---Normalize input---')
print(inspect.getsource(preprocessing.normalize_input))

print('\n---Represent source---')
print(inspect.getsource(representation.represent))
