"""Shim exposing the vendored torchvision v0.4.0 resnet builders used by agents/nets.py."""
from .resnet import resnet18, resnet50  # noqa: F401
