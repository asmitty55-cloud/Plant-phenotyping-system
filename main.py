#!/usr/bin/env python
"""
Plant Timelapse System - Simple entry point
Run from project root: python main.py
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pt.api.app import run_app

if __name__ == "__main__":
    run_app()
