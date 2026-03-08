#!/bin/bash
set -e
echo "=== Judge Reliability Harness - Setup & Run ==="

# Create virtual environment
python3 -m venv .venv

# Install dependencies
.venv/bin/pip install -q -r requirements.txt

echo "=== Running Judge Reliability Harness ==="
.venv/bin/python3 main.py
