#!/usr/bin/env python3
"""Compatibility wrapper. Canonical script: aws-pgx-setup/ec2/scripts/python/ec2_session_notify.py"""
from pathlib import Path
import runpy

target = (
    Path(__file__).resolve().parents[1]
    / "aws-pgx-setup"
    / "ec2"
    / "scripts"
    / "python"
    / "ec2_session_notify.py"
)
runpy.run_path(str(target), run_name="__main__")
