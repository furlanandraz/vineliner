"""Runs the pipeline stages in order and saves intermediates under runs/<run-id>/."""

from .run import RunResult, load_inputs, make_run_dir, plan_run, run_pipeline

__all__ = ["RunResult", "load_inputs", "make_run_dir", "plan_run", "run_pipeline"]
