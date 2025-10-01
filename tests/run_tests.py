#!/usr/bin/env python3
"""
Test Runner for py-TIM Library

This script provides an easy way to run the test suite with various options.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --performance      # Run performance tests only
    python run_tests.py --quick            # Run quick tests only (no slow/performance tests)
"""

import subprocess
import sys
import argparse


def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="py-TIM Test Runner")
    parser.add_argument("--coverage", action="store_true",
                       help="Run tests with coverage report")
    parser.add_argument("--performance", action="store_true",
                       help="Run only performance tests")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only (exclude slow/performance tests)")
    parser.add_argument("--html", action="store_true",
                       help="Generate HTML coverage report")

    args = parser.parse_args()

    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("pytest not installed. Run: pip install -r requirements-dev.txt")
        sys.exit(1)

    # Build pytest command
    cmd_parts = ["python -m pytest tests/ -v"]

    if args.performance:
        cmd_parts.append("-m performance")
    elif args.quick:
        cmd_parts.append("-m 'not slow and not performance'")

    if args.coverage or args.html:
        cmd_parts.append("--cov=./ --cov-report=term")
        if args.html:
            cmd_parts.append("--cov-report=html:htmlcov")

    cmd = " ".join(cmd_parts)

    print("▓▒░ Running py-TIM Test Suite ░▒▓")
    print(f"Command: {cmd}")
    print("-" * 50)

    success = run_command(cmd)

    if success:
        print("✓ All tests passed!")
        if args.html:
            print("HTML coverage report generated in htmlcov/")
    else:
        print("✗ Some tests failed!")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
