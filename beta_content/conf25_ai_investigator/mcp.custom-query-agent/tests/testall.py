#!/usr/bin/env python3
"""Run all tests for the Custom Query Agent MCP server."""

import subprocess
import sys
from pathlib import Path

def run_test(test_file: str) -> bool:
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)
    
    test_path = Path(__file__).parent / test_file
    result = subprocess.run([sys.executable, str(test_path)], capture_output=False)
    return result.returncode == 0

def main():
    """Run all tests."""
    print("Custom Query Agent MCP Server - Test Suite")
    print("="*60)
    
    tests = [
        "test_connection.py",
        # "test_query_generation.py"  # Excluded - requires manual run only
    ]
    
    results = []
    for test in tests:
        success = run_test(test)
        results.append((test, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary:")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test:<30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted")
        sys.exit(1)