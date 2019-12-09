"""
Unit and regression test for the active_learning package.
"""

# Import package, test suite, and other packages as needed
import active_learning
import pytest
import sys

def test_active_learning_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "active_learning" in sys.modules
