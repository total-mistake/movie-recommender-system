import unittest
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_preprocessing import TestPreprocessing
from test_collaborative import TestCollaborativeModel
from test_content_based import TestContentBasedModel
from test_hybrid import TestHybridModel

def run_tests():
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    test_suite.addTests(loader.loadTestsFromTestCase(TestCollaborativeModel))
    test_suite.addTests(loader.loadTestsFromTestCase(TestContentBasedModel))
    test_suite.addTests(loader.loadTestsFromTestCase(TestHybridModel))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

if __name__ == '__main__':
    run_tests() 