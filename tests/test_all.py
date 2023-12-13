import unittest

def create_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

if __name__ == '__main__':
    all_tests = create_test_suite()
    result = unittest.TextTestRunner(verbosity=2).run(all_tests)
    if result.wasSuccessful():
        print("All tests passed successfully.")
    else:
        print("Some tests failed.")
