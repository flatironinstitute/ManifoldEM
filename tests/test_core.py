# Pytest exits with code 5 if no tests are found. This is a workaround to ensure that pytest
# exits with code 0 if no tests are found.
def test_pytest_is_a_jerk():
    assert True
