import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--containername",
        action="append",
        default=[],
        help="specify container names to test",
    )

def pytest_generate_tests(metafunc):
    if "containername" in metafunc.fixturenames:
        metafunc.parametrize("containername", metafunc.config.getoption("containername"))