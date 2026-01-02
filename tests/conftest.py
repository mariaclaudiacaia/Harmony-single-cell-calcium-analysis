import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--rebase",
        action="store_true",
        default=False,
        help="Rebase golden files instead of comparing (write new golden files).",
    )


@pytest.fixture(scope="session")
def rebase(request):
    # allow environment variable or CLI flag
    env = os.environ.get("REBASE_GOLDEN", "0")
    if env in ("1", "true", "True"):
        return True
    return request.config.getoption("--rebase")
