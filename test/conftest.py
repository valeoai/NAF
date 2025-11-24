import pytest


def pytest_addoption(parser):
    parser.addoption("--embed-dim", action="store_true", help="Sweep embed dim")
    parser.addoption("--img-size", action="store_true", help="Sweep img size")
    parser.addoption("--ratio", action="store_true", help="Sweep upsampling factor")
    parser.addoption("--lr-size", action="store_true", help="Sweep low-res feature size")


def pytest_report_teststatus(report, config):
    """
    Suppress skipped and xfailed tests from console output completely.
    """
    is_xfail = getattr(report, "wasxfail", False)  # safely check for older pytest versions
    if report.skipped or is_xfail:
        return "hidden", "", ""
