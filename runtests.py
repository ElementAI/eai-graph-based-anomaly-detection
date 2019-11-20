#!/usr/bin/env python3
# flake8: noqa
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import sys
from flake8.main.cli import main as flake8_main
from mypy.main import main as mypy_main
DIR_ROOT = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(os.path.join(DIR_ROOT))

if __name__ == '__main__':
    print("### Checking PEP-8 code adherence ###")
    try:
        flake8_main()
    except SystemExit as err:
        success_flake8 = (not err.code)

    print("### Verifying type annotations and type coherence ###")
    try:
        # This requires packages and modules to be well-defined (i.e., have __init__.py)
        # Which is a useful way to keep type-checking out of messy experimental folders
        # with an opt-in mechanism
        mypy_main(None,
                  stdout=sys.stdout,
                  stderr=sys.stderr,
                  args=["--ignore-missing-imports",
                   "--strict-optional",
                   "--incremental",
                   "-p",
                   "eai_graph_tools"],
                  )
        success_mypy = True
    except SystemExit:
        success_mypy = False

    print("### Running unit tests ###")
    success_pytest = (pytest.main(sys.argv[1:] + ["tests/"]) == 0)

    # print("### Running integration tests ###")
    # TODO: review integration testing

    if all([success_flake8, success_mypy, success_pytest]):
        print("### Success ###")
    else:
        for success, msg in [
            (success_flake8, "PEP-8 linting"),
            (success_mypy, "Type verifications"),
            (success_pytest, "Unit tests"),
        ]:
            if not success:
                print(f"### FAIL: {msg} ###")
        sys.exit(1)
