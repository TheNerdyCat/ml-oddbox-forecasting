import pytest
import sys


def main():
    exit_code = pytest.main(["tests", "-v"])
    if exit_code != 0:
        sys.exit("Tests failed!")
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()
