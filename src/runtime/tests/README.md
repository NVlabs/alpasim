# Testing

Tests should be written in the `tests` directory. The test files should be named `test_<module>.py` where `<module>` is
the name of the module being tested. Tests should be written for all public functions and classes in the module, aside
from trivial data classes.

## Running Tests

A prerequisite for running the tests is to have a suitable environment and the `pytest` package installed. Given that
this project is expected to be run inside a docker container, this is used as a basis for the proposed configuration.


Note that the following instructions assume you are running from the src/runtime directory.

With `uv`:

```bash
cd src/runtime
uv sync --all-extras  # This installs dev dependencies including pytest
```

And then, to run:
```bash
uv run pytest
```
