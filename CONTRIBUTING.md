# Contributor Guide

Thank you for your interest in improving Retrievall!
This project is open-source under the [MIT License] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here are some important resources for contributors:

- [Source Code]
- [Documentation]
- [Issue Tracker]
- [Code of Conduct]

[MIT License]: https://opensource.org/licenses/MIT
[Source Code]: https://github.com/CohereHealth/retrievall
[Documentation]: https://retrievall.readthedocs.io/
[Issue Tracker]: https://github.com/CohereHealth/retrievall/issues
[Code of Conduct]: CODE_OF_CONDUCT.md

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, please answer these questions:

- What operating system and Python version are you using?
- What version of Retrievall are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## Setting up your development environment

You need Python 3.7+ and a virtual environment manager like [venv] or [virtualenv].

1. Fork the repository on GitHub.
2. Clone your fork to your local machine.
3. Create a new virtual environment for the project.
4. Install the package and its dependencies using `pip`:

   ```console
   $ pip install -e '.[dev]'
5. You can now run tests and work on the codebase.

## How to test the project

We use [pytest] for testing. To run the full test suite:

```console
$ pytest
```

Unit tests are located in the `tests/` directory.

[pytest]: https://docs.pytest.org/

## How to submit changes

1. Create a new branch for your changes.
2. Make your changes and commit them (don't forget to add tests!).
3. Push your changes to your fork on GitHub.
4. Submit a pull request to the main repository.

Before submitting a pull request, please make sure:

- All tests pass.
- Your code follows the existing style (we use [Black] for code formatting).
- Your changes are covered by tests.
- Any new features or changes to existing functionality are documented.

We recommend opening an issue to discuss your proposed changes before starting work on a pull request. This allows us to provide guidance and feedback.

[Black]: https://black.readthedocs.io/