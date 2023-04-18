# Directory to store internal developments

Take special care to not push contents of this folder to any public repo!

Contents in this directory will also have relaxed CI requirements (i.e. CI will not check for code coverage and docstrings for the code in this directory, although you are highly encouraged to add docstrings and sufficient tests in the `test/internal` directory!) The tests from the internal folder can be run using `make pytest-internal` from the root directory. 