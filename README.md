## Code Organization

### Files Root directory

* `README.md` explains the whole package.
* `GFNN.py` includes the `class GFNN`.
* `utils.py` includes utility functions (e.g. numerical integrators).

### Subdirectories of examples

* Examples of the two body problem, the Henon Heiles system, PCR3BP and the standard map are demonstrated in each subdirectory.
* In each example directory, file `prepare_data.py` is used to generate datasets.
* In each example directory, demos of GFNN are presented in the jupter-notebook files.
    + Each demo file include three parts
        1. Data gererating.
        2. Training.
        3. Predicting with results plotted.

## License

This package is for ICML submission only. All rights reserved.
Copyright © 2021 [Renyi Chen, Molei Tao]
