# Installation

Currently, bayespreg supports Python >= [3.12]. Please make sure that you are operating in a Python 3 environment.

## Installing a released version

`bayespreg` is available on both conda and pip, and can be installed with any of

```bash
conda install -c conda-forge bayespreg
```

or

```bash
pixi add bayespreg
```

or

```bash
pip install bayespreg
```

## Installing a development from source

For working with a development version, we recommend [miniforge] or [pixi]. To get started, clone this repository or download it manually then `cd` into the directory and run the following commands:

**using conda**

```bash
conda env create -f environment.yml
conda activate bayespreg
pip install -e .
```

**using pixi**

*note*: as of this writing, pixi does not support relative paths (like "."), hence the expansion using the environment variable `$PWD`

```bash
pixi init --import environment.yml
pixi add --pypi --editable  "bayespreg @ file://$PWD"
```

You can also [fork] the [pysal/bayespreg] repo and create a local clone of your fork. By making changes to your local clone and submitting a pull request to [pysal/bayespreg], you can contribute to the bayespreg development.

[3.12]: https://docs.python.org/3.12/
[miniforge]: https://github.com/conda-forge/miniforge
[fork]: https://help.github.com/articles/fork-a-repo/
[pysal/bayespreg]: https://github.com/pysal/bayespreg
[python package index]: https://pypi.org/pysal/bayespreg/
[pixi]: https://pixi.prefix.dev/latest/
