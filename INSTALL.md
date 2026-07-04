# Installation

This repo needs Python 3.11, R, `gsDesign`, and `rpy2` compatible with that R.

## macOS setup

1. Install R from [CRAN macOS R downloads](https://cran.r-project.org/bin/macosx/).
2. From the repo root, run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
R -q -e 'install.packages("gsDesign", repos="https://cloud.r-project.org")'
python -c "from rpy2.robjects.packages import importr; importr('gsDesign'); print('OK')"
python -m main
```

## If it fails

If you see `libRblas.dylib`, `libR.dylib`, or `_R_ClosureEnv` errors, replace only `rpy2` inside the existing venv:

```bash
pip uninstall -y rpy2 rpy2-rinterface rpy2-robjects
R_HOME="$(R RHOME)" pip install --no-cache-dir "rpy2<3.6"
```
