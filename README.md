# Homework Projects for CS224w

## Installation

```
pip install -r requirements.txt
```

## Data

Each hwX/ directory should contain a code/data/ subdirectory. In it, you will find a `get_datasets.sh` script to fill the code/data/ directory with the required data for the scripts.

```
sh get_datasets.sh
```

## Solutions

Normally, eahc hwX/ directory will contain a code/ directory. Obtaining the solutions should be as simple as executing

```
python qN.py
```

Alternatively, launch Junyper Notebook and open the hwX.ipynb file in code/

```
jupyter notebook
```

## LaTeX

Additionally, PDF solutions are provided in the tex/ subdirectory of each hwX directory. Usually, to compile the .tex, you will need to have run the code in order to generate the required images. It's possible these images will need to be moved around so that the pdflatex can find them.