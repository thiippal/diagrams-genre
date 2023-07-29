# Corpus-based insights into multimodality and genre in primary school science diagrams

## Description

This repository contains code associated with the article *Corpus-based insights into multimodality and genre in primary school science diagrams* by Tuomo Hiippala, published in [Visual Communication](https://doi.org/10.1177/14703572231161829) (open access).

## Preliminaries

To reproduce the results reported in the article, you must first download the following data:

 1. The Allen Institute for Artificial Intelligence Diagrams (AI2D) dataset ([direct download](http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip))
 2. The AI2D-RST corpus ([direct download](https://korp.csc.fi/download/AI2D-RST/v1.1/ai2d-rst-v1-1.zip))

Clone this repository and extract the AI2D corpus into the same directory under `ai2d`. Then extract the AI2D-RST corpus into the same directory as the AI2D corpus. The directory structure should be as below:

```
ai2d/
├── ai2d-rst/
├── annotations/
├── images/
└── questions/
└── categories.json
└── categories_ai2d-rst.json
```

You should also [create a fresh virtual environment](https://docs.python.org/3/library/venv.html) for Python 3.8+ and install the libraries defined in `requirements.txt` using the following command:

`pip install -r requirements.txt`

## Codebase

## Contact

Questions? Open an issue on GitHub or e-mail me at tuomo dot hiippala @ helsinki dot fi.
