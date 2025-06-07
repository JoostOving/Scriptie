# Thesis Information Science
This GitHub repository contains the code I wrote for my thesis in the last semester of my third year Information Science in 2025. The code runs three different NMT models: TOWER, MarianMT, and Facebook-NLLB-200 on a paralell Dutch-English corpus from the Europarl dataset.
The preprocess_data.py preprocesses the parallel corpus. The zero-shot.py and few-shot_tower.py functions are used for running the models on the corpus Dutch->English and returning results in the format of automatic evaluation metrics. The opposite file zero-shot-english.py returns the results for the English->Dutch translation.
We recommend the use of Habrok, or other powerful hardware to run the experiment. The code requires a GPU to run efficiently, so it is recommended to use one.

## Steps to replicate the experiment

1. Clone the repository
```
git clone https://github.com/JoostOving/Scriptie
```

2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate
```

3. install requirements
```
pip install -r requirements.txt
```

4. run the python scripts by executing the bash scripts
```
bash preprocess-data.sh
bash zero-shot.sh
bash few-shot_tower.sh
```
"""For english translation change the file name in the bash script to zero-shot-english.py"""

# Results
## Dutch -> English
(The best scores across the models are marked bold. * denotes the best result in the zero-shot setting.)
| System                    | COMET | BLEURT | chrF   | BLEU   |
| ------------------------- | ----- | ------ | ------ | ------ |
| MarianMT (Zero-Shot)      | 0.424 | -1.000 | 42.914* | 4.438  |
| Facebook-NLLB (Zero-Shot) | 0.696* | 0.000*  | 40.073 | 13.025 |
| TOWER (Zero-Shot)         | 0.664 | -0.110 | 40.636 | 15.104* |
| TOWER (One-Shot)          | 0.735 | 0.067  | 45.999 | 17.846 |
| TOWER (Three-Shot)        | 0.568 | -0.303 | 32.371 | 10.352 |
| TOWER (Five-Shot)         |**0.740** | **0.137**  | **46.952** | **18.438** |

## English -> Dutch
| System                     | COMET | BLEURT | chrF   | BLEU   |
|---------------------------|-------|--------|--------|--------|
| MarianMT (Zero-Shot)      | 0.556 | -0.498 | 43.336 | 3.427  |
| Facebook-NLLB (Zero-Shot) | 0.334 | -1.023 | 7.136  | 0.171  |
| TOWER (Zero-Shot)         | 0.715* | -0.304* | 44.094* | 14.652* |
| TOWER (One-Shot)          | 0.722 | -0.333 | 44.145 | 13.804 |
| TOWER (Three-Shot)        | 0.667 | -0.383 | 40.296 | 12.853 |
| TOWER (Five-Shot)         | **0.756** | **-0.184** | **47.118** | **16.735** |

