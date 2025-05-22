# Scriptie
This GitHub repository contains the code I wrote for my thesis in 2025. The code runs three different NMT models: TOWER, MarianMT, and Facebook-NLLB-200 on a paralell Dutch-English corpus from the Europarl dataset.
The preprocess_data.py preprocesses the parallel corpus. The zero-shot.py and few-shot_tower.py functions are used for running the models on the corpus and returning results in the format of automatic evaluation metrics.
We recommend the use of Habrok, or other powerful hardware to run the experiment. The code requires a GPU to run efficiently, so it is recommended to use one.

## Steps to replicate the experiment

1. Clone the repository
https://github.com/JoostOving/Scriptie

2. Create a virtual environment
```python -m venv venv```
```source venv/bin/activate```

3. install requirements
```pip install -r requirements.txt```

4. run the python scripts by executing the bash scripts
```bash preprocess-data.sh```
```bash zero-shot.sh```
```bash few-shot_tower.sh```

