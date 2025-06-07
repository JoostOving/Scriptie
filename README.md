# Scriptie
This GitHub repository contains the code I wrote for my thesis in the last semester of my third year Information Science in 2025. The code runs three different NMT models: TOWER, MarianMT, and Facebook-NLLB-200 on a paralell Dutch-English corpus from the Europarl dataset.
The preprocess_data.py preprocesses the parallel corpus. The zero-shot.py and few-shot_tower.py functions are used for running the models on the corpus Dutch->English and returning results in the format of automatic evaluation metrics. The opposite file zero-shot-english.py returns the results for the English->Dutch translation.
We recommend the use of Habrok, or other powerful hardware to run the experiment. The code requires a GPU to run efficiently, so it is recommended to use one.

## Steps to replicate the experiment

1. Clone the repository
```
git clone https://github.com/JoostOving/Scriptie
```

3. Create a virtual environment
```
python -m venv venv
source venv/bin/activate
```

4. install requirements
```
pip install -r requirements.txt
```

6. run the python scripts by executing the bash scripts
```
bash preprocess-data.sh
bash zero-shot.sh
bash few-shot_tower.sh
```

