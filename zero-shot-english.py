
# Firstly we import the libraries and load the dataset from the pre-processing_scriptie.py script

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, NllbTokenizer
import torch
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
import sacrebleu
import evaluate
from evaluate import load
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  
import gc 
import re

nltk.download('punkt')

# load the pre-processed datasets that were previously saved
with open("preprocessed_dutch.txt", "r", encoding="utf-8") as f:
    dutch_sentences = f.read().splitlines()
with open("preprocessed_english.txt", "r", encoding="utf-8") as f:
    english_sentences = f.read().splitlines()


device_0 = torch.device("cuda:0")
device_1 = torch.device("cuda:1") 

# First we load in the MarianMT model
tokenizer_marianMT = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
MarianMT = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")

# Second we load in the Facebook-NLLB model
tokenizer_facebook = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
Facebook_NLLB = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Lastly we load in the TOWER model
tokenizer_tower = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")
TOWER = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")

MarianMT = MarianMT.to(device_0)
Facebook_NLLB = Facebook_NLLB.to(device_0)
TOWER = TOWER.to(device_1)

models = {"marian": MarianMT, "nllb": Facebook_NLLB, "tower": TOWER}
tokenizers = {"marian": tokenizer_marianMT, "nllb": tokenizer_facebook, "tower": tokenizer_tower}
print("Model check:")
print(f"- MarianMT: {MarianMT.device}")
print(f"- Facebook_NLLB: {Facebook_NLLB.device}")
print(f"- TOWER: {TOWER.device}")

"""# We first define the evaluation metrics"""

bleu = evaluate.load("sacrebleu")
bleurt = load("bleurt", config_name="bleurt-tiny-128")
comet = load("comet")
chrf = load("chrf")

def calculate_bleu(predictions, references):
    result = bleu.compute(predictions=predictions, references=references)
    return result["score"]

def calculate_bleurt(predictions, references):
    return bleurt.compute(predictions=predictions, references=references)

def calculate_comet(predictions, references, sources):
    return comet.compute(predictions=predictions, references=references, sources=sources)

def calculate_chrf(predictions, references):
    return chrf.compute(predictions=predictions, references=references)

"""# We then run zero-shot on the models for Dutch->English"""

def clean_tower_output(decoded, prompts):
    cleaned = []
    for text, prompt in zip(decoded, prompts):
        stripped = text.replace(prompt, "").strip()
        match = re.split(r"(?:\.\s+|English:|Engels:|$)", stripped)[0] 
        cleaned.append(match.strip() + ".")
    return cleaned

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def perform_zero_shot(models, tokenizers, sentences, batch_size=16):
    results = {"marian": [], "nllb": [], "tower": []}
    tower_batch_size = 4  # reduced batch size for TOWER because of memory issues

    with tqdm(total=len(sentences), desc="Zero-shot Translation", unit="sentences", dynamic_ncols=True) as pbar:
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            with torch.no_grad():  
                # Marian
                marian_inputs = tokenizers["marian"](batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                marian_inputs = {k: v.to(device_0) for k, v in marian_inputs.items()}
                marian_outputs = models["marian"].generate(**marian_inputs)
                results["marian"].extend(tokenizers["marian"].batch_decode(marian_outputs, skip_special_tokens=True))
                
                # NLLB 
                tokenizers["nllb"].src_lang = "eng_Latn"
                nllb_inputs = tokenizers["nllb"](batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                nllb_inputs = {k: v.to(device_0) for k, v in nllb_inputs.items()}
                bos_token_id = tokenizers["nllb"].convert_tokens_to_ids("nld_Latn")
                nllb_outputs = models["nllb"].generate(**nllb_inputs, forced_bos_token_id=bos_token_id)
                results["nllb"].extend(tokenizers["nllb"].batch_decode(nllb_outputs, skip_special_tokens=True))

                del marian_inputs, marian_outputs # clean-up for memory
                del nllb_inputs, nllb_outputs

                # Tower 
                for j in range(0, len(batch), tower_batch_size):
                    tower_sub_batch = batch[j:j + tower_batch_size]
                    tower_prompts = [f"Translate the following from English to Dutch.\nEnglish: {s}\nDutch:" for s in tower_sub_batch]
                    tower_inputs = tokenizers["tower"](tower_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    tower_inputs = {k: v.to(device_1) for k, v in tower_inputs.items()}
                    tower_outputs = models["tower"].generate(**tower_inputs, max_new_tokens=70, pad_token_id=tokenizers["tower"].eos_token_id)
                    tower_decoded = tokenizers["tower"].batch_decode(tower_outputs, skip_special_tokens=True)
                    cleaned = clean_tower_output(tower_decoded, tower_prompts)
                    results["tower"].extend(cleaned)
                    
                    del tower_inputs, tower_outputs, tower_decoded, tower_prompts
                    torch.cuda.empty_cache()

            # progress bar for tracking the progress
            pbar.update(len(batch))
            torch.cuda.empty_cache()

    return results

sample_size = 5000 # i used 5000 as a sample for my thesis
sample_dutch_sentences = dutch_sentences[:sample_size]
sample_english_sentences = english_sentences[:sample_size]
results = perform_zero_shot(models, tokenizers, sample_english_sentences)

print("\n--- Sample Translations ---")
for i, sentence in enumerate(sample_english_sentences):
    print(f"\n?? Sentence {i+1}")
    print(f"?? English:    {sentence}")
    print(f"?? Reference:  {sample_dutch_sentences[i]}")
    print(f"?? Marian:   {results['marian'][i]}")
    print(f"?? NLLB:     {results['nllb'][i]}")
    print(f"?? Tower:    {results['tower'][i]}")

print("\n--- Evaluation Metrics ---")

marian_predictions = results['marian']
nllb_predictions = results['nllb']
tower_predictions = results['tower']



# BLEU
bleu_marian = calculate_bleu(marian_predictions, sample_dutch_sentences)
bleu_nllb = calculate_bleu(nllb_predictions, sample_dutch_sentences)
bleu_tower = calculate_bleu(tower_predictions, sample_dutch_sentences)
print("\nBLEU Scores:")
print(f"- MarianMT: {bleu_marian:.3f}")
print(f"- NLLB:     {bleu_nllb:.3f}")
print(f"- Tower:    {bleu_tower:.3f}")

# BLEURT 
bleurt_marian = sum(calculate_bleurt(marian_predictions, sample_dutch_sentences)['scores'])/len(sample_dutch_sentences)
bleurt_nllb = sum(calculate_bleurt(nllb_predictions, sample_dutch_sentences)['scores'])/len(sample_dutch_sentences)
bleurt_tower = sum(calculate_bleurt(tower_predictions, sample_dutch_sentences)['scores'])/len(sample_dutch_sentences)
print("\nBLEURT Scores (average):")
print(f"- MarianMT: {bleurt_marian:.3f}")
print(f"- NLLB:     {bleurt_nllb:.3f}")
print(f"- Tower:    {bleurt_tower:.3f}")

# COMET
comet_marian = calculate_comet(marian_predictions, sample_dutch_sentences, sample_english_sentences)['mean_score']
comet_nllb = calculate_comet(nllb_predictions, sample_dutch_sentences, sample_english_sentences)['mean_score']
comet_tower = calculate_comet(tower_predictions, sample_dutch_sentences, sample_english_sentences)['mean_score']
print("\nCOMET Scores:")
print(f"- MarianMT: {comet_marian:.3f}")
print(f"- NLLB:     {comet_nllb:.3f}")
print(f"- Tower:    {comet_tower:.3f}")

# CHRF
chrf_marian = calculate_chrf(marian_predictions, sample_dutch_sentences)['score']
chrf_nllb = calculate_chrf(nllb_predictions, sample_dutch_sentences)['score']
chrf_tower = calculate_chrf(tower_predictions, sample_dutch_sentences)['score']
print("\nCHRF Scores:")
print(f"- MarianMT: {chrf_marian:.3f}")
print(f"- NLLB:     {chrf_nllb:.3f}")
print(f"- Tower:    {chrf_tower:.3f}")
