""" This was the base for my zero-shot function, it was not longer actually used for the thesis"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, NllbTokenizer
import torch
from tqdm import tqdm
import nltk


# load the pre-processed datasets that were previously saved
with open("preprocessed_dutch.txt", "r", encoding="utf-8") as f:
    dutch_sentences = f.read().splitlines()
with open("preprocessed_english.txt", "r", encoding="utf-8") as f:
    english_sentences = f.read().splitlines()

"""# temporarily can later be removed and moved


"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First we load in the MarianMT model
tokenizer_marianMT = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
MarianMT = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-nl-en")

# Second we load in the Facebook-NLLB model
tokenizer_facebook = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
Facebook_NLLB = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Lastly we load in the TOWER model
tokenizer_tower = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")
TOWER = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")

MarianMT = MarianMT.to(device)
Facebook_NLLB = Facebook_NLLB.to(device)
TOWER = TOWER.to(device)

models = {"marian": MarianMT, "nllb": Facebook_NLLB, "tower": TOWER}
tokenizers = {"marian": tokenizer_marianMT, "nllb": tokenizer_facebook, "tower": tokenizer_tower}
print("Model check:")
print(f"- MarianMT: {MarianMT.device}")
print(f"- Facebook_NLLB: {Facebook_NLLB.device}")
print(f"- TOWER: {TOWER.device}")

"""# We first define the evaluation metrics"""



"""# We then run zero-shot on the models for Dutch->English"""

def perform_zero_shot(models, tokenizers, sentences):
  results = {"marian": [], "nllb" : [], "tower": []} 
  for sentence in tqdm(sentences, desc="Zero-shot Translation"):
    # MarianMT
    inputs = tokenizers["marian"](sentence, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()} 
    outputs = models["marian"].generate(**inputs)
    results["marian"].append(tokenizers["marian"].decode(outputs[0], skip_special_tokens=True))


  # NLLB
    tokenizers["nllb"].src_lang = "nld_Latn"
    inputs = tokenizers["nllb"](sentence, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()} 
    forced_bos_token_id = tokenizers["nllb"].convert_tokens_to_ids(">>eng_Latn<<")
    outputs = models["nllb"].generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    results["nllb"].append(tokenizers["nllb"].decode(outputs[0], skip_special_tokens=True))


  # TOWER
    prompt = f"Translate from Dutch to English: {sentence}"
    inputs = tokenizers["tower"](prompt, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  
    outputs = models["tower"].generate(**inputs, max_new_tokens=100, pad_token_id=tokenizers["tower"].eos_token_id)
    results["tower"].append(tokenizers["tower"].decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip())
    
  return results
  
sample_sentences = dutch_sentences[:10]
results = perform_zero_shot(models, tokenizers, sample_sentences)
for i, sentence in enumerate(sample_sentences):
  print(f"\n?? Sentence {i+1}")
  print(f"?? Dutch:    {sentence}")
  print(f"?? Marian:   {results['marian'][i]}")
  print(f"?? NLLB:     {results['nllb'][i]}")
  print(f"?? Tower:    {results['tower'][i]}")


  
  
  
