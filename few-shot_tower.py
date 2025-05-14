from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
import sacrebleu
import evaluate
from evaluate import load
import tensorflow as tf
import gc 
import re

# Disable TF GPU usage
tf.config.set_visible_devices([], 'GPU')

# Download NLTK data
nltk.download('punkt')

# Load preprocessed data
with open("preprocessed_dutch.txt", "r", encoding="utf-8") as f:
    dutch_sentences = f.read().splitlines()
with open("preprocessed_english.txt", "r", encoding="utf-8") as f:
    english_sentences = f.read().splitlines()

# Set device
device = torch.device("cuda:0")

# Load Tower model
tokenizer_tower = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")
TOWER = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")
TOWER = TOWER.to(device)

# Evaluation metrics
bleu = evaluate.load("sacrebleu")
bleurt = load("bleurt", config_name="bleurt-tiny-128")
comet = load("comet")
chrf = load("chrf")

def clean_tower_output(decoded, prompts):
    cleaned = []
    for text, prompt in zip(decoded, prompts):
        stripped = text.replace(prompt, "").strip()
        match = re.split(r"(?:\.\s+|Dutch:|Nederlands:|$)", stripped)[0]
        cleaned.append(match.strip() + ".")
    return cleaned

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def calculate_bleu(predictions, references):
    result = bleu.compute(predictions=predictions, references=references)
    return result["score"]

def calculate_bleurt(predictions, references):
    return bleurt.compute(predictions=predictions, references=references)

def calculate_comet(predictions, references, sources):
    return comet.compute(predictions=predictions, references=references, sources=sources)

def calculate_chrf(predictions, references):
    return chrf.compute(predictions=predictions, references=references)

# 1-shot translation + evaluation
sample_size = 5000  # Adjust this sample size as needed
sample_dutch_sentences = dutch_sentences[:sample_size]
sample_english_sentences = english_sentences[:sample_size]

results = {"tower": []}

with tqdm(total=len(sample_dutch_sentences), desc="5-shot Tower", unit="sentences", dynamic_ncols=True) as pbar:
    for i in range(0, len(sample_dutch_sentences), 4):  # batch size = 4
        batch = sample_dutch_sentences[i:i + 4]

        tower_prompts = []
        for s in batch:
            prompt = (
                "Dutch: de commissie moet onwettige steun en steun die de interne markt daadwerkelijk dwarsboomt, zien te traceren."
                "English: the commission must track down the illegal aid and the aid which actually hinders the internal market."
                "Dutch: in de huidige situatie sta ik daarom sceptisch tegenover het idee van een europese openbare aanklager, "
                "dat zeer waarschijnlijk niet kan worden uitgevoerd binnen het kader van de huidige verdragen."
                "English: in the present situation, I am therefore sceptical about the idea of a European prosecutor, "
                "which it is scarcely possible to implement within the framework of the present treaties."
                "Dutch: de heer nielson kan hier vandaag niet aanwezig zijn omdat hij in Zuid-Afrika is."
                "English: Mr. Nielson cannot be present today since he is in South Africa."
                "Dutch:    hieruit volgt dat elk voorstel dat een ingrijpende hervorming van de toepassing van de concurrent ieregels beoogt , grondig moet worden getoetst ."
                "English:  it therefore follows that any proposal which suggests major reform of the machinery for competition policy enforcement must be closely and carefully examined ."
                "Dutch: mijnheer de voorzitter , beste collega ' s , met uw goed vinden wil ik kort het woord nemen om een tweetal punten te belichten die ons in dit verslag op vallen en die van essentieel strategisch belang zijn voor onze visie op de toekomst van de europese unie ."
                "English: mr president , i would like to say a few words in order to highlight two points made in these reports which are of fundamental strategic importance to the way we see the union ."
                f"Translate the following from Dutch to English.\nDutch: {s}\nEnglish:" 
            )
            tower_prompts.append(prompt)

        # Perform the translation using TOWER
        with torch.no_grad():
            tower_inputs = tokenizer_tower(tower_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            tower_inputs = {k: v.to(device) for k, v in tower_inputs.items()}
            tower_outputs = TOWER.generate(**tower_inputs, max_new_tokens=70, pad_token_id=tokenizer_tower.eos_token_id)
            tower_decoded = tokenizer_tower.batch_decode(tower_outputs, skip_special_tokens=True)
            
            # Clean the decoded output
            cleaned = clean_tower_output(tower_decoded, tower_prompts)
            results["tower"].extend(cleaned)

            # Clean up memory
            del tower_inputs, tower_outputs, tower_decoded, tower_prompts
            torch.cuda.empty_cache()

        # Update the progress bar
        pbar.update(len(batch))

# --- Evaluation ---
tower_predictions = results['tower']

# Evaluate the results (use the same metrics calculation as before)
print("\n--- Sample Translations ---")
for i, sentence in enumerate(sample_dutch_sentences):
    print(f"\n?? Sentence {i+1}")
    print(f"?? Dutch:    {sentence}")
    print(f"?? Reference:  {sample_english_sentences[i]}")
    print(f"?? Tower:    {results['tower'][i]}")

print("\n--- Evaluation Metrics ---")
tower_predictions = results['tower']

# BLEU
bleu_tower = calculate_bleu(tower_predictions, sample_english_sentences)
print("\nBLEU Scores:")
print(f"- Tower:    {bleu_tower:.3f}")

# BLEURT (average of scores)
bleurt_tower = sum(calculate_bleurt(tower_predictions, sample_english_sentences)['scores']) / len(sample_english_sentences)
print("\nBLEURT Scores (average):")
print(f"- Tower:    {bleurt_tower:.3f}")

# COMET
comet_tower = calculate_comet(tower_predictions, sample_english_sentences, sample_dutch_sentences)['mean_score']
print("\nCOMET Scores:")
print(f"- Tower:    {comet_tower:.3f}")

# CHRF
chrf_tower = calculate_chrf(tower_predictions, sample_english_sentences)['score']
print("\nCHRF Scores:")
print(f"- Tower:    {chrf_tower:.3f}")


