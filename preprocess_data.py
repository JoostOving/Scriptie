# -*- coding: utf-8 -*-

# First, we load in all the required libraries for running the code


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM 
import torch
import re
import nltk
nltk.download('punkt_tab')
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

device = torch.device('cpu')

"""## We then first load the Europarl dataset and pre-process it for later use

"""

dutch_file = 'europarl-v7.nl-en.nl'
english_file = 'europarl-v7.nl-en.en'

def load_dataset(dutch_file, english_file):
  with open (dutch_file, 'r', encoding='utf-8') as dutch, open(english_file, 'r', encoding='utf-8') as english:
        dutch_lines = dutch.readlines()
        english_lines = english.readlines()
        dutch_lines = [line.strip() for line in dutch_lines]
        english_lines = [line.strip() for line in english_lines]
        return dutch_lines, english_lines

dutch_lines, english_lines = load_dataset(dutch_file, english_file)

# Here we will pre-process the whole dataset, i made seperate functions for every pre-processing step for clarity

def text_cleaning(lines):
  url_pattern = re.compile(r"http\S+")
  email_pattern = re.compile(r'\S+@\S+')
  unwanted_chars_pattern = re.compile(r"[^a-zA-ZÀ-ÿ0-9\s.,!?\'\"-]")
  whitespace_pattern = re.compile(r'\s+')

  cleaned_lines = []
  for line in tqdm(lines, desc="Cleaning lines", unit="line"):
      line = url_pattern.sub("", line)  # remove possible URLs
      line = email_pattern.sub('', line)  # remove possible email addresses
      line = unwanted_chars_pattern.sub("", line)  # remove unwanted chars
      line = whitespace_pattern.sub(' ', line).strip()  # remove extra whitespaces
      cleaned_lines.append(line)
  return cleaned_lines

def lowercasing_data(dutch_lines, english_lines):
  dutch_lines = [line.lower() for line in dutch_lines] # lowercases all dutch lines
  english_lines = [line.lower() for line in english_lines] # lowercases all english sentences
  return dutch_lines, english_lines

def tokenize_sentences(dutch_lines, english_lines):
  dutch_lines = [nltk.sent_tokenize(line) for line in tqdm(dutch_lines, desc="Tokenizing Dutch lines", unit="line")] # tokenizes dutch sentences
  english_lines = [nltk.sent_tokenize(line) for line in tqdm(english_lines, desc="Tokenizing English lines", unit="line")] # tokenizes english sentences
  return dutch_lines, english_lines

def subword_tokenization(dutch_lines, english_lines):
  bpe_tokenizer = Tokenizer(BPE())
  bpe_tokenizer.pre_tokenizer = Whitespace()
  trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
  all_sentences = [sent for line in dutch_lines + english_lines for sent in line]
  bpe_tokenizer.train_from_iterator(all_sentences, trainer)
  dutch_lines = [[bpe_tokenizer.encode(sent).tokens for sent in line] for line in dutch_lines]
  english_lines = [[bpe_tokenizer.encode(sent).tokens for sent in line] for line in english_lines]
  return dutch_lines, english_lines

def pre_process_dataset(dutch_lines, english_lines):
    # text cleaning
    dutch_lines = text_cleaning(dutch_lines)
    english_lines = text_cleaning(english_lines)
    print("After text cleaning:")
    print("Dutch:", dutch_lines[:2])
    print("English:", english_lines[:2])

    # lowercasing
    dutch_lines, english_lines = lowercasing_data(dutch_lines, english_lines)
    print("\nAfter lowercasing:")
    print("Dutch:", dutch_lines[:2])
    print("English:", english_lines[:2])

    # sentence tokenization
    dutch_lines, english_lines = tokenize_sentences(dutch_lines, english_lines)
    print("\nAfter sentence tokenization:")
    print("Dutch:", dutch_lines[:4])
    print("English:", english_lines[:4])

    # subword tokenization
    dutch_lines, english_lines = subword_tokenization(dutch_lines, english_lines)
    print("\nAfter subword tokenization:")
    print("Dutch:", dutch_lines[:4])
    print("English:", english_lines[:4])

    return dutch_lines, english_lines
    
preprocessed_dutch, preprocessed_english = pre_process_dataset(dutch_lines, english_lines)
dutch_sentences = [" ".join(token for sentence in line for token in sentence) for line in preprocessed_dutch]
english_sentences = [" ".join(token for sentence in line for token in sentence) for line in preprocessed_english]

# save the preprocessed data to files
with open("preprocessed_dutch.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(dutch_sentences))

with open("preprocessed_english.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(english_sentences))

print("\nPreprocessed files saved as 'preprocessed_dutch.txt' and 'preprocessed_english.txt'")


