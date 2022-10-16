print("Import packages")
import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import html
import re
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

tqdm.pandas()
print("Done importing packages")

print("Fetching datasets")

pos_df = pd.read_csv("./datasets/pos_english_text.csv", encoding="latin")
neg_df = pd.read_csv("./datasets/neg_english_text.csv", encoding="latin")
abbreviations_df = pd.read_csv("./datasets/abbreviations.csv")
apostrophe_df = pd.read_csv("./datasets/apostrophe.csv")
emoji_df = pd.read_csv("./datasets/emoji.csv")
emoticons_df = pd.read_csv("./datasets/emoticons.csv")
raw_df = pd.concat([pos_df, neg_df])

raw_df.drop_duplicates(inplace=True)

print("Done fetching datasets")

print("Convert dataframe to dict")

abbreviations_dict = dict(abbreviations_df.values)
apostrophe_dict = dict(apostrophe_df.values)
emoji_dict = dict(emoji_df.values)
emoticons_dict = dict(emoticons_df.values)

print("Done converting dataframe to dict")


def lookup_dict(text, dictionary):
    for word in text.split():
        if word.lower() in dictionary and word.lower() in text.split():
            text = text.replace(word, dictionary[word.lower()])
    return text


def translate(input_text):
    translator = Translator()
    text = translator.translate(input_text).text
    return str(text)


def preProcessing(input_text, isSpellCheck=False, isTranslate=False):
    # Step A : Converting html entities i.e. (&lt; &gt; &amp;)

    text = html.unescape(input_text)
    # Step B : Removing "@user" from all the tweets

    text = re.sub("@[\w]*", "", text)
    # Step C : Remove http & https links
    text = re.sub("http://\S+|https://\S+", "", text)
    # Translation
    if isTranslate:
        text = translate(text)
    # Step D : Changing all the tweets into lowercase
    text = lookup_dict(text, emoticons_dict)
    # Step G : Emoticon Lookup
    text = lookup_dict(text, emoji_dict)
    # text = text.lower()
    text = text.lower()
    # Step E : Apostrophe Lookup
    text = lookup_dict(text, apostrophe_dict)
    # Step F : Short Word Lookup
    text = lookup_dict(text, abbreviations_dict)
    # Step G : Emoticon Lookup
    # Step H : Replacing Punctuations with space
    text = re.sub(r"[^\w\s]", " ", text)
    # Step I : Replacing Special Characters & Numbers (integers) with space
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Step J : Spelling Correction - With TextBlob Library
    if isSpellCheck:

        text = str(TextBlob(text).correct())
    # Step K: Remove whitespace
    text = re.sub(r"\s+", " ", text)

    return text


print("Cleaning text")

raw_df["clear_text"] = raw_df["text"].progress_apply(lambda x: preProcessing(x))

print("Done cleaning text")


def tokenize(clean_text):
    tokens = word_tokenize(clean_text)
    stop_words = set(stopwords.words("english"))
    negation_words = [
        "not",
        "never",
        "neither",
        "nor",
        "barely",
        "hardly",
        "scarcely",
        "seldom",
        "rarely",
        "no",
        "nothing",
        "none",
        "nobody",
        "nowhere",
    ]
    return [
        token for token in tokens if token in negation_words or token not in stop_words
    ]


print("Creating tokens")
raw_df["tokens"] = raw_df["clear_text"].progress_apply(lambda x: tokenize(x))
print("Done creating tokens")


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def Lemmatizer(tokens):
    # Importing library for lemmatizing

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in tokens]


print("Creating lemmats")
raw_df["lemmats"] = raw_df["tokens"].progress_apply(lambda x: Lemmatizer(x))
print("Done creating lemmats")

print("Save data")
raw_df.to_csv("./datasets/raw.csv", index=False)
df = raw_df[["lemmats", "sentiment"]]
df.to_csv("./datasets/tokens.csv", index=False)
print("Done save data")

print("Done")
