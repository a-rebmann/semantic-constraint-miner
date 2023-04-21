import re
from pathlib import Path

import spacy
import gensim.downloader as api
import nltk
import requests
import inflect

from semconstmining.config import Config

NON_ALPHANUM = re.compile('[^a-zA-Z]')
CAMEL_PATTERN_1 = re.compile('(.)([A-Z][a-z]+)')
CAMEL_PATTERN_2 = re.compile('([a-z0-9])([A-Z])')


def sanitize_label(label):
    # handle some special cases
    label = label.replace('\n', ' ').replace('\r', '')
    label = label.replace('(s)', 's')
    label = re.sub(' +', ' ', label)
    # turn any non-alphanumeric characters into whitespace
    # label = re.sub("[^A-Za-z]"," ",label)
    # turn any non-alphanumeric characters into whitespace
    label = NON_ALPHANUM.sub(' ', label)
    label = label.strip()
    # remove single character parts
    label = " ".join([part for part in label.split() if len(part) > 1])
    # handle camel case
    label = camel_to_white(label)
    # make all lower case
    label = label.lower()
    # delete unnecessary whitespaces
    label = re.sub("\s{1,}", " ", label)
    return label


def split_label(label):
    result = re.split('[^a-zA-Z]', label)
    return result


def camel_to_white(label):
    label = CAMEL_PATTERN_1.sub(r'\1 \2', label)
    return CAMEL_PATTERN_2.sub(r'\1 \2', label)


class LabelUtil:

    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load(self.config.SPACY_MODEL)
        self.glove_embeddings = api.load(self.config.WORD_EMBEDDINGS)
        self.p = inflect.engine()

    def split_and_lemmatize_label(self, label):
        words = split_label(label)
        lemmas = [self.lemmatize_word(w) for w in words]
        return lemmas

    def lemmatize_word(self, word):
        if len(word) == 0:
            return word
        doc = self.nlp(word)
        lemma = doc[0].lemma_
        lemma = re.sub('ise$', 'ize', lemma)
        return lemma

    def compute_sim(self, words):
        if len(words.split(",")) < 2:
            return words
        doc = self.nlp(words)
        token1, token2 = doc[0], doc[1]
        return token1.similarity(token2)

    def check_propn(self, tok):
        doc = self.nlp(tok)
        pos = [tok.pos_ for tok in doc]
        if all(p == 'PROPN' for p in pos):
            return True
        else:
            return False

    def transform_action_w2v(self, word):
        word = word.lower()
        doc = self.nlp(word)
        present_tense_verbs = set()
        for token in doc:
            if token.tag_.startswith("VB"):
                present_tense_verbs.add(token.lemma_)
        if not present_tense_verbs:
            related_verbs = list()
            if word in self.glove_embeddings.key_to_index:
                related_words = self.glove_embeddings.most_similar(word, topn=1000)
                related_verbs.extend([w for w, score in related_words])
            for verb in related_verbs:
                doc = self.nlp(verb)
                for tok in doc:
                    if tok.tag_.startswith("VB"):
                        present_tense_verbs.add(tok.lemma_)
                #present_tense_verbs.update(tok.lemma_ for tok in doc if tok.tag_.startswith("VB"))
                if len(present_tense_verbs) > 0:
                    break
        if present_tense_verbs:
            return " ".join(present_tense_verbs)
        else:
            print(f"Could not find a verb for {word}")
            return word


if __name__ == '__main__':
    actions = ["Creation",
               "Processing",
               "Invoicing",
               "Reconciliation",
               "Created",
               "Processed",
               "Invoiced",
               "Reconciled",
               "Automation"]
    label_util = LabelUtil(Config(Path(__file__).parents[4].resolve(), "opal"))
    standardized_labels = []
    # the following code transforms actions that may be nouns or in past tense into present tense verbs
    for action in actions:
        transformed_action = label_util.transform_action_w2v(action)
        print(f"{action} -> {transformed_action}")
