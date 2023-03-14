import re

import spacy

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
    label = re.sub("\s{1,}"," ",label)
    return label


def split_label(label):
    label = label.lower()
    result = re.split('[^a-zA-Z]', label)
    return result


def camel_to_white(label):
    label = CAMEL_PATTERN_1.sub(r'\1 \2', label)
    return CAMEL_PATTERN_2.sub(r'\1 \2', label)


class LabelUtil:

    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load(self.config.SPACY_MODEL)

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
