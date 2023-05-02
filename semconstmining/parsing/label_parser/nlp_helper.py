import logging
import re
from itertools import chain
from os.path import exists
from pathlib import Path

import pandas as pd
import spacy
import gensim.downloader as api
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from semconstmining.mining.model.parsed_label import ParsedLabel
from semconstmining.parsing.label_parser.bert_wrapper import BertWrapper
from semconstmining.parsing.label_parser.bert_for_label_parsing import BertForLabelParsing
from textblob.en import Spelling
from textblob import TextBlob

from semconstmining.config import Config
from semconstmining.util.io import read_pickle, write_pickle

_logger = logging.getLogger(__name__)

NON_ALPHANUM = re.compile('[^a-zA-Z]')
CAMEL_PATTERN_1 = re.compile('(.)([A-Z][a-z]+)')
CAMEL_PATTERN_2 = re.compile('([a-z0-9])([A-Z])')


def sanitize_label(label):
    # handle some special cases
    label = str(label)
    if " - " in label:
        label = label.split(" - ")[-1]
    if "&" in label:
        label = label.replace("&", "and")
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


class NlpHelper:

    def __init__(self, config):
        self.config = config
        self.model = BertWrapper.load_serialized(config.MODEL_PATH, BertForLabelParsing)
        self.parse_map = {}
        self.nlp = spacy.load(self.config.SPACY_MODEL)
        self.glove_embeddings = api.load(self.config.WORD_EMBEDDINGS)
        # reference to the sentence model used (default SentenceTransformer)
        self._sent_model = None
        self.known_embedding_map_ser = self.config.DATA_INTERIM / \
                                       (self.config.MODEL_COLLECTION + "_" + self.config.EMB_MAP)
        self.knowm_sim_ser = self.config.DATA_INTERIM / \
                             (self.config.MODEL_COLLECTION + "_" + self.config.SIM_MAP)
        self.known_embeddings = {} if not exists(self.known_embedding_map_ser) else read_pickle(
            self.known_embedding_map_ser)
        # Maps pairs of labels to their similarity
        self.known_sims = {} if not exists(self.knowm_sim_ser) else read_pickle(self.knowm_sim_ser)
        _logger.info("Loaded %d known embeddings and %d known similarities" % (len(self.known_embeddings),
                                                                               len(self.known_sims)))
        # Maps a (partial) label to its synonyms
        self.synonym_map = {}

    def check_tok_for_object_type(self, split, pred):
        new_pred = []
        i = 1
        for tok, lab in zip(split, pred):
            if len(split) > i and tok == 'BO' and split[i] != 'BO' and self.check_propn(tok):
                new_pred.append('X')
            else:
                new_pred.append(lab)
            i += 1
        return new_pred

    def parse_label(self, label, print_outcome=False):
        if label in self.parse_map:
            return self.parse_map[label]
        split, tags = self.predict_single_label(label)
        result = ParsedLabel(self.config, label, split, tags, self.find_objects(split, tags), self.find_actions(split, tags, lemmatize=True), self.config.LANGUAGE)
        if print_outcome:
            print(label, ", act:", result.actions, ", bos:", result.bos, ', tags:', tags)
        self.parse_map[label] = result
        return result

    def parse_labels(self, splits: list) -> list:
        processed = self.model.predict(splits)
        tagged_list = []
        for split, tagged in zip(splits, processed[0]):
            tagged_clean = self.check_tok_for_object_type(split, tagged)
            tagged_list.append(tagged_clean)
        return tagged_list

    def predict_single_label(self, label):
        split = split_label(label)
        return split, self.model.predict([split])[0][0]

    def predict_single_label_full(self, label):
        split = split_label(label)
        return split, self.model.predict([split])

    def find_objects(self, split, tags):
        bos_temp = " ".join([tok if bo == 'BO' else '#+#' for tok, bo in zip(split, tags)])
        return bos_temp.split('#+#')

    def find_actor(self, split, tags):
        return [tok for tok, bo in zip(split, tags) if bo == 'ACTOR']

    def find_recipient(self, split, tags):
        return [tok for tok, bo in zip(split, tags) if bo == 'REC']

    def find_actions(self, split, tags, lemmatize=False):
        return [self.transform_action_w2v(tok) if lemmatize else
                tok for tok, a in zip(split, tags) if a in ['A', 'ASTATE', 'BOSTATE']]

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

    @property
    def sent_model(self):
        if self._sent_model is None:
            self._sent_model = SentenceTransformer(self.config.SENTENCE_TRANSFORMER)
        return self._sent_model

    def get_synonyms(self, verb):
        lemma = self.lemmatize_word(verb)
        if lemma in self.synonym_map:
            return self.synonym_map[lemma]
        synsets = wordnet.synsets(lemma)
        synonyms = set(chain.from_iterable([word.lemma_names() for word in synsets]))
        synonyms.add(lemma)
        self.synonym_map[lemma] = synonyms
        return synonyms

    def get_sims(self, unique_combinations):
        known_scores = [self.known_sims[combi]
                        for combi in unique_combinations if combi in self.known_sims]

        sentences1 = [combi[0] for combi in unique_combinations if combi not in self.known_sims]
        sentences2 = [combi[1] for combi in unique_combinations if combi not in self.known_sims]

        if len(sentences1 + known_scores) < 1:
            return []
        # Compute embedding for both lists
        sims = []
        if len(sentences1) > 0:
            embeddings1 = [self.known_embeddings[sent] for sent in sentences1]
            embeddings2 = [self.known_embeddings[sent] for sent in sentences2]
            # Compute cosine-similarities
            cosine_scores = [float(util.cos_sim(embedding1, embedding2)) for embedding1, embedding2 in
                             zip(embeddings1, embeddings2)]
            for i, _ in enumerate(sentences1):
                self.known_sims[(sentences1[i], sentences2[i])] = float(cosine_scores[i])
            sims = [float(cosine_scores[i]) for i in range(len(sentences1))]
            write_pickle(self.known_sims, self.knowm_sim_ser)
        return sims + known_scores

    def prepare_labels(self, row, resource_handler):
        model_ids = [x.strip() for x in row[self.config.MODEL_ID].split("|")]
        concat_labels = list(
            resource_handler.bpmn_model_elements[
                resource_handler.bpmn_model_elements[self.config.MODEL_ID].isin(model_ids)][self.config.CLEANED_LABEL].unique()
        )
        # Computing semantic similarity using sentence transformers is super expensive on CPU, therefore,
        # we randomly pick k names for which we make comparisons TODO any way to ground this procedure on something?
        if len(concat_labels) > 10:
            concat_labels = concat_labels[:5] + concat_labels[-5:]
        return [concat_label for concat_label in concat_labels if concat_label not in self.config.TERMS_FOR_MISSING]

    def prepare_names(self, row):
        names = row[self.config.MODEL_NAME].split("|")
        names = [sanitize_label(name)
                 for name in names if sanitize_label(name) not in self.config.TERMS_FOR_MISSING
                 and not type(name) == float]
        # Computing semantic similarity using sentence transformers is super expensive on CPU, therefore,
        # we randomly pick k names for which we make comparisons TODO any way to ground this procedure on something?
        if len(names) > 10:
            names = names[:5] + names[-5:]
        return names

    def prepare_objs(self, row, resource_handler):
        model_ids = [x.strip() for x in row[self.config.MODEL_ID].split("|")]
        concat_objects = set()
        for model_id in model_ids:
            if model_id in resource_handler.components.all_objects_per_model:
                concat_objects.update(resource_handler.components.all_objects_per_model[model_id])
        concat_objects = list(concat_objects)
        if len(concat_objects) > 10:
            concat_objects = concat_objects[:5] + concat_objects[-5:]
        if "" in concat_objects:
            concat_objects.remove("")
        return [bo for bo in concat_objects if not pd.isna(bo) and bo not in self.config.TERMS_FOR_MISSING]

    def prepare_actions(self, row):
        unique_actions = [row[self.config.LEFT_OPERAND]]
        if row[self.config.RIGHT_OPERAND] not in unique_actions:
            unique_actions.append(row[self.config.RIGHT_OPERAND])
        if "" in unique_actions:
            unique_actions.remove("")
        return [action for action in unique_actions if not pd.isna(action)
                and action not in self.config.TERMS_FOR_MISSING]

    def pre_compute_embeddings(self, constraints, resource_handler, sentences=None):
        """
        Pre-computes the embeddings for all natural language components relevant for the constraints
        :param constraints: the constraints
        :param resource_handler: the resource handler
        :param sentences: the sentences to pre-compute embeddings for
        """
        if sentences is None:
            sentences = list(set(name for names in constraints[self.config.MODEL_NAME].unique()
                                 for name in names.split("|")))
            sentences = [sanitize_label(name) for name in sentences if not pd.isna(name) and
                         sanitize_label(name) not in self.config.TERMS_FOR_MISSING]
            if resource_handler is not None:
                unique_ids = list(set(
                    model_id.strip() for model_ids in constraints[self.config.MODEL_ID].unique() for model_id in
                    model_ids.split("|")))
                concat_labels = list(
                    resource_handler.bpmn_model_elements[
                        resource_handler.bpmn_model_elements[self.config.MODEL_ID].isin(unique_ids)][self.config.CLEANED_LABEL].unique()
                )
                sentences += concat_labels
                concat_objects = set()
                sentences = [name for name in sentences if not type(name) == float]
                for model_id in unique_ids:
                    if model_id in resource_handler.components.all_objects_per_model:
                        concat_objects.update(resource_handler.components.all_objects_per_model[model_id])
                concat_objects = list(concat_objects)
                if "" in concat_objects:
                    concat_objects.remove("")
                sentences += concat_objects
                res_labels = list(constraints[constraints[self.config.LEVEL] == self.config.RESOURCE][self.config.LEFT_OPERAND].unique())
                concat_res = set()
                for model_id in unique_ids:
                    if model_id in resource_handler.components.all_resources_per_model:
                        concat_res.update(resource_handler.components.all_resources_per_model[model_id])
                concat_res = list(concat_res)
                if "" in concat_res:
                    concat_res.remove("")
                sentences += res_labels
        self.known_embeddings |= {sent: embedding for sent, embedding in
                                  zip(sentences, self.sent_model.encode(sentences, convert_to_tensor=True)) if
                                  sent not in self.known_embeddings}


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
    label_util = NlpHelper(Config(Path(__file__).parents[4].resolve(), "opal"))
    standardized_labels = []
    # the following code transforms actions that may be nouns or in past tense into present tense verbs
    for action in actions:
        transformed_action = label_util.transform_action_w2v(action)
        print(f"{action} -> {transformed_action}")
