import itertools
import logging
import multiprocessing
import random
import re
import time
from itertools import chain
from os.path import exists
from pathlib import Path

import pandas as pd
import gensim.downloader as api
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch.multiprocessing as mp

from semconstmining.mining.model.parsed_label import ParsedLabel
from semconstmining.parsing.label_parser.bert_wrapper import BertWrapper
from semconstmining.parsing.label_parser.bert_for_label_parsing import BertForLabelParsing

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
    label = label.replace("'", "")
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
        self.model_id_to_unique_resources = {}
        self.model_id_to_unique_objects = {}
        self.model_id_to_unique_labels = {}
        self.model_id_to_unique_name = {}
        self.model_id_to_name = {}
        self.config = config
        self.model = BertWrapper.load_serialized(config.MODEL_PATH, BertForLabelParsing)
        self.parse_map = {}
        import spacy
        self.nlp = spacy.load(self.config.SPACY_MODEL)
        self.glove_embeddings = api.load(self.config.WORD_EMBEDDINGS)
        # reference to the sentence model used (default SentenceTransformer)
        self._sent_model = None
        self.known_embedding_map_ser = self.config.DATA_INTERIM / \
                                       (self.config.MODEL_COLLECTION + "_" + self.config.EMB_MAP)
        self.knowm_sim_ser = self.config.DATA_INTERIM / \
                             (self.config.MODEL_COLLECTION + "_" + self.config.SIM_MAP)
        self.known_labels = dict()
        self.known_objects = dict()
        self.known_resources = dict()
        self.known_embeddings = {} if not exists(self.known_embedding_map_ser) else read_pickle(
            self.known_embedding_map_ser)
        # Maps pairs of labels to their similarity
        self.known_sims = {} if not exists(self.knowm_sim_ser) else read_pickle(self.knowm_sim_ser)
        _logger.info("Loaded %d known embeddings and %d known similarities" % (len(self.known_embeddings),
                                                                               len(self.known_sims)))
        # Maps a (partial) label to its synonyms
        self.synonym_map = {}
        self.similar_actions = {}

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
        result = ParsedLabel(self.config, label, split, tags, self.find_objects(split, tags),
                             self.find_actions(split, tags, lemmatize=True), self.config.LANGUAGE)
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
                # present_tense_verbs.update(tok.lemma_ for tok in doc if tok.tag_.startswith("VB"))
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
            self._sent_model = SentenceTransformer(self.config.DATA_ROOT / self.config.SENTENCE_TRANSFORMER)
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

    def get_similar_actions(self, act):
        if act in self.similar_actions:
            return self.similar_actions[act]
        try:
            related_words = self.glove_embeddings.most_similar(act, topn=100)
        except KeyError:
            return set()
        related_words = set([w for w, score in related_words if score > 0.8])
        similar_actions = set()
        for verb in related_words:
            doc = self.nlp(verb)
            for tok in doc:
                if tok.tag_.startswith("VB"):
                    similar_actions.add(tok.lemma_)
        self.similar_actions[act] = similar_actions
        return similar_actions

    def get_sims(self, unique_combinations):
        known_scores = [self.known_sims[combi]
                        for combi in unique_combinations if combi in self.known_sims]
        if len(known_scores) == len(unique_combinations):
            return known_scores
        sentences1 = [combi[0] for combi in unique_combinations if combi not in self.known_sims]
        sentences2 = [combi[1] for combi in unique_combinations if combi not in self.known_sims]

        if len(sentences1 + known_scores) < 1:
            return []
        # Compute embedding for both lists
        sims = []
        if len(sentences1) > 0:
            embeddings1 = [self.known_embeddings[sent] if sent in self.known_embeddings else
                           self.sent_model.encode(sent, convert_to_tensor=True, show_progress_bar=True)
                           for sent in sentences1]
            embeddings2 = [self.known_embeddings[sent] if sent in self.known_embeddings else
                           self.sent_model.encode(sent, convert_to_tensor=True, show_progress_bar=True)
                           for sent in sentences2]
            # Compute cosine-similarities
            cosine_scores = [float(util.cos_sim(embedding1, embedding2)) for embedding1, embedding2 in
                             zip(embeddings1, embeddings2)]
            for i, _ in enumerate(sentences1):
                self.known_sims[(sentences1[i], sentences2[i])] = float(cosine_scores[i])
            sims = [float(cosine_scores[i]) for i in range(len(sentences1))]
        return sims + known_scores

    def store_sims(self):
        write_pickle(self.known_sims, self.knowm_sim_ser)

    def prepare_labels(self, row):
        model_ids = [x.strip() for x in row[self.config.MODEL_ID].split("|")]
        concat_labels = set()
        for model_id in model_ids:
            sample_set = self.model_id_to_unique_labels[model_id]
            concat_labels.update(random.sample(sample_set, k=min(2, len(sample_set))))
        return [concat_label for concat_label in concat_labels if concat_label not in self.config.TERMS_FOR_MISSING]

    def prepare_names(self, row):
        model_ids = [x.strip() for x in row[self.config.MODEL_ID].split("|")]
        concat_names = list()
        for model_id in model_ids:
            if model_id in self.model_id_to_unique_name:
                concat_names.extend(self.model_id_to_unique_name[model_id])
        return [concat_label for concat_label in concat_names if concat_label not in self.config.TERMS_FOR_MISSING]

    def prepare_objs(self, row):
        model_ids = [x.strip() for x in row[self.config.MODEL_ID].split("|")]
        concat_objects = set()
        for model_id in model_ids:
            if model_id in self.model_id_to_unique_objects:
                sample_set = self.model_id_to_unique_objects[model_id]
                concat_objects.update(random.sample(sample_set, k=min(2, len(sample_set))))
        return [bo for bo in concat_objects if not pd.isna(bo) and bo not in self.config.TERMS_FOR_MISSING]

    def prepare_actions(self, row):
        unique_actions = [row[self.config.LEFT_OPERAND]]
        if row[self.config.RIGHT_OPERAND] not in unique_actions:
            unique_actions.append(row[self.config.RIGHT_OPERAND])
        if "" in unique_actions:
            unique_actions.remove("")
        return [action for action in unique_actions if not pd.isna(action)
                and action not in self.config.TERMS_FOR_MISSING]

    def precompute_embeddings_and_sims(self, resource_handler, sims=False):
        self.model_id_to_unique_labels = {
            model_id: [elm for elm in group[self.config.CLEANED_LABEL].dropna().unique() if elm != ''] for
            model_id, group in
            resource_handler.bpmn_model_elements.groupby(self.config.MODEL_ID)}
        self.model_id_to_unique_objects = resource_handler.components.all_objects_per_model
        self.model_id_to_unique_resources = resource_handler.components.all_resources_per_model
        self.model_id_to_name = {model_id: list(group[self.config.NAME].unique()) for model_id, group in
                                 resource_handler.bpmn_models.groupby(self.config.MODEL_ID)}
        # combine all labels, objects and resources into a set with unique elements
        elements = set([element for model_id, labels in self.model_id_to_unique_labels.items() for element in labels] +
                       [element for model_id, objects in self.model_id_to_unique_objects.items() for element in
                        objects] +
                       [element for model_id, resources in self.model_id_to_unique_resources.items() for element in
                        resources])
        # remove empty strings
        elements = [element for element in elements if element != ""]
        # compute embeddings for all elements
        embeddings = self.sent_model.encode(elements, convert_to_tensor=True, show_progress_bar=True)

        self.known_embeddings = {element: embedding for element, embedding in zip(elements, embeddings)}
        _logger.info("Number of elements: {}".format(len(elements)))
        if sims:
            # compute similarities between all labels across different models
            label_pairs = set()

            # Iterate over pairs of model IDs
            for model_id1, model_id2 in itertools.combinations(self.model_id_to_unique_labels.keys(), 2):
                labels1 = self.model_id_to_unique_labels[model_id1]
                labels2 = self.model_id_to_unique_labels[model_id2]
                # Compute the Cartesian product of the labels
                label_product = itertools.product(labels1, labels2)
                # Filter out redundant pairs and pairs with the same label
                unique_pairs = {(label1, label2) for label1, label2 in label_product if label1 != label2}
                new_pairs = unique_pairs - label_pairs
                # Add the new pairs to the set of label pairs
                label_pairs |= new_pairs
            _logger.info("Number of label pairs: {}".format(len(label_pairs)))
            label_pairs = list(label_pairs)
            # compute similarities between all objects across different models
            # sizeof the em
            object_pairs = set()
            # Iterate over pairs of model IDs
            for model_id1, model_id2 in itertools.combinations(self.model_id_to_unique_objects.keys(), 2):
                objects1 = self.model_id_to_unique_objects[model_id1]
                objects2 = self.model_id_to_unique_objects[model_id2]
                # Compute the Cartesian product of the objects
                object_product = itertools.product(objects1, objects2)
                # Filter out redundant pairs and pairs with the same object
                unique_pairs = {(object1, object2) for object1, object2 in object_product if object1 != object2}
                new_pairs = unique_pairs - object_pairs
                # Add the new pairs to the set of object pairs
                object_pairs |= new_pairs
            _logger.info("Number of object pairs: {}".format(len(object_pairs)))
            object_pairs = list(object_pairs)
            # compute similarities between all resources across different models
            resource_pairs = set()
            # Iterate over pairs of model IDs
            for model_id1, model_id2 in itertools.combinations(self.model_id_to_unique_resources.keys(), 2):
                resources1 = self.model_id_to_unique_resources[model_id1]
                resources2 = self.model_id_to_unique_resources[model_id2]
                # Compute the Cartesian product of the resources
                resource_product = itertools.product(resources1, resources2)
                # Filter out redundant pairs and pairs with the same resource
                unique_pairs = {(resource1, resource2) for resource1, resource2 in resource_product if
                                resource1 != resource2}
                new_pairs = unique_pairs - resource_pairs
                # Add the new pairs to the set of resource pairs
                resource_pairs |= new_pairs
            _logger.info("Number of resource pairs: {}".format(len(resource_pairs)))
            self.compute_sims_multi_processing(label_pairs)
            self.compute_sims_multi_processing(object_pairs)
            self.compute_sims_multi_processing(resource_pairs)
            self.store_sims()

    def pre_compute_embeddings(self, sentences):
        """
        Pre-computes the embeddings for all natural language components
        :param sentences: the sentences to pre-compute embeddings for
        """
        self.known_embeddings |= {sent: embedding for sent, embedding in
                                  zip(sentences, self.sent_model.encode(sentences, convert_to_tensor=True,
                                                                        show_progress_bar=True)) if
                                  sent not in self.known_embeddings}

    def compute_sims(self, params):
        embeddings1 = [self.known_embeddings[sent] for sent in params[0]]
        embeddings2 = [self.known_embeddings[sent] for sent in params[1]]
        cosine_scores = [float(util.cos_sim(embedding1, embedding2)) for embedding1, embedding2 in
                         zip(embeddings1, embeddings2)]
        return cosine_scores

    def compute_sims_multi_processing(self, params):
        sentences1 = [combi[0] for combi in params]
        sentences2 = [combi[1] for combi in params]
        num_sentences = len(sentences1)
        # Split the sentences into equal parts
        batch_size = 70000
        sentence_batches = [(sentences1[i:i + batch_size], sentences2[i:i + batch_size])
                            for i in range(0, num_sentences, batch_size)]

        # Compute cosine-similarities
        from concurrent.futures import ProcessPoolExecutor
        num_processes = multiprocessing.cpu_count() - 10
        _logger.info("Number of processes: {}".format(num_processes))
        mp.set_start_method('spawn')
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for batch in tqdm(sentence_batches):
                futures.append(executor.submit(self.compute_sims, batch))
            _logger.info("Number of futures: {}".format(len(futures)))
            for future in tqdm(futures):
                cosine_scores = future.result()
                for batch in sentence_batches:
                    for i, _ in enumerate(batch[0]):
                        self.known_sims[(batch[0][i], batch[1][i])] = float(cosine_scores[i])

    def replace_stuff(self, x):
        res = x[self.config.NAT_LANG_TEMPLATE].replace("{1}", x[self.config.LEFT_OPERAND]) if x[
                                                                                                  self.config.LEFT_OPERAND] != "" else \
        x[
            self.config.NAT_LANG_TEMPLATE]
        if not pd.isna(x[self.config.RIGHT_OPERAND]) and x[self.config.RIGHT_OPERAND] != "":
            res += res.replace("{2}", x[self.config.RIGHT_OPERAND])
        return res, x.index

    def cluster(self, id_to_sentence):
        ids, sentences = zip(*id_to_sentence.items())
        ids = list(ids)
        sentences = list(sentences)
        _logger.info("Encode the corpus. This might take a while")
        corpus_embeddings = self.sent_model.encode(sentences, batch_size=64, show_progress_bar=True,
                                                   convert_to_tensor=True)
        _logger.info("Start clustering")
        start_time = time.time()
        # Two parameters to tune:
        # min_cluster_size: Only consider cluster that have at least 25 elements
        # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
        clusters = util.community_detection(corpus_embeddings, min_community_size=25,
                                            threshold=0.7)

        _logger.info("Clustering done after {:.2f} sec".format(time.time() - start_time))
        cluster_to_ids = {}
        # Print for all clusters the top 3 and bottom 3 elements
        for i, cluster in enumerate(clusters):
            _logger.info("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
            for sentence_id in cluster[0:3]:
                _logger.info("\t", sentences[sentence_id])
            _logger.info("\t", "...")
            for sentence_id in cluster[-3:]:
                _logger.info("\t", sentences[sentence_id])
            for sentence_id in cluster:
                if i not in cluster_to_ids:
                    cluster_to_ids[i] = []
                cluster_to_ids[i].append(ids[sentence_id])
        return cluster_to_ids

    def cluster_constraints(self, constraints):
        sentences_and_ids = constraints.apply(lambda x: self.replace_stuff(x), axis=1).tolist()
        sentences = [sent_and_id[0] for sent_and_id in sentences_and_ids]
        ids = [sent_and_id[1] for sent_and_id in sentences_and_ids]
        corpus_sentences = list(sentences)
        _logger.info("Encode the corpus. This might take a while")
        corpus_embeddings = self.sent_model.encode(corpus_sentences, batch_size=64, show_progress_bar=True,
                                                   convert_to_tensor=True)

        _logger.info("Start clustering")
        start_time = time.time()
        # Two parameters to tune:
        # min_cluster_size: Only consider cluster that have at least 25 elements
        # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
        clusters = util.community_detection(corpus_embeddings, min_community_size=1000, threshold=0.75)

        _logger.info("Clustering done after {:.2f} sec".format(time.time() - start_time))
        constraints[self.config.CLUSTER] = -1
        # Print for all clusters the top 3 and bottom 3 elements
        for i, cluster in enumerate(clusters):
            _logger.info("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
            for sentence_id in cluster[0:3]:
                _logger.info("\t", corpus_sentences[sentence_id])
            _logger.info("\t", "...")
            for sentence_id in cluster[-3:]:
                _logger.info("\t", corpus_sentences[sentence_id])
            for sentence_id in cluster:
                constraints.at[ids[sentence_id], self.config.CLUSTER] = i + 1


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
