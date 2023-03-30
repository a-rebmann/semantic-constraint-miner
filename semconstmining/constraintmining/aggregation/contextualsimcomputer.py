import logging
import pickle
from os.path import exists
from statistics import mean

import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, util
from semconstmining.constraintmining.bert_parser import label_utils
from semconstmining.constraintmining.bert_parser.label_utils import LabelUtil
from nltk.corpus import wordnet
from itertools import chain

from semconstmining.parsing.resource_handler import ResourceHandler

_logger = logging.getLogger(__name__)


def write_pickle(p_map, path):
    with open(path, 'wb') as handle:
        pickle.dump(p_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as handle:
        p_map = pickle.load(handle)
        return p_map


class ContextualSimilarityComputer:

    def __init__(self, config, constraints: DataFrame, resource_handler: ResourceHandler = None):
        self.config = config
        self.constraints = constraints
        # reference to the sentence model used (default SentenceTransformer)
        self._sent_model = None
        # reference to the word model used (default Spacy)
        self.word_model = LabelUtil(config)
        # reference to the resource handler
        self.resource_handler = resource_handler
        self.known_embedding_map_ser = self.config.DATA_INTERIM / \
                                       (self.resource_handler.model_collection_id + "_" + self.config.EMB_MAP)
        self.knowm_sim_ser = self.config.DATA_INTERIM / \
                             (self.resource_handler.model_collection_id + "_" + self.config.SIM_MAP)
        self.known_embeddings = {} if not exists(self.known_embedding_map_ser) else read_pickle(
            self.known_embedding_map_ser)
        # Maps pairs of labels to their similarity
        self.known_sims = {} if not exists(self.knowm_sim_ser) else read_pickle(self.knowm_sim_ser)
        _logger.info("Loaded %d known embeddings and %d known similarities" % (len(self.known_embeddings),
                                                                               len(self.known_sims)))
        if len(self.known_sims) == 0:
            if len(self.known_embeddings) == 0:
                self.pre_compute_embeddings()
        # Maps a (partial) label to its synonyms
        self.synonym_map = {}

    @property
    def sent_model(self):
        if self._sent_model is None:
            self._sent_model = SentenceTransformer(self.config.SENTENCE_TRANSFORMER)
        return self._sent_model

    def compute_label_based_contextual_dissimilarity(self, mode=mean):
        _logger.info("Computing label-based contextual similarity")
        if self.config.LABEL_BASED_SIM in self.constraints.columns:
            return
        self.constraints[self.config.LABEL_BASED_SIM] = 0.0
        if self.resource_handler is None:
            _logger.error("Cannot access individual model data without a resource_handler being set! Set it to use "
                          "this method")
            return
        for idx, row in self.constraints.iterrows():
            concat_labels = self._prepare_labels(row)
            if len(concat_labels) < 2:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(concat_labels) for b in concat_labels[idx + 1:]]
            sims = self.get_sims(unique_combinations)
            self.constraints.at[idx, self.config.LABEL_BASED_SIM] = 1 - mode(sims)
            # Persist known similarities to avoid later reprocessing
        write_pickle(self.known_sims, self.knowm_sim_ser)

    def compute_name_based_contextual_dissimilarity(self, mode=mean):
        _logger.info("Computing name-based contextual similarity")
        if self.config.NAME_BASED_SIM in self.constraints.columns:
            return
        self.constraints[self.config.NAME_BASED_SIM] = 0.0
        for idx, row in self.constraints.iterrows():
            names = self._prepare_names(row)
            if len(names) < 2:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(names) for b in names[idx + 1:]]
            sims = self.get_sims(unique_combinations)
            self.constraints.at[idx, self.config.NAME_BASED_SIM] = 1 - mode(sims)
            # Persist known similarities to avoid later reprocessing
        write_pickle(self.known_sims, self.knowm_sim_ser)

    def compute_object_based_contextual_dissimilarity(self, mode=mean):
        _logger.info("Computing object-based contextual similarity")
        if self.config.OBJECT_BASED_SIM in self.constraints.columns:
            return
        self.constraints[self.config.OBJECT_BASED_SIM] = 0.0
        for const, group in self.constraints[(self.constraints[self.config.LEVEL] == self.config.OBJECT) &
                                             (self.constraints[self.config.OPERATOR_TYPE] == self.config.BINARY)].groupby(self.config.CONSTRAINT_STR):
            unique_bos = self._prepare_objs(group)
            if len(unique_bos) < 2:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(unique_bos) for b in unique_bos[idx + 1:]]
            sim_scores = self.get_sims(unique_combinations)
            agg_score = mode(sim_scores)
            for i, row in group.iterrows():
                self.constraints.at[i, self.config.OBJECT_BASED_SIM] = 1 - agg_score

    def compute_label_based_contextual_similarity_external(self, res_const, external, mode=mean):
        _logger.info("Computing label-based contextual similarity with external labels")
        res_const[self.config.LABEL_BASED_SIM_EXTERNAL] = 0.0
        if self.resource_handler is None:
            _logger.error("Cannot access individual model data without a resource_handler being set! Set it to use "
                          "this method")
            return
        for idx, row in res_const.iterrows():
            concat_labels = self._prepare_labels(row)
            if len(concat_labels) < 1:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(concat_labels) for b in external]
            sims = self.get_sims(unique_combinations)
            res_const.at[idx, self.config.LABEL_BASED_SIM_EXTERNAL] = mode(sims)
            # Persist known similarities to avoid later reprocessing
        write_pickle(self.known_sims, self.knowm_sim_ser)
        return res_const

    def compute_name_based_contextual_similarity_external(self, res_const, external, mode=mean):
        _logger.info("Computing name-based contextual similarity with external names")
        res_const[self.config.NAME_BASED_SIM_EXTERNAL] = 0.0
        external = [label_utils.sanitize_label(ext) for ext in external]
        for idx, row in self.constraints.iterrows():
            names = self._prepare_names(row)
            if len(names) < 1:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(names) for b in external]
            sims = self.get_sims(unique_combinations)
            res_const.at[idx, self.config.NAME_BASED_SIM_EXTERNAL] = mode(sims)
            # Persist known similarities to avoid later reprocessing
        write_pickle(self.known_sims, self.knowm_sim_ser)
        return res_const

    def compute_object_based_contextual_similarity_external(self, res_const, external):
        _logger.info("Computing object-based contextual similarity with external objects " + str(external))
        res_const[self.config.OBJECT_BASED_SIM_EXTERNAL] = {}
        res_const[self.config.OBJECT_BASED_SIM_EXTERNAL] = res_const.apply(
            lambda row: self._compute_object_based_contextual_similarity_external(row, external), axis=1)
        res_const[self.config.MAX_OBJECT_BASED_SIM_EXTERNAL] = res_const[self.config.OBJECT_BASED_SIM_EXTERNAL].apply(
            lambda x: max(x[self.config.OBJECT].values()) if self.config.OBJECT in x else max(x[self.config.LEFT_OPERAND].values())
            if self.config.LEFT_OPERAND in x else max(x[self.config.RIGHT_OPERAND].values()) if self.config.RIGHT_OPERAND in x else 0.0)
        return res_const

    def _compute_object_based_contextual_similarity_external(self, row, external):
        res = {}
        if row[self.config.LEVEL] == self.config.OBJECT:
            res[self.config.OBJECT] = {}
            for ext in external:
                combi = [(row[self.config.OBJECT], ext)]
                res[self.config.OBJECT][ext] = self.get_sims(combi)[0]
        elif row[self.config.LEVEL] == self.config.MULTI_OBJECT:
            res[self.config.LEFT_OPERAND] = {}
            res[self.config.RIGHT_OPERAND] = {}
            for ext in external:
                if not pd.isna(row[self.config.LEFT_OPERAND]):
                    combi = [(row[self.config.LEFT_OPERAND], ext)]
                    res[self.config.LEFT_OPERAND][ext] = self.get_sims(combi)[0]
                if not pd.isna(row[self.config.RIGHT_OPERAND]):
                    combi = [(row[self.config.RIGHT_OPERAND], ext)]
                    res[self.config.RIGHT_OPERAND][ext] = self.get_sims(combi)[0]
        return res

    def compute_object_based_contextual_similarity_external_old(self, res_const, external, mode=max):
        _logger.info("Computing object-based contextual similarity with external objects " + str(external))
        res_const[self.config.OBJECT_BASED_SIM_EXTERNAL] = 0.0
        for const, group in self.constraints[(self.constraints[self.config.LEVEL] == self.config.OBJECT) &
                                             (self.constraints[self.config.OPERATOR_TYPE] == self.config.BINARY)].groupby(
            [self.config.CONSTRAINT_STR, self.config.OBJECT]):
            unique_bos = self._prepare_objs(group)
            if len(unique_bos) < 1:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(unique_bos) for b in external]
            sim_scores = [self.word_model.compute_sim(",".join(combi)) for combi in unique_combinations]
            agg_score = mode(sim_scores)
            for i, row in group.iterrows():
                res_const.at[i, self.config.OBJECT_BASED_SIM_EXTERNAL] = agg_score
        for const, group in self.constraints[(self.constraints[self.config.LEVEL] == self.config.MULTI_OBJECT) &
                                             (self.constraints[self.config.OPERATOR_TYPE] == self.config.BINARY)].groupby(self.config.CONSTRAINT_STR):
            unique_bos = self._prepare_objs(group, multi_obj=True)
            if len(unique_bos) < 1:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(unique_bos) for b in external]
            sim_scores = [self.word_model.compute_sim(",".join(combi)) for combi in unique_combinations]
            agg_score = mode(sim_scores)
            for i, row in group.iterrows():
                res_const.at[i, self.config.OBJECT_BASED_SIM_EXTERNAL] = agg_score
        return res_const

    def get_synonyms(self, verb):
        lemma = self.word_model.lemmatize_word(verb)
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
        return sims + known_scores

    def _prepare_labels(self, row):
        model_ids = [x.strip() for x in row[self.config.MODEL_ID].split("|")]
        concat_labels = list(
            self.resource_handler.bpmn_model_elements[
                self.resource_handler.bpmn_model_elements[self.config.MODEL_ID_BACKUP].isin(model_ids)][self.config.LABEL_LIST].unique()
        )
        # Computing semantic similarity using sentence transformers is super expensive on CPU, therefore,
        # we randomly pick k names for which we make comparisons TODO any way to ground this procedure on something?
        if len(concat_labels) > 10:
            concat_labels = concat_labels[:5] + concat_labels[-5:]
        return [label_utils.sanitize_label(concat_label) for concat_label in concat_labels if
                label_utils.sanitize_label(concat_label) not in self.config.TERMS_FOR_MISSING]

    def _prepare_names(self, row):
        names = row[self.config.MODEL_NAME].split("|")
        names = [label_utils.sanitize_label(name)
                 for name in names if label_utils.sanitize_label(name) not in self.config.TERMS_FOR_MISSING]
        # Computing semantic similarity using sentence transformers is super expensive on CPU, therefore,
        # we randomly pick k names for which we make comparisons TODO any way to ground this procedure on something?
        if len(names) > 10:
            names = names[:5] + names[-5:]
        return names

    def _prepare_objs(self, group, multi_obj=False):
        if multi_obj:
            unique_bos = list(group[self.config.LEFT_OPERAND].unique())
            for obj in group[self.config.RIGHT_OPERAND].unique():
                if obj not in unique_bos:
                    unique_bos.append(obj)
        else:
            unique_bos = list(group[self.config.OBJECT].unique())
        if "" in unique_bos:
            unique_bos.remove("")
        return unique_bos

    def pre_compute_embeddings(self, sentences=None):
        if sentences is None:
            sentences = list(set(name for names in self.constraints[self.config.MODEL_NAME].unique() for name in names.split("|")))

            if self.resource_handler is not None:
                unique_ids = list(set(
                    model_id.strip() for model_ids in self.constraints[self.config.MODEL_ID].unique() for model_id in
                    model_ids.split("|")))
                concat_labels = list(
                    self.resource_handler.bpmn_model_elements[
                        self.resource_handler.bpmn_model_elements[self.config.MODEL_ID_BACKUP].isin(unique_ids)][self.config.LABEL_LIST].unique()
                )
                sentences += concat_labels
        sentences = [label_utils.sanitize_label(name) for name in sentences if not pd.isna(name) and
                     label_utils.sanitize_label(name) not in self.config.TERMS_FOR_MISSING]
        sentences += self._prepare_objs(self.constraints)
        sentences += self._prepare_objs(self.constraints, multi_obj=True)
        self.known_embeddings |= {sent: embedding for sent, embedding in
                                  zip(sentences, self.sent_model.encode(sentences, convert_to_tensor=True)) if
                                  sent not in self.known_embeddings}
        # Too big to store
        # write_pickle(self.known_embeddings, self.known_embedding_map_ser)
