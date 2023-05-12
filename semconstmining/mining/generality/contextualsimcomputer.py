import logging
import pickle
from statistics import mean
from pandas import DataFrame
from tqdm import tqdm

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

    def __init__(self, config, constraints: DataFrame, nlp_helper, resource_handler: ResourceHandler = None):
        self.config = config
        self.constraints = constraints
        # reference to the word model used (default Spacy)
        self.nlp_helper = nlp_helper
        # reference to the resource handler
        self.resource_handler = resource_handler
        self.nlp_helper.precompute_embeddings_and_sims(self.resource_handler)

    def compute_label_based_contextual_dissimilarity(self, mode=mean):
        _logger.info("Computing label-based contextual similarity")
        if self.config.LABEL_BASED_GENERALITY in self.constraints.columns:
            return
        self.constraints[self.config.LABEL_BASED_GENERALITY] = 0.0
        if self.resource_handler is None:
            _logger.error("Cannot access individual model data without a resource_handler being set! Set it to use "
                          "this method")
            return
        _logger.info("Total number of constraints: %d", len(self.constraints))
        for idx, row in tqdm(self.constraints.iterrows()):
            concat_labels = self.nlp_helper.prepare_labels(row)
            if len(concat_labels) < 2:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(concat_labels) for b in concat_labels[idx + 1:]]
            sims = self.nlp_helper.get_sims(unique_combinations)
            self.constraints.at[idx, self.config.LABEL_BASED_GENERALITY] = 1 - mode(sims)

    def compute_name_based_contextual_dissimilarity(self, mode=mean):
        _logger.info("Computing name-based contextual similarity")
        if self.config.NAME_BASED_GENERALITY in self.constraints.columns:
            return
        self.constraints[self.config.NAME_BASED_GENERALITY] = 0.0
        for idx, row in self.constraints.iterrows():
            names = self.nlp_helper.prepare_names(row)
            if len(names) < 2:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(names) for b in names[idx + 1:]]
            sims = self.nlp_helper.get_sims(unique_combinations)
            self.constraints.at[idx, self.config.NAME_BASED_GENERALITY] = 1 - mode(sims)

    def compute_object_based_contextual_dissimilarity(self, mode=mean):
        _logger.info("Computing object-based contextual similarity")
        if self.config.OBJECT_BASED_GENERALITY in self.constraints.columns:
            return
        self.constraints[self.config.OBJECT_BASED_GENERALITY] = 0.0
        if self.resource_handler is None:
            _logger.error("Cannot access individual model data without a resource_handler being set! Set it to use "
                          "this method")
            return
        _logger.info("Total number of constraints: %d", len(self.constraints))
        for idx, row in tqdm(self.constraints.iterrows()):
            concat_labels = self.nlp_helper.prepare_objs(row)
            if len(concat_labels) < 2:
                continue
            unique_combinations = [(a, b) for idx, a in enumerate(concat_labels) for b in concat_labels[idx + 1:]]
            sims = self.nlp_helper.get_sims(unique_combinations)
            self.constraints.at[idx, self.config.OBJECT_BASED_GENERALITY] = 1 - mode(sims)
