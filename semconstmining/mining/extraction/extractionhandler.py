import logging
from os.path import exists

import pandas as pd

from semconstmining.mining.extraction.declareextractor import DeclareExtractor
from semconstmining.mining.extraction.modelextractor import ModelExtractor
from semconstmining.parsing.resource_handler import ResourceHandler

_logger = logging.getLogger(__name__)


class ExtractionHandler:
    """
    # Class for extracting observations from the model collection, which are in turn used to establish constraints
    """

    def __init__(self, config,  resource_handler: ResourceHandler):
        self.config = config
        self._all_const = None
        self.resource_handler = resource_handler
        self.model_extractor = ModelExtractor(config, resource_handler)
        self.declare_extractor = DeclareExtractor(config, resource_handler)

        # MP Constraints are not treated differently anymore
        self.mp_observations_ser_file = self.config.DATA_INTERIM / (self.config.MODEL_COLLECTION + "_" + self.config.MP_OBSERVATIONS_SER_FILE)
        self.declare_ser_file = self.config.DATA_INTERIM / (self.config.MODEL_COLLECTION + "_" + self.config.DECLARE_CONST)
        self.constraint_kb_ser_file = self.config.DATA_INTERIM / (self.config.MODEL_COLLECTION + "_" + self.config.CONSTRAINT_KB_SER_FILE)

        self.per_object_observations = None
        self.inter_object_observations = None
        self.resource_observations = None
        self.decision_observations = None

        self.all_observations = []

    def extract_observations_from_models(self):
        """
        # Extraction of resource constraints and decision constraints from the models directly
        :return:
        """
        if exists(self.config.DATA_INTERIM / self.mp_observations_ser_file):
            _logger.info("Loading stored model-based observations.")
            df_observations_mp = pd.read_pickle(self.mp_observations_ser_file)
            _logger.info("Loaded stored model-based observations.")
        else:
            df_observations_mp = self.model_extractor.get_perspectives_from_models()
            _logger.info(f"{len(df_observations_mp)} model-based records extracted.")
            df_observations_mp.to_pickle(self.mp_observations_ser_file)
        return df_observations_mp

    def extract_declare_constraints_from_logs(self):
        """
        # Extraction of 'semantic' DECLARE constraints from the model-generated logs
        :return:
        """
        if exists(self.declare_ser_file):
            _logger.info("Loading stored constraints.")
            df_declare = pd.read_pickle(self.declare_ser_file)
            _logger.info("Loaded stored constraints.")
        else:
            df_declare = self.declare_extractor.extract_declare_from_logs()
            df_observations_mp = self.model_extractor.get_perspectives_from_models()
            df_declare = pd.concat([df_declare, df_observations_mp])
            _logger.info(f"{len(df_declare)} declare records extracted.")
            df_declare.to_pickle(self.declare_ser_file)
        return df_declare

    def aggregate_constraints(self, min_support=1):
        kb_path = str(self.constraint_kb_ser_file).replace("constraint_", "constraint_min_support=" + str(min_support))
        if exists(kb_path):
            _logger.info("Loading stored observations.")
            temp = pd.read_pickle(kb_path)
            return temp
        temp = self.get_all_observations().copy(deep=True)
        # We determine the support of the extracted observations and remove duplicate rows
        temp[self.config.SUPPORT] = temp.groupby(self.config.CONSTRAINT_STR)[self.config.CONSTRAINT_STR].transform('count')
        # Duplicates are determined based on
        # the constraint, the type of constraint, the model name, and the object type, if any
        temp = temp.drop_duplicates(subset=[self.config.CONSTRAINT_STR, self.config.LEVEL,
                                            self.config.MODEL_NAME, self.config.OBJECT])
        # We combine the model names to get an idea of the common context a constraint occurs in
        temp[self.config.MODEL_NAME] = temp.groupby(
            self.config.CONSTRAINT_STR)[self.config.MODEL_NAME].transform(lambda x: ' | '.join(x))
        temp[self.config.MODEL_ID] = temp.groupby(
            self.config.CONSTRAINT_STR)[self.config.MODEL_ID].transform(lambda x: ' | '.join(x))
        temp = temp.drop_duplicates(subset=[self.config.CONSTRAINT_STR, self.config.LEVEL, self.config.OBJECT])
        # We only retain constraints that have the minimum support
        temp = temp[temp[self.config.SUPPORT] > min_support]
        temp.to_pickle(kb_path)
        return temp

    def get_all_observations(self):
        if not self._all_const:
            #dfs = [self.extract_declare_constraints_from_logs(),
            # self.extract_observations_from_models()]
            # , eh.extract_observations_from_logs()
            self._all_const = self.extract_declare_constraints_from_logs().set_index(["obs_id"])  # pd.concat(dfs).set_index(["obs_id"])
        return self._all_const

