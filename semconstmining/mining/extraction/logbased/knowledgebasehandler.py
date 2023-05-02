import logging
from os.path import exists

import pandas as pd
from tqdm import tqdm

from semconstmining.mining.extraction import verboceanextractor
from semconstmining.mining.extraction.logbased.constraintknowledgebase import ConstraintKnowledgeBase

_logger = logging.getLogger(__name__)


def populate_from_ser_fragments(config, case_names=None, add_verbocean=False):
    kb = ConstraintKnowledgeBase(config)
    # if case_names is specified, filter observations
    if case_names:
        pass  # TODO
    if exists(config.DATA_INTERIM / config.CF_OBSERVATIONS_SER_FILE):
        _logger.info("Loading detected languages.")
        df_observations = pd.read_pickle(config.DATA_INTERIM / config.CF_OBSERVATIONS_SER_FILE)
        for observation in tqdm(df_observations.itertuples()):
            kb.add_observation(observation.left, observation.right, observation.obs_type, observation.model_name)
    if exists(config.DATA_INTERIM / config.MP_OBSERVATIONS_SER_FILE):
        _logger.info("Loading detected languages.")
        df_mp_observations = pd.read_pickle(config.DATA_INTERIM / config.MP_OBSERVATIONS_SER_FILE)
        for observation in tqdm(df_mp_observations.itertuples()):
            kb.add_observation(observation.left, observation.right, observation.obs_type, observation.model_name)
    if add_verbocean:
        for rec in verboceanextractor.extract(config):
            kb.add_observation(rec[0], rec[1], rec[3])
    print("loaded kb with", kb.get_record_numbers(), "records")
    return kb

