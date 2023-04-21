import os
import sys
import warnings
from os.path import exists
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from semconstmining.log.loghandler import LogHandler
from semconstmining.log.loginfo import LogInfo
from semconstmining.log.logstats import LogStats
from semconstmining.conformance.declare_checker import DeclareChecker
from semconstmining.config import Config
from semconstmining.constraintmining.aggregation.contextualsimcomputer import ContextualSimilarityComputer, read_pickle, \
    write_pickle
from semconstmining.constraintmining.aggregation.dictionaryfilter import DictionaryFilter
from semconstmining.constraintmining.aggregation.subsumptionanalyzer import SubsumptionAnalyzer
from semconstmining.declare.ltl.declare2ltlf import to_ltl_str
from semconstmining.declare.parsers.decl_parser import parse_single_constraint
from semconstmining.declare.enums import nat_lang_templates
from semconstmining.recommandation.consistency import ConsistencyChecker
from semconstmining.recommandation.constraintfilter import ConstraintFilter
from semconstmining.recommandation.constraintrecommender import ConstraintRecommender
from semconstmining.recommandation.constraintfitter import ConstraintFitter
from semconstmining.recommandation.filter_config import FilterConfig
from semconstmining.recommandation.recommendation_config import RecommendationConfig

warnings.simplefilter('ignore')
import logging
from semconstmining.constraintmining.extraction.extractionhandler import ExtractionHandler
from semconstmining.parsing.resource_handler import ResourceHandler

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    level=logging.INFO)

_logger = logging.getLogger(__name__)

ONLY_ENGLISH = True


def get_log_info(config: Config):
    log_infos = {} if not exists(config.DATA_INTERIM / config.LOG_INFO) else read_pickle(
        config.DATA_INTERIM / config.LOG_INFO)
    still_there = []
    to_remove = []
    lh = LogHandler(config)
    for (dir_path, dir_names, filenames) in os.walk(config.DATA_LOGS):
        for filename in filenames:
            still_there.append(filename)
            print(dir_path, filename)
            if filename not in log_infos:
                pd_log = lh.read_log(dir_path, filename)
                if pd_log is not None:
                    log_infos[filename] = LogStats(name=filename, num_traces=pd_log[config.XES_CASE].nunique(),
                                                   num_activities=pd_log[config.XES_NAME].nunique(),
                                                   num_events=pd_log.size)
    for log_info in log_infos.keys():
        if log_info not in still_there:
            to_remove.append(log_info)
    for log_info in to_remove:
        del log_infos[log_info]
    write_pickle(log_infos, config.DATA_INTERIM / config.LOG_INFO)
    return list(log_infos.values())


def store_preprocessed(config, constraints, min_support, dict_filter, mark_redundant, with_nat_lang):
    constraints.to_pickle(
        config.DATA_INTERIM / (config.MODEL_COLLECTION + "_" + "supp=" + str(min_support) +
                               "_" + "dict=" + str(dict_filter) +
                               "_" + "redundant=" + str(mark_redundant) +
                               "_" + "nat_lang=" + str(with_nat_lang) +
                               "_" + config.PREPROCESSED_CONSTRAINTS))


def load_preprocessed(config, min_support, dict_filter, mark_redundant, with_nat_lang):
    consts = pd.read_pickle(
        config.DATA_INTERIM / (config.MODEL_COLLECTION + "_" + "supp=" + str(min_support) +
                               "_" + "dict=" + str(dict_filter) +
                               "_" + "redundant=" + str(mark_redundant) +
                               "_" + "nat_lang=" + str(with_nat_lang) +
                               "_" + config.PREPROCESSED_CONSTRAINTS))
    return consts


def get_all_constraints(config, resource_handler, min_support=2, dict_filter=False,
                        mark_redundant=True, with_nat_lang=True):
    """
    Get all constraints from the observations extracted from the process model collection
    :param resource_handler: The resources, i.e., the process models, to be analyzed
    :param min_support: the minimum number of times an observation
    needs to be made in order to be considered a constraint
    :param dict_filter: whether a dictionary of object types shall be used to keep or ignore objects.
    The terms to be matched are the keys, the values of the dictionary are optional translations into natural
    language names of their object type
    :param mark_redundant: whether to analyze redundancy of constraints (subsumption, duplication, negation)
    :param with_nat_lang: whether to include a natural language explanation for the constraints
    :param with_context_similarity: whether to calculate generalizability scores based on the semantics of the textual
    content of the constraints
    :return: a DataFrame containing the extracted and preprocessed constraints

    Parameters
    ----------
    config
    """
    # TODO this main getter should have a smarter way of managing its parameters

    eh = ExtractionHandler(config, resource_handler)

    if exists(config.DATA_INTERIM / (config.MODEL_COLLECTION + "_" + "supp=" + str(min_support) +
                                     "_" + "dict=" + str(dict_filter) +
                                     "_" + "redundant=" + str(mark_redundant) +
                                     "_" + "nat_lang=" + str(with_nat_lang) +
                                     "_" + config.PREPROCESSED_CONSTRAINTS)):
        constraints = load_preprocessed(config, min_support, dict_filter, mark_redundant, with_nat_lang)
    else:
        if min_support > 1:
            constraints = eh.aggregate_constraints(min_support=min_support)
        else:
            constraints = eh.get_all_observations()
        if dict_filter:
            dict_fil = DictionaryFilter(config, constraints)
            dict_fil.mark_natural_language_objects()
            constraints = dict_fil.filter_with_proprietary_dict(prop_dict={})
        #constraints = constraints[constraints[config.LEVEL] != config.ACTIVITY]
        if mark_redundant:
            # We analyze the extracted constraints with respect to their hierarchy,
            # we then keep stronger constraints with the same support as weaker ones, which we remove
            subsumption_analyzer = SubsumptionAnalyzer(config, constraints)
            subsumption_analyzer.check_refinement()
            subsumption_analyzer.check_subsumption()
            subsumption_analyzer.check_equal()
            # translate the DECLARE constraints into LTL formulae
        constraints.reset_index(inplace=True)
        constraints[config.LTL] = constraints.apply(
            lambda x: x[config.RECORD_ID] + " := " + to_ltl_str(x[config.CONSTRAINT_STR]) + ";", axis=1)
        if with_nat_lang:
            constraints[config.NAT_LANG_TEMPLATE] = constraints[config.CONSTRAINT_STR].apply(
                lambda x: nat_lang_templates[
                    parse_single_constraint(x)["template"].templ_str if parse_single_constraint(x) is not None else
                    x.split("[")[0]])

        store_preprocessed(config, constraints, min_support, dict_filter, mark_redundant, with_nat_lang)

    return constraints[~constraints[config.REDUNDANT]]


def check_constraints(config, log_name, constraints=None):
    lh = LogHandler(config)
    constraints = DataFrame() if constraints is None else constraints
    pd_log = lh.read_log(config.DATA_LOGS, log_name)
    if pd_log is None:
        return None
    declare_checker = DeclareChecker(config, pd_log, constraints)
    return declare_checker.check_constraints()


def get_resource_handler(config):
    """
    # Preparing the resources for constraint mining from the given model collection
    # Either the models from the SAP-SAM data set are used (default) or the models from a Signavio workspace
    (configure the workspace ID in the log.auth script)
    :return:
    """
    resource_handler = ResourceHandler(config)
    resource_handler.load_bpmn_model_elements()
    resource_handler.load_dictionary_if_exists()
    resource_handler.determine_model_languages()
    if ONLY_ENGLISH:
        resource_handler.filter_only_english()
    resource_handler.load_bpmn_models()
    resource_handler.get_logs_for_sound_models()
    resource_handler.tag_task_labels()
    resource_handler.load_or_create_components()
    return resource_handler


def get_context_sim_computer(config, constraints, resource_handler, min_support=2, dict_filter=False,
                             mark_redundant=True,
                             with_nat_lang=True):
    """
    # To give an indicator of how generalizable constraints are, we compute the semantic similarity between
    # models these constraints were extracted from. The assumption is that if that similarity is very low
    # the constraint generalizes better, because it appeared in semantically different contexts.
    Parameters
    ----------
    dict_filter
    min_support
    with_nat_lang
    mark_redundant
    constraints
    resource_handler
    """
    contextual_similarity_computer = ContextualSimilarityComputer(config, constraints, resource_handler)
    contextual_similarity_computer.compute_object_based_contextual_dissimilarity()
    contextual_similarity_computer.compute_label_based_contextual_dissimilarity()
    contextual_similarity_computer.compute_name_based_contextual_dissimilarity()
    store_preprocessed(config, constraints, min_support, dict_filter, mark_redundant, with_nat_lang)
    return contextual_similarity_computer


def recommend_constraints_for_log(config, constraints, process, contextual_similarity_computer):
    lh = LogHandler(config)
    pd_log = lh.read_log(config.DATA_LOGS, process)
    if pd_log is None:
        _logger.info("No log found for process " + process)
        return None
    labels = list(pd_log[config.XES_NAME].unique())
    log_info = LogInfo(contextual_similarity_computer.resource_handler.bert_parser, labels, [process])
    recommender = ConstraintRecommender(config, contextual_similarity_computer, log_info)
    # recommended_constraints = recommender.non_conflicting_max_relevance(constraints, recommender_config)
    recommended_constraints = recommender.recommend(constraints, recommender_config)
    constraint_fitter = ConstraintFitter(config, process, recommended_constraints)
    fitted_constraints = constraint_fitter.fit_constraints()
    fitted_constraints = recommender.recommend_by_activation(fitted_constraints, recommender_config)
    return fitted_constraints


def run_full_extraction_pipeline(config: Config, process: str, filter_config: FilterConfig = None,
                                 recommender_config: RecommendationConfig = None):
    # General pipeline for constraint extraction, no log-specific recommendation
    resource_handler = get_resource_handler(config)
    options = resource_handler.get_filter_options()
    all_constraints = get_all_constraints(config, resource_handler)
    contextual_similarity_computer = get_context_sim_computer(config, all_constraints, resource_handler)
    const_filter = ConstraintFilter(config, filter_config, resource_handler)
    filtered_constraints = const_filter.filter_constraints(all_constraints)

    # Log-specific constraint recommendation
    recommended_constraints = recommend_constraints_for_log(config, filtered_constraints, process,
                                                            contextual_similarity_computer)
    consistency_checker = ConsistencyChecker(config)
    inconsistent_subsets = consistency_checker.check_consistency(recommended_constraints)
    # TODO ask user to select correction set, or just recommend subset where least relevant correction set is removed
    check_constraints(config, process, recommended_constraints)
    _logger.info("Done")


CURRENT_LOG_WS = "defaultview-2"
CURRENT_LOG_FILE = "semconsttest.xes"

if __name__ == "__main__":
    conf = Config(Path(__file__).parents[2].resolve(), "opal")
    filter_config = FilterConfig(conf)
    recommender_config = RecommendationConfig(conf)
    run_full_extraction_pipeline(config=conf, process=CURRENT_LOG_FILE,
                                 filter_config=filter_config, recommender_config=recommender_config)
    sys.exit(0)
