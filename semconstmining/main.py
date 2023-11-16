import os
import sys
import time
import warnings
from os.path import exists
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from semconstmining.log.loghandler import LogHandler
from semconstmining.log.loginfo import LogInfo
from semconstmining.log.logstats import LogStats
from semconstmining.checking.constraintchecking.declare_checker import DeclareChecker
from semconstmining.config import Config
from semconstmining.mining.generality.contextualsimcomputer import ContextualSimilarityComputer, read_pickle, \
    write_pickle
from semconstmining.mining.aggregation.dictionaryfilter import DictionaryFilter
from semconstmining.mining.aggregation.subsumptionanalyzer import SubsumptionAnalyzer
from semconstmining.declare.ltl.declare2ltlf import to_ltl_str
from semconstmining.declare.parsers.decl_parser import parse_single_constraint
from semconstmining.declare.enums import nat_lang_templates
from semconstmining.parsing.label_parser.nlp_helper import NlpHelper
from semconstmining.selection.consistency.consistency import ConsistencyChecker
from semconstmining.selection.instantiation.constraintfilter import ConstraintFilter
from semconstmining.selection.instantiation.constraintrecommender import ConstraintRecommender
from semconstmining.selection.instantiation.constraintfitter import ConstraintFitter
from semconstmining.selection.instantiation.filter_config import FilterConfig
from semconstmining.selection.instantiation.recommendation_config import RecommendationConfig
from semconstmining.selection.relevance.relevance_computer import RelevanceComputer

warnings.simplefilter('ignore')
import logging
from semconstmining.mining.extraction.extractionhandler import ExtractionHandler
from semconstmining.parsing.resource_handler import ResourceHandler

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    level=logging.INFO)

_logger = logging.getLogger(__name__)

ONLY_ENGLISH = True


def get_parts_of_constraints(config, constraints: DataFrame):
    objects = list(
        constraints[constraints[config.LEVEL] == config.OBJECT][config.OBJECT].dropna().unique())
    objects += list(constraints[constraints[config.LEVEL] == config.MULTI_OBJECT][
                        config.LEFT_OPERAND].dropna().unique())
    objects += list(constraints[constraints[config.LEVEL] == config.MULTI_OBJECT][
                        config.RIGHT_OPERAND].dropna().unique())
    labels = list(
        constraints[constraints[config.LEVEL] == config.ACTIVITY][config.LEFT_OPERAND].dropna().unique())
    labels += list(constraints[constraints[config.LEVEL] == config.ACTIVITY][
                       config.RIGHT_OPERAND].dropna().unique())
    labels += list(
        constraints[constraints[config.LEVEL] == config.RESOURCE][config.LEFT_OPERAND].dropna().unique())
    resources = list(
        constraints[constraints[config.LEVEL] == config.RESOURCE][config.OBJECT].dropna().unique())
    return objects + labels + resources


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


def get_or_mine_constraints(config, resource_handler, min_support=2, dict_filter=False,
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
        constraints = eh.aggregate_constraints(min_support=min_support)
        if dict_filter:
            dict_fil = DictionaryFilter(config, constraints)
            dict_fil.mark_natural_language_objects()
            constraints = dict_fil.filter_with_proprietary_dict(prop_dict={})
        # constraints = constraints[constraints[config.LEVEL] != config.ACTIVITY]
        if mark_redundant:
            # We analyze the extracted constraints with respect to their hierarchy,
            # we then keep stronger constraints with the same support as weaker ones, which we remove
            subsumption_analyzer = SubsumptionAnalyzer(config, constraints)
            subsumption_analyzer.check_refinement()
            subsumption_analyzer.check_subsumption()
            subsumption_analyzer.check_equal()
            constraints = subsumption_analyzer.constraints
        constraints.reset_index(inplace=True)
        if with_nat_lang:
            constraints[config.NAT_LANG_TEMPLATE] = constraints[config.CONSTRAINT_STR].apply(
                lambda x: nat_lang_templates[
                    parse_single_constraint(x)["template"].templ_str if parse_single_constraint(x) is not None else
                    x.split("[")[0]])

        store_preprocessed(config, constraints, min_support, dict_filter, mark_redundant, with_nat_lang)
    for to_ignore in config.CONSTRAINT_TYPES_TO_IGNORE:
        _logger.info("Ignoring constraints of type " + to_ignore)
        constraints = constraints[~(constraints[config.TEMPLATE] == to_ignore)]
    for level, to_ignore in config.CONSTRAINT_TEMPLATES_TO_IGNORE_PER_TYPE.items():
        _logger.info("Ignoring constraints of type " + str(to_ignore) + " at level " + level)
        constraints = constraints[(constraints[config.LEVEL] != level) |
                                  ((constraints[config.LEVEL] == level) & (
                                      ~constraints[config.TEMPLATE].isin(to_ignore)))]
    constraints[config.LTL] = constraints.apply(
        lambda x: x[config.RECORD_ID] + " := " + to_ltl_str(x[config.CONSTRAINT_STR]) + ";", axis=1)
    non_redundant = constraints[~constraints[config.REDUNDANT]]
    # Absence1 does not help us here at all except for role constraints, so we remove it
    mask = non_redundant[config.CONSTRAINT_STR].str.contains("Absence1") & ~(
            non_redundant[config.LEVEL] == config.RESOURCE)
    non_redundant = non_redundant[~mask]
    return non_redundant


def check_constraints(config, process, constraints, nlp_helper, pd_log=None, with_id=False):
    lh = LogHandler(config)
    if pd_log is None:
        pd_log = lh.read_log(config.DATA_LOGS, process)
        if pd_log is None:
            _logger.info("No log found for process " + process)
            return None
    else:
        lh.log = pd_log
    constraints = DataFrame() if constraints is None else constraints
    declare_checker = DeclareChecker(config, lh, constraints, nlp_helper)
    return declare_checker.check_constraints(with_aggregates=False, with_id=with_id)


def get_resource_handler(config, nlp_helper):
    """
    # Preparing the resources for constraint mining from the given model collection
    # Either the models from the SAP-SAM data set are used (default) or the models from a Signavio workspace
    (configure the workspace ID in the log.auth script)
    :return:
    """
    resource_handler = ResourceHandler(config, nlp_helper)
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


def get_context_sim_computer(config, constraints, nlp_helper, resource_handler, min_support=2, dict_filter=False,
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
    contextual_similarity_computer = ContextualSimilarityComputer(config, constraints, nlp_helper, resource_handler)
    nlp_helper.store_sims()
    contextual_similarity_computer.compute_object_based_contextual_dissimilarity()
    contextual_similarity_computer.compute_label_based_contextual_dissimilarity()
    contextual_similarity_computer.compute_name_based_contextual_dissimilarity()
    # nlp_helper.cluster(contextual_similarity_computer.constraints)
    _logger.info("Generality computed")
    store_preprocessed(config, contextual_similarity_computer.constraints, min_support, dict_filter, mark_redundant,
                       with_nat_lang)
    return contextual_similarity_computer


def get_k_most_relevant(config, constraints, k=1000):
    constraints = pd.concat([constraints[(constraints[config.OPERATOR_TYPE] == config.UNARY) &
                                         (constraints[config.LEVEL] == config.ACTIVITY)].nlargest(
        k, [config.SEMANTIC_BASED_RELEVANCE]),
        constraints[(constraints[config.OPERATOR_TYPE] == config.BINARY) &
                    (constraints[config.LEVEL] == config.ACTIVITY)].nlargest(
            k, [config.SEMANTIC_BASED_RELEVANCE]),
        constraints[(constraints[config.OPERATOR_TYPE] == config.UNARY) &
                    (constraints[config.LEVEL] == config.OBJECT)].nlargest(
            k, [config.SEMANTIC_BASED_RELEVANCE]),
        constraints[(constraints[config.OPERATOR_TYPE] == config.BINARY) &
                    (constraints[config.LEVEL] == config.OBJECT)].nlargest(
            k, [config.SEMANTIC_BASED_RELEVANCE]),
        constraints[(constraints[config.OPERATOR_TYPE] == config.UNARY) &
                    (constraints[config.LEVEL] == config.MULTI_OBJECT)].nlargest(
            k, [config.SEMANTIC_BASED_RELEVANCE]),
        constraints[(constraints[config.OPERATOR_TYPE] == config.BINARY) &
                    (constraints[config.LEVEL] == config.MULTI_OBJECT)].nlargest(
            k, [config.SEMANTIC_BASED_RELEVANCE]),
        constraints[(constraints[config.OPERATOR_TYPE] == config.UNARY) &
                    (constraints[config.LEVEL] == config.RESOURCE)].nlargest(
            k, [config.SEMANTIC_BASED_RELEVANCE])
    ])
    return constraints


def compute_relevance_for_log(config, constraints, nlp_helper, process, pd_log=None,
                              precompute=True, store_sims=True, log_id=None, k_most_relevant=None):
    lh = LogHandler(config)
    if pd_log is None:
        pd_log = lh.read_log(config.DATA_LOGS, process)
        if pd_log is None:
            _logger.info("No log found for process " + process)
            return None
    else:
        lh.log = pd_log
    if len(pd_log) == 0:
        _logger.info("Log is empty")
        return constraints
    labels = list(pd_log[config.XES_NAME].unique())
    resources_to_tasks = lh.get_resources_to_tasks()
    log_info = LogInfo(nlp_helper, labels, [process], resources_to_tasks, log_id=log_id)
    start_time = time.time()
    relevance_computer = RelevanceComputer(config, nlp_helper, log_info)
    constraints = relevance_computer.compute_relevance(constraints, pre_compute=precompute, store_sims=store_sims)
    nlp_helper.store_sims()
    if k_most_relevant is not None:
        constraints = get_k_most_relevant(config, constraints, k_most_relevant)
    _logger.info("Relevance computation took " + str(time.time() - start_time) + " seconds")
    return constraints


def get_labels(config, x, log_info, left=True):
    labels = []
    if x[config.LEVEL] == config.ACTIVITY:
        if left:
            labels.append(log_info.label_to_original_label[x[config.LEFT_OPERAND]] \
                              if x[config.LEFT_OPERAND] in log_info.label_to_original_label else x[config.LEFT_OPERAND])
        else:
            labels.append(log_info.label_to_original_label[x[config.RIGHT_OPERAND]] \
                              if x[config.RIGHT_OPERAND] in log_info.label_to_original_label else x[
                config.RIGHT_OPERAND])
    elif x[config.LEVEL] == config.OBJECT:
        for obj, original_labels in log_info.object_to_original_labels.items():
            if obj == x[config.OBJECT]:
                for original_label in log_info.object_to_original_labels[obj]:
                    if left and x[config.LEFT_OPERAND] in log_info.action_to_original_labels and original_label in \
                            log_info.action_to_original_labels[x[config.LEFT_OPERAND]]:
                        labels.append(original_label)
                    if (not left) and x[
                        config.RIGHT_OPERAND] in log_info.action_to_original_labels and original_label in \
                            log_info.action_to_original_labels[x[config.RIGHT_OPERAND]]:
                        labels.append(original_label)
    elif x[config.LEVEL] == config.MULTI_OBJECT:
        for obj, original_labels in log_info.object_to_original_labels.items():
            if left and obj == x[config.LEFT_OPERAND]:
                labels = original_labels
            if (not left) and obj == x[config.RIGHT_OPERAND]:
                labels = original_labels
    elif x[config.LEVEL] == config.RESOURCE:
        if left:
            labels.append(log_info.label_to_original_label[x[config.LEFT_OPERAND]] \
                              if x[config.LEFT_OPERAND] in log_info.label_to_original_label else x[config.LEFT_OPERAND])
    return labels


def add_original_labels(config, consistent_recommended_constraints, log_info):
    # add original event labels to fitted constraints dataframe
    consistent_recommended_constraints[config.LOG_LABEL_LEFT] = consistent_recommended_constraints.apply(
        lambda x: get_labels(config, x, log_info), axis=1)
    consistent_recommended_constraints[config.LOG_LABEL_RIGHT] = consistent_recommended_constraints.apply(
        lambda x: get_labels(config, x, log_info, left=False), axis=1)
    return consistent_recommended_constraints


def recommend_constraints_for_log(config, rec_config, constraints, nlp_helper, process, pd_log=None):
    lh = LogHandler(config)
    if pd_log is None:
        pd_log = lh.read_log(config.DATA_LOGS, process)
        if pd_log is None:
            _logger.info("No log found for process " + process)
            return None
    else:
        lh.log = pd_log
    labels = list(pd_log[config.XES_NAME].unique())
    log_info = LogInfo(nlp_helper, labels, [process])
    constraint_fitter = ConstraintFitter(config, process, constraints)
    fitted_constraints = constraint_fitter.fit_constraints()
    recommender = ConstraintRecommender(config, rec_config, log_info)
    recommended_constraints = recommender.recommend(fitted_constraints)
    selected_constraints = recommender.recommend_by_activation(recommended_constraints)
    return selected_constraints



def get_log_and_info(conf, nlp_helper, process):
    log_handler = LogHandler(conf)
    event_log = log_handler.read_log(conf.DATA_LOGS, process)
    labels = list(event_log[conf.XES_NAME].unique())
    resources_to_tasks = log_handler.get_resources_to_tasks()
    log_info = LogInfo(nlp_helper, labels, [CURRENT_LOG_FILE], resources_to_tasks)
    return event_log, log_info


def filter_violations(violations):
    recs = []
    for const_type, violations in violations.items():
        if const_type == conf.OBJECT:
            for obj_type, obj_violations in violations.items():
                for case, case_violations in obj_violations.items():
                    if len(case_violations) > 0:
                        recs.append({"case": case, "obj": obj_type, "violations": case_violations})
        else:
            for case, case_violations in violations.items():
                if len(case_violations) > 0:
                    recs.append({"case": case, "obj": "", "violations": case_violations})
    res = pd.DataFrame.from_records(recs)
    return res


def get_violation_to_cases_with_id(config, violations):
    violation_to_cases = {}
    for const_type, violations in violations.items():
        if const_type == config.OBJECT:
            for obj_type, obj_violations in violations.items():
                for case, case_violations in obj_violations.items():
                    if len(case_violations) > 0:
                        for violation in case_violations:
                            if violation[1] not in violation_to_cases:
                                violation_to_cases[violation[1]] = []
                            violation_to_cases[violation[1]].append(case)
        else:
            for case, case_violations in violations.items():
                if len(case_violations) > 0:
                    for violation in case_violations:
                        if violation[1] not in violation_to_cases:
                            violation_to_cases[violation[1]] = []
                        violation_to_cases[violation[1]].append(case)
    return violation_to_cases


def get_violation_to_cases(config, violations, with_id=False):
    if with_id:
        return get_violation_to_cases_with_id(config, violations)
    violation_to_cases = {}
    for const_type, violations in violations.items():
        if const_type == config.OBJECT:
            for obj_type, obj_violations in violations.items():
                for case, case_violations in obj_violations.items():
                    if len(case_violations) > 0:
                        for violation in case_violations:
                            if violation + obj_type not in violation_to_cases:
                                violation_to_cases[violation + obj_type] = []
                            violation_to_cases[violation + obj_type].append(case)
        else:
            for case, case_violations in violations.items():
                if len(case_violations) > 0:
                    for violation in case_violations:
                        if violation not in violation_to_cases:
                            violation_to_cases[violation] = []
                        violation_to_cases[violation].append(case)
    return violation_to_cases


def run_full_extraction_pipeline(config: Config, process: str, filter_config: FilterConfig = None,
                                 recommender_config: RecommendationConfig = None, write_results=False):
    # General pipeline for constraint extraction, no log-specific recommendation
    start_time = time.time()
    nlp_helper = NlpHelper(config)
    resource_handler = get_resource_handler(config, nlp_helper)
    all_constraints = get_or_mine_constraints(config, resource_handler, min_support=1)
    end_stage_1 = time.time()
    _logger.info("Stage 1 took " + str(end_stage_1 - start_time) + " seconds")
    start_time_dynamic = time.time()
    nlp_helper.pre_compute_embeddings(sentences=get_parts_of_constraints(config, all_constraints))
    # get_context_sim_computer(config, all_constraints, nlp_helper, resource_handler) # not part of this version
    # Filter constraints (optional)
    const_filter = ConstraintFilter(config, filter_config, resource_handler)
    filtered_constraints = const_filter.filter_constraints(all_constraints)
    event_log, log_info = get_log_and_info(config, nlp_helper, process)
    # Log-specific constraint recommendation
    if not exists(config.DATA_INTERIM / (CURRENT_LOG_FILE + "-constraints_with_relevance.pkl")):
        filtered_constraints = compute_relevance_for_log(config, filtered_constraints, nlp_helper, CURRENT_LOG_FILE,
                                                         pd_log=event_log, precompute=True)
        filtered_constraints.to_pickle(config.DATA_INTERIM / (CURRENT_LOG_FILE + "-constraints_with_relevance.pkl"))
    else:
        filtered_constraints = pd.read_pickle(
            config.DATA_INTERIM / (CURRENT_LOG_FILE + "-constraints_with_relevance.pkl"))
    recommended_constraints = recommend_constraints_for_log(config, recommender_config, filtered_constraints,
                                                            nlp_helper,
                                                            process, pd_log=event_log)
    consistency_checker = ConsistencyChecker(config)
    # Check for trivial inconsistencies
    recommended_constraints = consistency_checker.check_trivial_consistency(recommended_constraints)
    # Check for quasi inconsistencies
    inconsistent_subsets = []
    try:
        inconsistent_subsets = consistency_checker.check_consistency(recommended_constraints)
    except Exception as e:
        _logger.error("Error checking consistency of recommended constraints: " + str(e))
    if len(inconsistent_subsets) > 0:  # TODO used an external API, which is not stable in terms of availability
        consistent_recommended_constraints = consistency_checker.make_set_consistent_max_relevance(
            recommended_constraints,
            inconsistent_subsets)
    else:
        consistent_recommended_constraints = recommended_constraints
    # TODO ask user to select correction set, or just recommend subset where least relevant correction set is removed
    consistent_recommended_constraints = consistent_recommended_constraints[
        # (consistent_recommended_constraints["constraint_string"].str.contains("Alternate Succession"))&
        (~(consistent_recommended_constraints["template"].str.contains("Not")))
    ]
    consistent_recommended_constraints = add_original_labels(config, consistent_recommended_constraints, log_info)
    end_time = time.time()
    _logger.info("Stage 2 took " + str(end_time - start_time_dynamic) + " seconds")
    start_time_checking = time.time()
    violations = check_constraints(config, process, consistent_recommended_constraints, nlp_helper, pd_log=event_log,
                                   with_id=True)
    violations_to_cases = get_violation_to_cases(config, violations, with_id=True)
    violation_df = pd.DataFrame.from_records(
        [{"violation": violation, "num_violations": len(cases), "cases": cases} for violation, cases in
         violations_to_cases.items()])
    if len(violation_df) > 0:
        merged_df = pd.merge(consistent_recommended_constraints.reset_index(), violation_df,
                             left_on=config.RECORD_ID, right_on='violation', how='inner')
        end_time_checking = time.time()
        _logger.info("Stage 3 took " + str(end_time_checking - start_time_checking) + " seconds")
        # filtered_violations = filter_violations(violations)
        if write_results:
            merged_df["model_id"] = merged_df["model_id"].apply(lambda x: x.split(" | "))
            merged_df["model_name"] = merged_df["model_name"].apply(lambda x: x.split(" | "))
            merged_df.drop(columns=["activation", "inconsistent", "redundant", "index", "ltl"], inplace=True)
            merged_df.to_csv(config.DATA_OUTPUT / (CURRENT_LOG_FILE + "-violations.csv"), index=False)
            consistent_recommended_constraints.drop(columns=["activation", "inconsistent", "redundant", "index", "ltl"],
                                                    inplace=True)
            consistent_recommended_constraints.to_csv(config.DATA_OUTPUT / (CURRENT_LOG_FILE +
                                                                            "-recommended_constraints.csv"),
                                                      index=False)
        _logger.info("Done")


CURRENT_LOG_FILE = "runningexample.xes"
MODEL_COLLECTION = "semantic_sap_sam_filtered"

if __name__ == "__main__":
    conf = Config(Path(__file__).parents[2].resolve(), MODEL_COLLECTION)
    if CURRENT_LOG_FILE == "":
        _logger.error("Please specify log file (CURRENT_LOG_FILE) and put it into " + str(conf.DATA_LOGS))
        sys.exit(1)
    filt_config = FilterConfig(conf)
    rec_config = RecommendationConfig(conf, semantic_weight=0.9, top_k=250)
    run_full_extraction_pipeline(config=conf, process=CURRENT_LOG_FILE,
                                 filter_config=filt_config,
                                 recommender_config=rec_config, write_results=True)
    sys.exit(0)
