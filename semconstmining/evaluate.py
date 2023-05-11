import logging
import os
import random
import time
from copy import deepcopy
from os.path import exists
from pathlib import Path

import numpy as np
import pandas as pd
from pm4py.objects.log.obj import Trace, Event, EventLog
import pm4py

from semconstmining.config import Config
from sklearn.model_selection import KFold

from semconstmining.declare.enums import Template
from semconstmining.declare.parsers import parse_single_constraint
from semconstmining.main import get_resource_handler, get_or_mine_constraints, get_context_sim_computer, \
    compute_relevance_for_log, recommend_constraints_for_log, check_constraints
from semconstmining.mining.extraction.extractionhandler import ExtractionHandler
from semconstmining.mining.extraction.modelextractor import ModelExtractor
from semconstmining.parsing.label_parser.nlp_helper import NlpHelper
from semconstmining.selection.consistency.consistency import ConsistencyChecker
from semconstmining.selection.instantiation.constraintfilter import ConstraintFilter
from semconstmining.selection.instantiation.filter_config import FilterConfig
from semconstmining.selection.instantiation.recommendation_config import RecommendationConfig

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    level=logging.INFO)

_logger = logging.getLogger(__name__)

violation_type_to_constraint = {
    0: {Template.EXISTENCE.templ_str,
        Template.PRECEDENCE.templ_str,
        Template.RESPONSE.templ_str,
        Template.SUCCESSION.templ_str,
        Template.ALTERNATE_RESPONSE.templ_str,
        Template.ALTERNATE_PRECEDENCE.templ_str,
        Template.ALTERNATE_SUCCESSION.templ_str,
        Template.EXACTLY.templ_str,
        Template.EXCLUSIVE_CHOICE.templ_str,
        Template.RESPONDED_EXISTENCE.templ_str
        },
    1: {Template.ALTERNATE_SUCCESSION.templ_str,
        Template.ALTERNATE_RESPONSE.templ_str,
        Template.ALTERNATE_PRECEDENCE.templ_str,
        Template.EXACTLY.templ_str,
        Template.ABSENCE.templ_str,
        Template.EXCLUSIVE_CHOICE.templ_str,
        },
    2: {Template.SUCCESSION.templ_str,
        Template.ALTERNATE_SUCCESSION.templ_str,
        Template.RESPONSE.templ_str,
        Template.ALTERNATE_RESPONSE.templ_str,
        Template.PRECEDENCE.templ_str,
        Template.ALTERNATE_PRECEDENCE.templ_str
        },
    3: {Template.ABSENCE.templ_str}
}

constraint_to_violation_types = {
    Template.EXISTENCE.templ_str: {0},
    Template.RESPONDED_EXISTENCE.templ_str: {0},
    Template.PRECEDENCE.templ_str: {0},
    Template.RESPONSE.templ_str: {0},
    Template.SUCCESSION.templ_str: {0},
    Template.ALTERNATE_RESPONSE.templ_str: {0, 1},
    Template.ALTERNATE_PRECEDENCE.templ_str: {0, 1},
    Template.ALTERNATE_SUCCESSION.templ_str: {0, 1, 2},
    Template.EXACTLY.templ_str: {0, 1},
    Template.ABSENCE.templ_str: {1, 3},
    Template.EXCLUSIVE_CHOICE.templ_str: {0, 1}
}

violation_number_to_type = {
    0: {"remove"},
    1: {"insert"},
    2: {"swap"},
    3: {"resource"}
}


# TODO add attribute to event log to indicate whether it is noisy or not + type of noise
def _get_event_classes(log):
    classes = set()
    for trace in log:
        for event in trace:
            classes.add(event["concept:name"])
    return classes


def _remove_event(trace: Trace):
    del_index = random.randint(0, len(trace) - 1)
    trace2 = Trace()
    for i in range(0, len(trace)):
        if i != del_index:
            trace2.append(trace[i])
    trace2.attributes[conf.VIOLATION_TYPE] = 0
    return trace2


def _insert_event(trace: Trace, tasks):
    ins_index = random.randint(0, len(trace))
    task = random.choice(list(tasks))
    e = Event()
    e["concept:name"] = task
    e[conf.VIOLATION_TYPE] = 1
    trace.insert(ins_index, e)
    trace.attributes[conf.VIOLATION_TYPE] = 1
    return trace


def _swap_events(trace: Trace):
    if len(trace) == 1:
        return trace
    indices = list(range(len(trace)))
    index1 = random.choice(indices)
    indices.remove(index1)
    index2 = random.choice(indices)
    trace2 = Trace()
    for i in range(len(trace)):
        if i == index1:
            trace2.append(trace[index2])
        elif i == index2:
            trace2.append(trace[index1])
        else:
            trace2.append(trace[i])
    trace2.attributes[conf.VIOLATION_TYPE] = 2
    return trace2


def _swap_resources(trace: Trace, resources: dict):
    ins_index = random.randint(0, len(trace) - 1)
    resources_to_pick_from = set()
    e = trace[ins_index]
    task = e[conf.XES_NAME]
    for resources, tasks in resources.items():
        if task not in tasks:
            resources_to_pick_from.add(resources)
    if len(resources_to_pick_from) == 0:
        return trace
    resource = random.choice(list(resources_to_pick_from))
    e[conf.XES_ROLE] = resource
    e[conf.VIOLATION_TYPE] = 3
    trace.attributes[conf.VIOLATION_TYPE] = 3
    return trace


def insert_noise(row, noisy_trace_prob, noisy_event_prob, log_size, resource_handler):
    if len(row[conf.LOG]) < log_size:
        # add additional traces until desired log size reached
        log_cpy = EventLog()
        for i in range(0, log_size):
            log_cpy.append(deepcopy(row[conf.LOG][i % len(row[conf.LOG])]))
        log = log_cpy
    else:
        log = deepcopy(row[conf.LOG])
    classes = _get_event_classes(row[conf.LOG])
    log_new = EventLog()
    for idx, trace in enumerate(log):
        if len(trace) > 0:
            trace_cpy = deepcopy(trace)
            trace_cpy.attributes[conf.XES_NAME] = "Case " + str(idx)
            # check if trace makes random selection
            if random.random() <= noisy_trace_prob:
                insert_more_noise = True
                while insert_more_noise:
                    # randomly select which kind of noise to insert
                    if row[conf.MODEL_ID] in resource_handler.components.task_per_resource_per_model:
                        noise_type = random.randint(0, 3)
                    else:
                        noise_type = random.randint(0, 2)
                    if noise_type == 0:
                        _remove_event(trace_cpy)
                    if noise_type == 1:
                        _insert_event(trace_cpy, classes)
                    if noise_type == 2:
                        _swap_events(trace_cpy)
                    if noise_type == 3:
                        _swap_resources(trace_cpy,
                                        resource_handler.components.task_per_resource_per_model[row[conf.MODEL_ID]])
                    # flip coin to see if more noise will be inserted
                    insert_more_noise = (random.random() <= noisy_event_prob)
            if len(trace_cpy) > 1:
                log_new.append(trace_cpy)
    _logger.debug("Inserted noise into " + row[conf.MODEL_ID])
    return log_new


def add_role_info(row, resource_handler):
    log = EventLog()
    if row[conf.MODEL_ID] not in resource_handler.components.task_per_resource_per_model:
        return row[conf.LOG]
    resources = resource_handler.components.task_per_resource_per_model[row[resource_handler.config.MODEL_ID]]
    for trace in row[conf.LOG]:
        new_trace = Trace()
        for att in trace.attributes:
            new_trace.attributes[att] = trace.attributes[att]
        for event in trace:
            if event["concept:name"] in conf.TERMS_FOR_MISSING:
                continue
            for resource, tasks in resources.items():
                if event["concept:name"] in tasks:
                    event[conf.XES_ROLE] = resource
                    break
            if conf.XES_ROLE not in event:
                event[conf.XES_ROLE] = "unknown"
            new_trace.append(event)
        log.append(new_trace)
    return log


def load_or_generate_logs_from_sap_sam(resource_handler):
    if exists(conf.DATA_EVAL / "sap_sam_noisy.pkl"):
        df_results = pd.read_pickle(conf.DATA_EVAL / "sap_sam_noisy.pkl")
        return df_results
    logs = resource_handler.bpmn_logs[resource_handler.bpmn_logs[conf.LOG].notna()]
    logs = logs[logs.apply(lambda row: len(row[conf.LOG]) > 0, axis=1)]
    # We do the following to get the task to role mapping
    me = ModelExtractor(conf, resource_handler)
    me.get_perspectives_from_models()
    logs[conf.LOG] = logs.apply(lambda row: add_role_info(row, resource_handler), axis=1)
    logs[conf.NOISY_LOG] = None
    df_results = []
    split_df = np.array_split(logs, 100)
    start = time.time()
    for i, df in enumerate(split_df):
        file_name = "sap_sam_noisy_" + str(i) + ".pkl"
        if exists(conf.DATA_EVAL / file_name):
            df = pd.read_pickle(conf.DATA_EVAL / file_name)
        else:
            # Now we expand and add noise to the logs
            df[conf.NOISY_LOG] = df.apply(lambda row: insert_noise(row, NOISY_TRACE_PROB,
                                                           NOISY_EVENT_PROB, LOG_SIZE,
                                                           resource_handler), axis=1)
            df.to_pickle(conf.DATA_EVAL / file_name)
            _logger.info(f"Saved noisy logs to {conf.DATA_EVAL / file_name}")
        df_results.append(df)
    stop = time.time()
    completed_in = round(stop - start, 2)
    _logger.info(f"Completed in {completed_in} seconds")
    noisy_logs = pd.concat(df_results)
    # Now we store the noisy logs
    noisy_logs.to_pickle(conf.DATA_EVAL / "sap_sam_noisy.pkl")
    for (dir_path, dir_names, filenames) in os.walk(conf.DATA_EVAL):
        for filename in filenames:
            if filename != "sap_sam_noisy.pkl":
                os.remove(dir_path + "/" + filename)
    return noisy_logs


def get_violation_type(constraint):
    return 0


def evaluate_single_run(config_index, log_id, true_violations_per_type, violations_per_type):
    tp = 0
    fp = 0
    fn = 0
    for const_type, violations in violations_per_type.items():
        if const_type not in true_violations_per_type:
            fp += len(violations)
        else:
            for violation in violations:
                if violation not in true_violations_per_type[const_type]:
                    fp += 1
                else:
                    tp += 1
    for const_type, violations in true_violations_per_type.items():
        if const_type not in violations_per_type:
            fn += len(violations)
        else:
            for violation in violations:
                if violation not in violations_per_type[const_type]:
                    fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    return {"config": config_index, "log_id": log_id, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def run_configuration(config_index, constraints, logs, base_const):
    eval_results = []
    for idx, log_row in logs.iterrows():
        # Log-specific constraint recommendation
        name = resource_handler.bpmn_models.loc[log_row[conf.MODEL_ID]][conf.NAME]
        #log = pm4py.convert_to_dataframe(log_row[conf.LOG])
        noisy_log = pm4py.convert_to_dataframe(log_row[conf.NOISY_LOG])
        constraints = compute_relevance_for_log(conf, constraints, nlp_helper, resource_handler, name, noisy_log,
                                                precompute_all_sims=True)
        recommended_constraints = recommend_constraints_for_log(conf, rec_config, constraints, nlp_helper, name, noisy_log)
        consistency_checker = ConsistencyChecker(conf)
        inconsistent_subsets = consistency_checker.check_consistency(recommended_constraints)
        if len(inconsistent_subsets) > 0:
            consistent_recommended_constraints = consistency_checker.make_set_consistent_max_relevance(recommended_constraints,
                                                                                                       inconsistent_subsets)
        else:
            consistent_recommended_constraints = recommended_constraints
        true_constraints = base_const[base_const[conf.MODEL_ID] == log_row[conf.MODEL_ID]]
        violations_per_type = check_constraints(conf, log_row[conf.NAME], consistent_recommended_constraints, nlp_helper, noisy_log)
        true_violations_per_type = check_constraints(conf, log_row[conf.NAME], true_constraints, nlp_helper, noisy_log)
        eval_result = evaluate_single_run(config_index, log_row[conf.MODEL_ID], true_violations_per_type,
                                          violations_per_type)
        eval_results.append(eval_result)
    return eval_results


conf = Config(Path(__file__).parents[2].resolve(), "sap_sam_filtered")

eval_configurations = [
    (RecommendationConfig(config=conf, frequency_weight=0.5, semantic_weight=0.5, top_k=1000),
     FilterConfig(config=conf))
]

LOG_SIZE = 100
NOISY_TRACE_PROB = 0.5
NOISY_EVENT_PROB = 0.5

if __name__ == '__main__':
    _logger.info("Loading data")
    nlp_helper = NlpHelper(conf)
    resource_handler = get_resource_handler(conf, nlp_helper)
    base_constraints = ExtractionHandler(conf, resource_handler).get_all_observations()
    _logger.info("Loading constraints")
    all_constraints = get_or_mine_constraints(conf, resource_handler)
    _logger.info("Loading generality scores")
    all_constraints = all_constraints.drop(columns=[conf.LABEL_BASED_GENERALITY, conf.OBJECT_BASED_GENERALITY])
    all_constraints = get_context_sim_computer(conf, all_constraints, nlp_helper, resource_handler,
                                               precompute_all_sims=True).constraints
    _logger.info("Loading logs")
    noisy_logs = load_or_generate_logs_from_sap_sam(resource_handler)
    noisy_logs = noisy_logs[noisy_logs[conf.LOG].apply(lambda x: len(x) > 0)]
    _logger.info("Loaded logs")
    # Now we split the logs into train and test
    log_ids = list(noisy_logs[conf.MODEL_ID].unique())
    _logger.info(f"Unique model ids: {len(log_ids)}")
    _logger.info("Starting evaluation")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    average_results_per_config = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(log_ids)):
        fold_results = []
        train_ids = [log_ids[i] for i in train_idx]
        test_ids = [log_ids[i] for i in test_idx]
        constraint_ids = base_constraints[~(base_constraints[conf.MODEL_ID].isin(test_ids))].index.unique()
        _logger.info(f"Starting fold {fold}")
        train_constraints = all_constraints[all_constraints[conf.RECORD_ID].isin(constraint_ids)]
        fold_test_logs = noisy_logs[noisy_logs[conf.MODEL_ID].isin(test_ids)]
        for config_idx, eval_config in enumerate(eval_configurations):
            rec_config = eval_config[0]
            filt_config = eval_config[1]
            const_filter = ConstraintFilter(conf, filt_config, resource_handler)
            filtered_constraints = const_filter.filter_constraints(train_constraints)
            config_results = run_configuration(config_idx, filtered_constraints, fold_test_logs, base_constraints)
            fold_results.extend(config_results)
        fold_results_df = pd.DataFrame.from_records(fold_results)
        average_results_per_config.append(fold_results_df.groupby("config").mean())
        fold_results_df.to_csv(conf.DATA_EVAL / f"fold_{fold}.csv", index=False)
    average_results_df = pd.concat(average_results_per_config)
    average_results_df.to_csv(conf.DATA_EVAL / "average.csv", index=False)
    _logger.info("Finished evaluation")