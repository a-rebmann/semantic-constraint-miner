import logging
import multiprocessing
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
from tqdm import tqdm

from semconstmining.config import Config
from sklearn.model_selection import KFold, train_test_split

from semconstmining.declare.enums import Template
from semconstmining.log.loghandler import LogHandler
from semconstmining.log.loginfo import LogInfo
from semconstmining.main import get_resource_handler, get_or_mine_constraints, \
    compute_relevance_for_log, check_constraints, get_parts_of_constraints
from semconstmining.mining.aggregation.subsumptionanalyzer import SubsumptionAnalyzer
from semconstmining.mining.extraction.extractionhandler import ExtractionHandler
from semconstmining.mining.extraction.modelextractor import ModelExtractor
from semconstmining.parsing.label_parser.nlp_helper import NlpHelper
from semconstmining.selection.consistency.consistency import ConsistencyChecker
from semconstmining.selection.instantiation.constraintfilter import ConstraintFilter
from semconstmining.selection.instantiation.constraintfitter import ConstraintFitter
from semconstmining.selection.instantiation.constraintrecommender import ConstraintRecommender
from semconstmining.selection.instantiation.filter_config import FilterConfig
from semconstmining.selection.instantiation.recommendation_config import RecommendationConfig
from semconstmining.declare.parsers.decl_parser import parse_single_constraint

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    level=logging.INFO)

_logger = logging.getLogger(__name__)


def _get_event_classes(log):
    classes = set()
    for trace in log:
        for event in trace:
            classes.add(event["concept:name"])
    return classes


def _remove_event(trace: Trace):
    if len(trace) <= 1:
        return trace
    del_index = random.randint(0, len(trace) - 1)
    trace2 = Trace()
    for i in range(0, len(trace)):
        if i != del_index:
            trace2.append(trace[i])
    trace2.attributes[conf.VIOLATION_TYPE] = 0
    for att in trace.attributes:
        trace2.attributes[att] = trace.attributes[att]
    return trace2


def _insert_event(trace: Trace, tasks):
    if len(trace) <= 1:
        return trace
    ins_index = random.randint(0, len(trace))
    task = random.choice(list(tasks))
    e = Event()
    e["concept:name"] = task
    e[conf.VIOLATION_TYPE] = 1
    trace.insert(ins_index, e)
    trace.attributes[conf.VIOLATION_TYPE] = 1
    return trace


def _swap_events(trace: Trace):
    if len(trace) <= 1:
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
    for att in trace.attributes:
        trace2.attributes[att] = trace.attributes[att]
    return trace2


def _swap_resources(trace: Trace, resources: dict):
    if len(trace) < 1:
        return trace
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
    for att in trace.attributes:
        trace.attributes[att] = trace.attributes[att]
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
                        trace_cpy = _remove_event(trace_cpy)
                    if noise_type == 1:
                        trace_cpy = _insert_event(trace_cpy, classes)
                    if noise_type == 2:
                        trace_cpy = _swap_events(trace_cpy)
                    if noise_type == 3:
                        trace_cpy = _swap_resources(trace_cpy,
                                                    resource_handler.components.task_per_resource_per_model[
                                                        row[conf.MODEL_ID]])
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
    if exists(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "_noisy.pkl")):
        df_results = pd.read_pickle(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "_noisy.pkl"))
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
        file_name = (conf.MODEL_COLLECTION + "_noisy_" + str(i) + ".pkl")
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
    noisy_logs.to_pickle(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "_noisy.pkl"))
    for (dir_path, dir_names, filenames) in os.walk(conf.DATA_EVAL):
        for filename in filenames:
            if "noisy" in filename and not filename.endswith("noisy.pkl"):
                os.remove(dir_path + "/" + filename)
    return noisy_logs


def count_tp_fp_fn_strict(config, true_violations_per_type, violations_per_type, base_const):
    tp = 0
    fp = 0
    fn = 0
    const_type_tp = {config.ACTIVITY: 0, config.RESOURCE: 0, config.OBJECT: 0, config.MULTI_OBJECT: 0}
    const_type_fp = {config.ACTIVITY: 0, config.RESOURCE: 0, config.OBJECT: 0, config.MULTI_OBJECT: 0}
    const_type_fn = {config.ACTIVITY: 0, config.RESOURCE: 0, config.OBJECT: 0, config.MULTI_OBJECT: 0}
    for const_type, violations in violations_per_type.items():
        if const_type == config.OBJECT:
            for obj_type, obj_violations in violations.items():
                for case, case_violations in obj_violations.items():
                    for violation in case_violations:
                        if obj_type not in true_violations_per_type[const_type]:
                            fp += 1
                            const_type_fp[const_type] += 1
                        elif violation not in true_violations_per_type[const_type][obj_type][case]:
                            fp += 1
                            const_type_fp[const_type] += 1
                        else:
                            tp += 1
                            const_type_tp[const_type] += 1
        else:
            for case, case_violations in violations.items():
                for violation in case_violations:
                    if violation not in true_violations_per_type[const_type][case]:
                        fp += 1
                        const_type_fp[const_type] += 1
                    else:
                        tp += 1
                        const_type_tp[const_type] += 1
    for const_type, violations in true_violations_per_type.items():
        if const_type == config.OBJECT:
            for obj_type, obj_violations in violations.items():
                for case, case_violations in obj_violations.items():
                    for violation in case_violations:
                        if obj_type not in violations_per_type[const_type]:
                            fn += 1
                            const_type_fn[const_type] += 1
                        elif violation not in violations_per_type[const_type][obj_type][case]:
                            if violation in base_const:
                                fn += 1
                                const_type_fn[const_type] += 1
        else:
            for case, case_violations in violations.items():
                for violation in case_violations:
                    if violation not in violations_per_type[const_type][case]:
                        if violation in base_const:
                            fn += 1
                            const_type_fn[const_type] += 1
    return tp, fp, fn, const_type_tp, const_type_fp, const_type_fn


def count_tp_fp_fn(config, y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    const_type_tp = {config.ACTIVITY: 0, config.RESOURCE: 0, config.OBJECT: 0, config.MULTI_OBJECT: 0}
    const_type_fp = {config.ACTIVITY: 0, config.RESOURCE: 0, config.OBJECT: 0, config.MULTI_OBJECT: 0}
    const_type_fn = {config.ACTIVITY: 0, config.RESOURCE: 0, config.OBJECT: 0, config.MULTI_OBJECT: 0}
    for const_type, violations in y_pred.items():
        if const_type == config.OBJECT:
            for obj_type, obj_violations in violations.items():
                for case, case_violations in obj_violations.items():
                    for violation, consts in case_violations.items():
                        if obj_type not in y_true[const_type]:
                            fp += 1
                            const_type_fp[const_type] += 1
                        elif violation not in y_true[const_type][obj_type][case]:
                            fp += 1
                            const_type_fp[const_type] += 1
                        else:
                            tp += 1
                            const_type_tp[const_type] += 1
        else:
            for case, case_violations in violations.items():
                for violation, consts in case_violations.items():
                    if violation not in y_true[const_type][case]:
                        fp += 1
                        const_type_fp[const_type] += 1
                    else:
                        tp += 1
                        const_type_tp[const_type] += 1
    for const_type, violations in y_true.items():
        if const_type == config.OBJECT:
            for obj_type, obj_violations in violations.items():
                for case, case_violations in obj_violations.items():
                    for violation, consts in case_violations.items():
                        if obj_type not in y_pred[const_type]:
                            fn += 1
                            const_type_fn[const_type] += 1
                        elif violation not in y_pred[const_type][obj_type][case]:
                            fn += 1
                            const_type_fn[const_type] += 1
        else:
            for case, case_violations in violations.items():
                for violation, consts in case_violations.items():
                    if violation not in y_pred[const_type][case]:
                        fn += 1
                        const_type_fn[const_type] += 1
    return tp, fp, fn, const_type_tp, const_type_fp, const_type_fn




def evaluate_single_run(config, config_index, log_id, true_violations_per_type, violations_per_type, run_time,
                        base_const, y_true, y_pred):
    #tp, fp, fn, const_type_tp, const_type_fp, const_type_fn = count_tp_fp_fn_strict(config, true_violations_per_type, violations_per_type, base_const)
    tp, fp, fn, const_type_tp, const_type_fp, const_type_fn = count_tp_fp_fn(config, y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    support = tp + fn
    const_type_precision = {
        const_type: const_type_tp[const_type] / (const_type_tp[const_type] + const_type_fp[const_type]) if
        const_type_tp[const_type] + const_type_fp[const_type] > 0 else 1.0 for const_type in const_type_tp.keys()}
    const_type_recall = {
        const_type: const_type_tp[const_type] / (const_type_tp[const_type] + const_type_fn[const_type]) if
        const_type_tp[const_type] + const_type_fn[const_type] > 0 else 1.0 for const_type in const_type_tp.keys()}
    const_type_f1 = {const_type: 2 * (const_type_precision[const_type] * const_type_recall[const_type]) / (
            const_type_precision[const_type] + const_type_recall[const_type]) if const_type_precision[const_type] +
                                                                                 const_type_recall[
                                                                                     const_type] > 0 else 0 for
                     const_type in const_type_tp.keys()}
    const_type_support = {const_type: const_type_tp[const_type] + const_type_fn[const_type] for const_type in
                          const_type_tp.keys()}
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    res = {"config": config_index, "log_id": log_id, "tp": tp, "fp": fp, "fn": fn, "precision": precision,
           "recall": recall, "f1": f1, "support": support, "run_time": run_time}
    for const_type in const_type_tp.keys():
        res[f"{const_type}_precision"] = const_type_precision[const_type]
        res[f"{const_type}_recall"] = const_type_recall[const_type]
        res[f"{const_type}_f1"] = const_type_f1[const_type]
        res[f"{const_type}_support"] = const_type_support[const_type]
        res[f"{const_type}_tp"] = const_type_tp[const_type]
        res[f"{const_type}_fp"] = const_type_fp[const_type]
        res[f"{const_type}_fn"] = const_type_fn[const_type]
    return res


def group_violations_by_params(config, violations_per_type):
    grouped_violations = {}
    for const_type, violations in violations_per_type.items():
        grouped_violations[const_type] = {}
        if const_type == config.OBJECT:
            for obj_type, obj_violations in violations.items():
                if obj_type not in grouped_violations[const_type]:
                    grouped_violations[const_type][obj_type] = {}
                for case, case_violations in obj_violations.items():
                    if case not in grouped_violations[const_type][obj_type]:
                        grouped_violations[const_type][obj_type][case] = {}
                    for violation in case_violations:
                        const = parse_single_constraint(violation)
                        if const["template"].is_binary:
                            left = const["activities"][0]
                            right = const["activities"][1]
                            if left in grouped_violations[const_type][obj_type][case]:
                                grouped_violations[const_type][obj_type][case][left].append(violation)
                            else:
                                grouped_violations[const_type][obj_type][case][left] = [violation]
                            if right in grouped_violations[const_type][obj_type][case]:
                                grouped_violations[const_type][obj_type][case][right].append(violation)
                            else:
                                grouped_violations[const_type][obj_type][case][right] = [violation]
                        else:
                            left = const["activities"][0]
                            if left in grouped_violations[const_type][obj_type][case]:
                                grouped_violations[const_type][obj_type][case][left].append(violation)
                            else:
                                grouped_violations[const_type][obj_type][case][left] = [violation]

        else:
            for case, case_violations in violations.items():
                if case not in grouped_violations[const_type]:
                    grouped_violations[const_type][case] = {}
                for violation in case_violations:
                    const = parse_single_constraint(violation)
                    if const["template"].is_binary:
                        left = const["activities"][0]
                        right = const["activities"][1]
                        if left in grouped_violations[const_type][case]:
                            grouped_violations[const_type][case][left].append(violation)
                        else:
                            grouped_violations[const_type][case][left] = [violation]
                        if right in grouped_violations[const_type][case]:
                            grouped_violations[const_type][case][right].append(violation)
                        else:
                            grouped_violations[const_type][case][right] = [violation]
                    else:
                        left = const["activities"][0]
                        if left in grouped_violations[const_type][case]:
                            grouped_violations[const_type][case][left].append(violation)
                        else:
                            grouped_violations[const_type][case][left] = [violation]
    return grouped_violations


def run_eval_for_log(config, const, base_const, df, config_index, nlp, r_config):
    eval_results = []
    num_tested = 0
    for idx, log_row in df.iterrows():
        name = log_row[config.NAME]
        noisy_log = pm4py.convert_to_dataframe(log_row[config.NOISY_LOG])
        if len(noisy_log) == 0 or len(noisy_log[config.XES_NAME].unique()) < 3:
            _logger.warning(f"Empty log {name}")
            continue
        start = time.time()
        constraints = compute_relevance_for_log(config, const, nlp, name, noisy_log,
                                                precompute=True, store_sims=True, log_id=log_row[config.MODEL_ID])
        #recommended_constraints = recommend_constraints_for_log(config, r_config, constraints, nlp, name, noisy_log)
        lh = LogHandler(config)
        lh.log = noisy_log
        labels = list(noisy_log[config.XES_NAME].unique())
        log_info = LogInfo(nlp, labels, [name])
        recommender = ConstraintRecommender(config, r_config, log_info)
        recommended_constraints = recommender.recommend(constraints)
        constraint_fitter = ConstraintFitter(config, name, recommended_constraints)
        fitted_constraints = constraint_fitter.fit_constraints(r_config.relevance_thresh)
        fitted_constraints = recommender.recommend_by_activation(fitted_constraints)
        consistency_checker = ConsistencyChecker(config)
        inconsistent_subsets = consistency_checker.check_consistency(recommended_constraints) #TODO uncomment when consistency checker is ready
        if len(inconsistent_subsets) > 0:
            consistent_recommended_constraints = consistency_checker.make_set_consistent_max_relevance(
                fitted_constraints,
                inconsistent_subsets)
        else:
            consistent_recommended_constraints = fitted_constraints
        true_constraints = base_const[base_const[config.MODEL_ID] == log_row[config.MODEL_ID]]
        violations_per_type = check_constraints(config, log_row[config.NAME], consistent_recommended_constraints, nlp,
                                                noisy_log)
        _logger.info(f"Finished checking constraints for {name}")
        end = time.time()
        run_time = round(end - start, 2)
        true_violations_per_type = check_constraints(config, log_row[config.NAME], true_constraints, nlp, noisy_log)
        _logger.info(f"Finished checking true constraints for {name}")

        # missing, superfluous, order
        # group violations by parameters
        y_true = group_violations_by_params(config, true_violations_per_type)
        y_pred = group_violations_by_params(config, violations_per_type)
        _logger.info(f"Finished grouping violations for {name}")
        eval_result = evaluate_single_run(config, config_index, log_row[config.MODEL_ID], true_violations_per_type,
                                          violations_per_type, run_time,
                                          set(true_constraints[~true_constraints[config.REDUNDANT]][config.CONSTRAINT_STR].values),
                                          y_true, y_pred)
        eval_result["inconsistent_subsets"] = len(inconsistent_subsets)
        eval_result["num_recommended_constraints"] = len(recommended_constraints)
        eval_result["num_fitted_constraints"] = len(fitted_constraints)
        eval_result["num_consistent_recommended_constraints"] = len(consistent_recommended_constraints)
        eval_results.append(eval_result)
        num_tested += 1
        if num_tested == 20:
            break
    return eval_results


def run_configuration(config_index, constraints, logs, base_const, r_config, nlp, multi_process=False):
    eval_results = []
    if multi_process:
        from concurrent.futures import ProcessPoolExecutor
        num_processes = multiprocessing.cpu_count() - 10
        _logger.info("Number of processes: {}".format(num_processes))
        split_df = np.array_split(logs, num_processes)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for batch in tqdm(split_df):
                r_config_copy = deepcopy(r_config)
                const_copy = deepcopy(constraints)
                base_const_copy = deepcopy(base_const)
                nlp_helper_copy = deepcopy(nlp)
                config_copy = deepcopy(conf)
                futures.append(executor.submit(run_eval_for_log, config_copy, const_copy, base_const_copy, batch,
                                               config_index, nlp_helper_copy, r_config_copy))
            _logger.info("Number of futures: {}".format(len(futures)))
            for future in tqdm(futures):
                eval_results.extend(future.result())
    else:
        eval_results.extend(run_eval_for_log(conf, constraints, base_const, logs, config_index, nlp, r_config))
    return eval_results


def run_train_test_split(log_ids, noisy_logs, base_constraints, all_constraints, configs, resource_handler,
                         test_ids, nlp):
    run_results = []
    # use all constraint that are not in the test set
    constraint_ids = base_constraints[~(base_constraints[conf.MODEL_ID].isin(test_ids))].index.unique()
    train_constraints = all_constraints[all_constraints[conf.RECORD_ID].isin(constraint_ids)]
    #fold_test_logs = noisy_logs[noisy_logs[conf.MODEL_ID]=="3b110169afda4c1c87e89d7617bb77ef"] TODO
    fold_test_logs = noisy_logs[noisy_logs[conf.MODEL_ID].isin(test_ids)]
    for config_idx, eval_config in enumerate(configs):
        rec_config = eval_config[0]
        filt_config = eval_config[1]
        const_filter = ConstraintFilter(conf, filt_config, resource_handler)
        filtered_constraints = const_filter.filter_constraints(train_constraints)
        config_results = run_configuration(config_idx, filtered_constraints, fold_test_logs, base_constraints,
                                           rec_config, nlp,
                                           multi_process=MULTI_PROCESS)
        run_results.extend(config_results)
    return run_results


def evaluate(run_k_fold=False):
    _logger.info("Loading data")
    nlp_helper = NlpHelper(conf)
    resource_handler = get_resource_handler(conf, nlp_helper)
    if not exists(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "_eval_constraints.pkl")):
        base_constraints = ExtractionHandler(conf, resource_handler).get_all_observations()
        base_constraints[conf.SUPPORT] = 1
        grouped = base_constraints.groupby(conf.MODEL_ID)
        dfs = []
        for group_id, group in tqdm(grouped):
            subsumption_analyzer = SubsumptionAnalyzer(conf, group)
            subsumption_analyzer.check_refinement()
            subsumption_analyzer.check_subsumption()
            subsumption_analyzer.check_equal()
            dfs.append(subsumption_analyzer.constraints)
        base_constraints = pd.concat(dfs)
        base_constraints.to_pickle(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "_eval_constraints.pkl"))
    else:
        base_constraints = pd.read_pickle(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "_eval_constraints.pkl"))
    base_constraints = base_constraints[~(base_constraints[conf.TEMPLATE].isin(conf.CONSTRAINT_TYPES_TO_IGNORE))]
    _logger.info("Loading constraints")
    all_constraints = get_or_mine_constraints(conf, resource_handler, min_support=1)
    # _logger.info("Loading generality scores") # Not part of this version
    # all_constraints = get_context_sim_computer(conf, all_constraints, nlp_helper, resource_handler).constraints
    nlp_helper.pre_compute_embeddings(sentences=get_parts_of_constraints(conf, all_constraints))
    _logger.info("Loading logs")
    noisy_logs = load_or_generate_logs_from_sap_sam(resource_handler)
    noisy_logs = noisy_logs[noisy_logs[conf.LOG].apply(lambda x: len(x) > 0)]
    _logger.info("Loaded logs")
    # Now we split the logs into train and test
    log_ids = list(noisy_logs[conf.MODEL_ID].unique())
    _logger.info(f"Unique model ids: {len(log_ids)}")
    _logger.info("Starting evaluation")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # average_results_per_config = []
    if run_k_fold:
        all_results = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(log_ids)):
            _logger.info(f"Starting fold {fold}")
            test_ids = [log_ids[i] for i in test_idx]
            fold_results = run_train_test_split(log_ids, noisy_logs, base_constraints, all_constraints,
                                                eval_configurations, resource_handler, test_ids, nlp_helper)
            fold_results_df = pd.DataFrame.from_records(fold_results)
            fold_results_df["fold"] = fold
            all_results.append(fold_results_df)
            # average_results_per_config.append(fold_results_df.groupby("config").mean())
        run_results_df = pd.concat(all_results)
        run_results_df.to_csv(conf.DATA_EVAL / f"folds_{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}.csv",
                              index=False)
    else:
        train_ids, test_ids = train_test_split(
            log_ids, test_size=0.3, random_state=42
        )
        run_results = run_train_test_split(log_ids, noisy_logs, base_constraints, all_constraints,
                                           eval_configurations, resource_handler, test_ids, nlp_helper)
        run_results = [r for r in run_results if r["support"] > 0]
        run_results_df = pd.DataFrame.from_records(run_results)
        run_results_df["fold"] = 0
        run_results_df.to_csv(conf.DATA_EVAL / f"run_{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}.csv",
                              index=False)
        # compute the weighted average per configuration
    const_types = [conf.ACTIVITY, conf.RESOURCE, conf.OBJECT, conf.MULTI_OBJECT]
    avg_frames = []
    for config, group in run_results_df.groupby("config"):
        config_results = {}
        avg_precision = group.loc[group["support"] > 0, "precision"].mean()
        avg_recall = group.loc[group["support"] > 0, "recall"].mean()
        avg_f1 = group.loc[group["support"] > 0, "f1"].mean()
        avg_support = group.loc[group["support"] > 0, "support"].mean()
        total_support = group["support"].sum()
        avg_run_time = group["run_time"].mean()
        tp_avg = group["tp"].mean()
        fp_avg = group["fp"].mean()
        fn_avg = group["fn"].mean()
        tp_total = group["tp"].sum()
        fp_total = group["fp"].sum()
        fn_total = group["fn"].sum()
        config_results["config"] = config
        config_results["avg_precision"] = avg_precision
        config_results["avg_recall"] = avg_recall
        config_results["avg_f1"] = avg_f1
        config_results["avg_support"] = avg_support
        config_results["total_support"] = total_support
        config_results["avg_run_time"] = avg_run_time
        config_results["tp_avg"] = tp_avg
        config_results["fp_avg"] = fp_avg
        config_results["fn_avg"] = fn_avg
        config_results["tp_total"] = tp_total
        config_results["fp_total"] = fp_total
        config_results["fn_total"] = fn_total
        per_const_dict = {}
        for const_type in const_types:
            const_type_avg_precision = group.loc[group[f"{const_type}_support"] > 0, f"{const_type}_precision"].mean()
            const_type_avg_recall = group.loc[group[f"{const_type}_support"] > 0, f"{const_type}_recall"].mean()
            const_type_avg_f1 = group.loc[group[f"{const_type}_support"] > 0, f"{const_type}_f1"].mean()
            const_type_avg_support = group.loc[group[f"{const_type}_support"] > 0, f"{const_type}_support"].mean()
            const_type_total_support = group[f"{const_type}_support"].sum()
            const_tp_avg = group[f"{const_type}_tp"].mean()
            const_fp_avg = group[f"{const_type}_fp"].mean()
            const_fn_avg = group[f"{const_type}_fn"].mean()
            const_tp_total = group[f"{const_type}_tp"].sum()
            const_fp_total = group[f"{const_type}_fp"].sum()
            const_fn_total = group[f"{const_type}_fn"].sum()
            micro_precision = group[f"{const_type}_tp"].sum() / (
                    group[f"{const_type}_tp"].sum() + group[f"{const_type}_fp"].sum()
            )
            micro_recall = group[f"{const_type}_tp"].sum() / (
                    group[f"{const_type}_tp"].sum() + group[f"{const_type}_fn"].sum()
            )
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
            per_const_dict |= {
                f"{const_type}_precision": const_type_avg_precision,
                f"{const_type}_recall": const_type_avg_recall,
                f"{const_type}_f1": const_type_avg_f1,
                f"{const_type}_support": const_type_avg_support,
                f"{const_type}_total_support": const_type_total_support,
                f"{const_type}_tp": const_tp_avg,
                f"{const_type}_fp": const_fp_avg,
                f"{const_type}_fn": const_fn_avg,
                f"{const_type}_total_tp": const_tp_total,
                f"{const_type}_total_fp": const_fp_total,
                f"{const_type}_total_fn": const_fn_total,
                f"{const_type}_micro_precision": micro_precision,
                f"{const_type}_micro_recall": micro_recall,
                f"{const_type}_micro_f1": micro_f1
            }
        config_results.update(per_const_dict)
        avg_frames.append(config_results)
    avg_results_df = pd.DataFrame.from_records(avg_frames)
    avg_results_df.to_csv(conf.DATA_EVAL / f"average_{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}.csv",
                          index=False)
    _logger.info("Finished evaluation")


conf = Config(Path(__file__).parents[2].resolve(), "semantic_sap_sam_filtered")
conf.CONSTRAINT_TYPES_TO_IGNORE.extend(conf.NEGATIVE_TEMPLATES)
conf.CONSTRAINT_TYPES_TO_IGNORE.remove(Template.NOT_CO_EXISTENCE.templ_str)
conf.CONSTRAINT_TYPES_TO_IGNORE.append(Template.INIT.templ_str)
conf.CONSTRAINT_TYPES_TO_IGNORE.append(Template.END.templ_str)

eval_configurations = [
    # (RecommendationConfig(config=conf, semantic_weight=0.5, relevance_thresh=0.5, top_k=1000),
    #  FilterConfig(config=conf)),
    (RecommendationConfig(config=conf, semantic_weight=0.9, relevance_thresh=0.5, top_k=250),
     FilterConfig(config=conf)),
    (RecommendationConfig(config=conf, semantic_weight=0.9, relevance_thresh=0.5, top_k=100),
     FilterConfig(config=conf)),
    (RecommendationConfig(config=conf, semantic_weight=0.9, relevance_thresh=0.5, top_k=50),
     FilterConfig(config=conf)),
    (RecommendationConfig(config=conf, semantic_weight=0.9, relevance_thresh=0.5, top_k=10),
     FilterConfig(config=conf))
]

LOG_SIZE = 100
NOISY_TRACE_PROB = 0.5
NOISY_EVENT_PROB = 0.5

MULTI_PROCESS = False


if __name__ == '__main__':
    evaluate()
