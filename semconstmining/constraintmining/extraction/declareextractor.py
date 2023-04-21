import logging
import uuid

import pandas as pd
from pm4py.objects.log.obj import EventLog, Trace, Event

from semconstmining.declare.parsers import parse_single_constraint
from semconstmining.parsing.conversion.petrinetanalysis import _is_relevant_label
from semconstmining.constraintmining.model.parsed_label import get_dummy
from semconstmining.declare.declare import Declare
from semconstmining.parsing.resource_handler import ResourceHandler

_logger = logging.getLogger(__name__)


def _get_constraint_template(constraint):
    return parse_single_constraint(constraint)["template"].templ_str if parse_single_constraint(constraint) is not \
                                                                        None else None


class DeclareExtractor:

    def __init__(self, config, resource_handler: ResourceHandler):
        self.config = config
        self.resource_handler = resource_handler

    def add_operands(self, res):
        for rec in res:
            if rec[self.config.OPERATOR_TYPE] == self.config.UNARY:
                op_l = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").strip()
                rec[self.config.LEFT_OPERAND] = op_l if op_l not in self.config.TERMS_FOR_MISSING else ""
            if rec[self.config.OPERATOR_TYPE] == self.config.BINARY:
                ops = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").split(",")
                op_l = ops[0].strip()
                op_r = ops[1].strip()
                rec[self.config.LEFT_OPERAND] = op_l if op_l not in self.config.TERMS_FOR_MISSING else ""
                rec[self.config.RIGHT_OPERAND] = op_r if op_r not in self.config.TERMS_FOR_MISSING else ""
        return res

    def get_object_constraints_flat(self, res, associations=None):
        res = [{self.config.RECORD_ID: str(uuid.uuid4()),
                self.config.LEVEL: self.config.OBJECT,
                self.config.OBJECT: bo,
                self.config.CONSTRAINT_STR: const,
                self.config.OPERATOR_TYPE: self.config.BINARY if any(
                    temp in const for temp in self.config.BINARY_TEMPLATES) else self.config.UNARY,
                self.config.DICTIONARY: associations[bo][const][self.config.DICTIONARY],
                self.config.DATA_OBJECT: associations[bo][const][self.config.DATA_OBJECT],
                self.config.TEMPLATE: _get_constraint_template(const)
                } for bo, consts in res.items() for const in consts]
        res = self.add_operands(res)
        for rec in res:
            if rec[self.config.OPERATOR_TYPE] == self.config.UNARY:
                rec[self.config.LEFT_OPERAND] = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace(
                    "|", "").strip()
            if rec[self.config.OPERATOR_TYPE] == self.config.BINARY:
                ops = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").split(",")
                rec[self.config.LEFT_OPERAND] = ops[0].strip()
                rec[self.config.RIGHT_OPERAND] = ops[1].strip()
        return res

    def get_multi_object_constraints_flat(self, res, associations=None):
        res = [{self.config.RECORD_ID: str(uuid.uuid4()),
                self.config.LEVEL: self.config.MULTI_OBJECT,
                self.config.OBJECT: "",
                self.config.CONSTRAINT_STR: const,
                self.config.OPERATOR_TYPE: self.config.BINARY if any(
                    temp in const for temp in self.config.BINARY_TEMPLATES) else self.config.UNARY,
                self.config.DICTIONARY: associations[const][self.config.DICTIONARY],
                self.config.DATA_OBJECT: associations[const][self.config.DATA_OBJECT],
                self.config.TEMPLATE: _get_constraint_template(const)
                } for const in res]
        for rec in res:
            if rec[self.config.OPERATOR_TYPE] == self.config.UNARY:
                rec[self.config.LEFT_OPERAND] = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace(
                    "|", "").strip()
            if rec[self.config.OPERATOR_TYPE] == self.config.BINARY:
                ops = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").split(",")
                rec[self.config.LEFT_OPERAND] = ops[0].strip()
                rec[self.config.RIGHT_OPERAND] = ops[1].strip()
        return res

    def extract_declare_from_logs(self):
        """
        Extract DECLARE-like constraints from log traces
        :return: a pandas dataframe with extracted DECLARE constraints
        """
        _logger.info("Extracting DECLARE constraints from played-out logs")
        #Discover regular declare constraints
        dfs_reg = [self.discover_declare_constraints(t) for t in
                    self.resource_handler.bpmn_logs.reset_index().itertuples()]

        # Discover action based constraints per object
        dfs_obj = [self.discover_object_based_declare_constraints(t) for t in
                   self.resource_handler.bpmn_logs.reset_index().itertuples()]
        # Discover multi-object constraints
        dfs_multi_obj = [self.discover_multi_object_declare_constraints(t) for t in
                         self.resource_handler.bpmn_logs.reset_index().itertuples()]

        # Combine all constraints that were extracted into a common dataframe
        dfs = [df for df in
               dfs_reg +  # regular declare constraints
               dfs_obj +  # object based constraints
               dfs_multi_obj  # multi-object constraints
               if df is not None]
        new_df = pd.concat(dfs).astype({self.config.LEVEL: "category"})
        return new_df

    def discover_multi_object_declare_constraints(self, row_tuple):
        if row_tuple.log is None:
            return None
        parsed_tasks = self.get_parsed_tasks(row_tuple.log, resource_handler=self.resource_handler)
        filtered_traces = self.get_filtered_traces(row_tuple.log, parsed_tasks=parsed_tasks)
        res = set()
        d4py = Declare(self.config)
        d4py.log = self.object_log_projection(filtered_traces)
        d4py.compute_frequent_itemsets(min_support=0.99, len_itemset=2)
        d4py.discovery(consider_vacuity=False, max_declare_cardinality=2, do_unary=False)
        individual_res, associations = d4py.filter_discovery(min_support=0.99)
        if any(len(y) > 0 for val in associations.values() for x in val.values() for y in x):
            #_logger.info(associations)
            pass
        res.update(const for const, checker_results in individual_res.items()
                   if "[]" not in const and "[none]" not in const
                   and ''.join([i for i in const.split("[")[0] if not i.isdigit()]) not in
                   self.config.CONSTRAINT_TYPES_TO_IGNORE)
        return (
            pd.DataFrame.from_records(self.get_multi_object_constraints_flat(res, associations)).assign(
                model_id=row_tuple.model_id).assign(
                model_name=row_tuple.name)
        )

    def discover_object_based_declare_constraints(self, row_tuple):
        if row_tuple.log is None:
            return None
        parsed_tasks = self.get_parsed_tasks(row_tuple.log, resource_handler=self.resource_handler)
        filtered_traces = self.get_filtered_traces(row_tuple.log, parsed_tasks=parsed_tasks)
        res = {}
        all_associations = {}
        bos = set([x.main_object for trace in filtered_traces for x in trace if
                   x.main_object not in self.config.TERMS_FOR_MISSING])
        # _logger.info(bos)
        for bo in bos:
            d4py = Declare(self.config)
            d4py.log = self.object_action_log_projection(bo, filtered_traces)
            d4py.compute_frequent_itemsets(min_support=0.99, len_itemset=2)
            d4py.discovery(consider_vacuity=False, max_declare_cardinality=2)
            individual_res, associations = d4py.filter_discovery(min_support=0.99)
            # print(individual_res)
            if bo not in res:
                res[bo] = set()
            res[bo].update(const for const, checker_results in individual_res.items() if "[]" not in const
                           and "[none]" not in const
                           and ''.join([i for i in const.split("[")[0] if not i.isdigit()]) not in self.config.CONSTRAINT_TYPES_TO_IGNORE)
            all_associations[bo] = associations
        return (
            pd.DataFrame.from_records(self.get_object_constraints_flat(res, all_associations)).assign(model_id=row_tuple.model_id).assign(
                model_name=row_tuple.name)
        )

    def discover_declare_constraints(self, row_tuple):
        if row_tuple.log is None:
            return None
        d4py = Declare(self.config)
        parsed_tasks = self.get_parsed_tasks(row_tuple.log, resource_handler=self.resource_handler)
        filtered_traces = self.get_filtered_traces(row_tuple.log, parsed_tasks=parsed_tasks, with_loops=True)
        d4py.log = self.clean_log_projection(filtered_traces)
        d4py.compute_frequent_itemsets(min_support=0.99, len_itemset=2)
        d4py.discovery(consider_vacuity=False, max_declare_cardinality=2)
        individual_res, associations = d4py.filter_discovery(min_support=0.99)
        res = {const for const, checker_results in individual_res.items() if "[]" not in const
               and "[none]" not in const
               and ''.join([i for i in const.split("[")[0] if not i.isdigit()]) not in self.config.CONSTRAINT_TYPES_TO_IGNORE}
        return (
            pd.DataFrame.from_records(self.get_constraints_flat(res, associations)).assign(model_id=row_tuple.model_id).assign(
                model_name=row_tuple.name)
        )

    def has_loop(self, trace):
        trace_labels = [x[self.config.XES_NAME] for x in trace]
        return len(trace_labels) > len(set(trace_labels))

    def get_parsed_tasks(self, log, resource_handler, only_relevant_labels=True):
        relevant_tasks = set([x[self.config.XES_NAME] for trace in log for x in trace if
                              _is_relevant_label(x[self.config.XES_NAME])]) if only_relevant_labels else set(
            [x[self.config.XES_NAME] for trace in log for x in trace])
        return {t: resource_handler.get_parsed_task(t) for t in relevant_tasks}

    def get_filtered_traces(self, log, parsed_tasks=None, with_loops=False):
        if parsed_tasks is not None:
            return [
                [parsed_tasks[e[self.config.XES_NAME]] if e[self.config.XES_NAME] in parsed_tasks else get_dummy(
                    self.config, e[self.config.XES_NAME], self.config.EN) for i, e in
                 enumerate(trace)] for trace in log if with_loops or not self.has_loop(trace)]
        else:
            return [[e[self.config.XES_NAME] for i, e in enumerate(trace)] for trace in log if
                    with_loops or not self.has_loop(trace)]

    def clean_log_projection(self, traces):
        """
        Same log, just with clean labels.
        """
        projection = EventLog()
        if traces is None:
            raise RuntimeError("You must load a log before.")
        for i, trace in enumerate(traces):
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = str(i)
            for parsed in trace:
                if parsed.label not in self.config.TERMS_FOR_MISSING:
                    event = Event(
                        {
                            self.config.XES_NAME: parsed.label,
                            self.config.DICTIONARY: parsed.dictionary_entries,
                            self.config.DATA_OBJECT: parsed.data_objects,
                        }
                    )
                    tmp_trace.append(event)
            if len(tmp_trace) > 0:
                projection.append(tmp_trace)
        return projection

    def object_action_log_projection(self, obj, traces):
        """
        Return for each trace a time-ordered list of the actions for a given object type.

        Returns
        -------
        projection
            traces containing only actions applied to the same obj.
        """
        projection = EventLog()
        if traces is None:
            raise RuntimeError("You must load a log before.")
        for i, trace in enumerate(traces):
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = str(i)
            for parsed in trace:
                if parsed.main_object == obj:
                    if parsed.main_action != "":
                        event = Event(
                            {
                                self.config.XES_NAME: parsed.main_action,
                                self.config.DICTIONARY: parsed.dictionary_entries,
                                self.config.DATA_OBJECT: parsed.data_objects,
                            }
                        )
                        tmp_trace.append(event)
            if len(tmp_trace) > 0:
                projection.append(tmp_trace)
        return projection

    def object_log_projection(self, traces):
        """
        Return for each trace a time-ordered list of the actions for a given object type.

        Returns
        -------
        projection
            traces containing only actions applied to the same obj.
        """
        projection = EventLog()
        if traces is None:
            raise RuntimeError("You must load a log before.")
        for i, trace in enumerate(traces):
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = str(i)
            last = ""
            for parsed in trace:
                if type(parsed.main_object) != str:
                    _logger.warning("main_object is not a string: %s" % parsed.main_object)
                    continue
                if parsed.main_object not in self.config.TERMS_FOR_MISSING and parsed.main_object != last:
                    event = Event(
                        {
                            self.config.XES_NAME: parsed.main_object,
                            self.config.DICTIONARY: parsed.dictionary_entries,
                            self.config.DATA_OBJECT: parsed.data_objects,
                        }
                    )
                    tmp_trace.append(event)
                last = parsed.main_object
            projection.append(tmp_trace)
        return projection

    def get_constraints_flat(self, res, associations=None):
        res = [{self.config.RECORD_ID: str(uuid.uuid4()),
                self.config.LEVEL: self.config.ACTIVITY,
                self.config.OBJECT: "",
                self.config.CONSTRAINT_STR: const,
                self.config.OPERATOR_TYPE: self.config.BINARY if any(
                    temp in const for temp in self.config.BINARY_TEMPLATES) else self.config.UNARY,
                self.config.DICTIONARY: associations[const][self.config.DICTIONARY],
                self.config.DATA_OBJECT: associations[const][self.config.DATA_OBJECT],
                self.config.TEMPLATE: _get_constraint_template(const)
                } for const in res]
        for rec in res:
            if rec[self.config.OPERATOR_TYPE] == self.config.UNARY:
                rec[self.config.LEFT_OPERAND] = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace(
                    "|", "").strip()
            if rec[self.config.OPERATOR_TYPE] == self.config.BINARY:
                ops = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").split(",")
                rec[self.config.LEFT_OPERAND] = ops[0].strip()
                rec[self.config.RIGHT_OPERAND] = ops[1].strip()
        return res

