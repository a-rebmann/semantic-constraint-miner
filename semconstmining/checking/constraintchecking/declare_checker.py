from collections import Counter

from pandas import DataFrame
from pm4py.objects.log.obj import EventLog, Trace, Event

from semconstmining.log.loghandler import LogHandler
from semconstmining.parsing.conversion.petrinetanalysis import is_relevant_label
from semconstmining.mining.model.parsed_label import get_dummy
from semconstmining.declare.declare import Declare
from semconstmining.declare.parsers import parse_decl
import pm4py

from semconstmining.parsing.label_parser.nlp_helper import NlpHelper


class DeclareChecker:

    def __init__(self, config, lh: LogHandler, constraints: DataFrame, nlp_helper: NlpHelper):
        self.config = config
        self.log_handler = lh
        self.log = self.log_handler.log
        self.res_to_task = self.log_handler.get_resources_to_tasks()
        self.task_to_res = {task: res for res, tasks in self.res_to_task.items() for task in tasks}
        self.nlp_helper = nlp_helper
        self.constraints = constraints
        self.activities = pm4py.get_event_attribute_values(self.log, self.config.XES_NAME,
                                                           case_id_key=self.config.XES_CASE)
        self.activities_to_parsed = {activity: self.nlp_helper.parse_label(activity) for activity in self.activities}

    def check_constraints(self):
        res = {
            # First we check the object-level constraints
            self.config.OBJECT: self.check_object_level_constraints()
            # Then we check the multi-object constraints
            , self.config.MULTI_OBJECT: self.check_multi_object_constraints()
            # Then we check the activity-level constraints
            , self.config.ACTIVITY: self.check_activity_level_constraints()
            # Then we check the resource constraints
            , self.config.RESOURCE: self.check_resource_level_constraints()
            if self.config.XES_ROLE in self.log.columns else {}
        }
        return res

    def get_constraint_strings(self, level: str):
        constraint_strings = {}
        if len(self.constraints) == 0:
            return constraint_strings
        for idx, row in self.constraints[self.constraints[self.config.LEVEL] == level].iterrows():
            constraint_strings[row[self.config.CONSTRAINT_STR]] = row[self.config.TEMPLATE]
        return constraint_strings

    def check_object_level_constraints(self):
        filtered_traces = self.get_filtered_traces(self.log, parsed_tasks=self.activities_to_parsed,
                                                   with_loops=self.config.LOOPS)
        res = {}
        bos = set([x.main_object for trace in filtered_traces.values() for x in trace if
                   x.main_object not in self.config.TERMS_FOR_MISSING])
        for bo in bos:
            d4py = Declare(self.config)
            d4py.log = self.object_action_log_projection(bo, filtered_traces)
            constraint_strings = self.get_constraint_strings(level=self.config.OBJECT)
            d4py.model = parse_decl(constraint_strings.keys())
            tmp_res = d4py.conformance_checking(consider_vacuity=True)
            res[bo] = self.verify_violations(tmp_res, d4py.log)
        return res

    def check_multi_object_constraints(self):
        filtered_traces = self.get_filtered_traces(self.log, parsed_tasks=self.activities_to_parsed,
                                                   with_loops=self.config.LOOPS)
        d4py = Declare(self.config)
        d4py.log = self.object_log_projection(filtered_traces)
        constraint_strings = self.get_constraint_strings(level=self.config.MULTI_OBJECT)
        d4py.model = parse_decl(constraint_strings.keys())
        tmp_res = d4py.conformance_checking(consider_vacuity=True)
        return self.verify_violations(tmp_res, d4py.log)

    def check_activity_level_constraints(self):
        filtered_traces = self.get_filtered_traces(self.log, parsed_tasks=self.activities_to_parsed,
                                                   with_loops=self.config.LOOPS)
        d4py = Declare(self.config)
        d4py.log = self.clean_log_projection(filtered_traces)
        constraint_strings = self.get_constraint_strings(level=self.config.ACTIVITY)
        d4py.model = parse_decl(constraint_strings.keys())
        tmp_res = d4py.conformance_checking(consider_vacuity=True)
        return self.verify_violations(tmp_res, d4py.log)

    def check_resource_level_constraints(self):
        filtered_traces = self.get_filtered_traces(self.log, parsed_tasks=self.activities_to_parsed,
                                                   with_loops=self.config.LOOPS)
        d4py = Declare(self.config)
        d4py.log = self.clean_log_projection(filtered_traces, with_resources=True)
        constraint_strings = self.get_constraint_strings(level=self.config.RESOURCE)
        d4py.model = parse_decl(constraint_strings.keys())
        tmp_res = d4py.conformance_checking(consider_vacuity=True)
        return self.verify_violations(tmp_res, d4py.log)

    def has_loop(self, trace):
        return trace[self.config.XES_NAME].nunique() > len(trace)

    def get_filtered_traces(self, log: DataFrame, parsed_tasks=None, with_loops=False):
        if parsed_tasks is not None:
            res = {trace_id:
                       [parsed_tasks[event[self.config.XES_NAME]] if event[
                                                                         self.config.XES_NAME] in parsed_tasks else get_dummy(
                           self.config, event[self.config.XES_NAME], self.config.EN)
                        for event_index, event in
                        trace.iterrows()] for trace_id, trace in log.groupby(self.config.XES_CASE) if
                    with_loops or not self.has_loop(trace)}
            return res
        else:
            res = {trace_id: [event[self.config.XES_NAME] for event_idx, event in trace.iterrows()] for trace_id, trace
                   in
                   log.groupby(self.config.XES_CASE)
                   if with_loops or not self.has_loop(trace)}
            return res

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
        for trace_id, trace in traces.items():
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = trace_id
            for parsed in trace:
                if parsed.main_object == obj:
                    if parsed.main_action != "":
                        event = Event({self.config.XES_NAME: parsed.main_action})
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
        for trace_id, trace in traces.items():
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = trace_id
            last = ""
            for parsed in trace:
                if parsed.main_object not in self.config.TERMS_FOR_MISSING and parsed.main_object != last:
                    event = Event({self.config.XES_NAME: parsed.main_object})
                    tmp_trace.append(event)
                last = parsed.main_object
            projection.append(tmp_trace)
        return projection

    def clean_log_projection(self, traces, with_resources=False):
        """
        Same log, just with clean labels.
        """
        projection = EventLog()
        if traces is None:
            raise RuntimeError("You must load a log before.")
        for trace_id, trace in traces.items():
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = trace_id
            for parsed in trace:
                if parsed.label not in self.config.TERMS_FOR_MISSING:
                    event = Event({self.config.XES_NAME: parsed.label})
                    if with_resources:
                        event[self.config.XES_ROLE] = self.task_to_res[parsed.label]
                    tmp_trace.append(event)
            if len(tmp_trace) > 0:
                projection.append(tmp_trace)
        return projection

    def verify_violations(self, tmp_res, log):
        counts = Counter(const for vals in tmp_res.values() for const in vals)
        res = {key: {val for val in vals if counts[val] < len(log)} for key, vals in tmp_res.items()}
        return res

