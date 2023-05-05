from pandas import DataFrame
from pm4py.objects.log.obj import EventLog, Trace, Event

from semconstmining.parsing.conversion.petrinetanalysis import is_relevant_label
from semconstmining.mining.model.parsed_label import get_dummy
from semconstmining.declare.declare import Declare
from semconstmining.declare.parsers import parse_decl
import pm4py

from semconstmining.parsing.label_parser.nlp_helper import NlpHelper


class DeclareChecker:

    def __init__(self, config, log: DataFrame, constraints: DataFrame, nlp_helper: NlpHelper):
        self.config = config
        self.log = log
        self.nlp_helper = nlp_helper
        self.constraints = constraints
        self.activities = pm4py.get_event_attribute_values(self.log, self.config.XES_NAME, case_id_key=self.config.XES_CASE)
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
        }
        return res

    def get_constraint_strings(self):
        constraint_strings = []
        for row in self.constraints.itertuples():
            constraint_strings.append(row.constraint_string)
        return constraint_strings

    def check_object_level_constraints(self):
        filtered_traces = self.get_filtered_traces(self.log, parsed_tasks=self.activities_to_parsed)
        res = {}
        bos = set([x.main_object for trace in filtered_traces for x in trace if x.main_object not in self.config.TERMS_FOR_MISSING])
        for bo in bos:
            d4py = Declare(self.config)
            d4py.log = self.object_action_log_projection(bo, filtered_traces)
            constraint_strings = self.get_constraint_strings()
            d4py.model = parse_decl(constraint_strings)
            tmp_res = d4py.conformance_checking(consider_vacuity=True)
            res[bo] = {"-".join(key): val for key, val in tmp_res.items()}
        return res

    def check_multi_object_constraints(self):
        filtered_traces = self.get_filtered_traces(self.log, parsed_tasks=self.activities_to_parsed)
        d4py = Declare(self.config)
        d4py.log = self.object_log_projection(filtered_traces)
        constraint_strings = self.get_constraint_strings()
        d4py.model = parse_decl(constraint_strings)
        tmp_res = d4py.conformance_checking(consider_vacuity=True)
        res = {"-".join(key): val for key, val in tmp_res.items()}
        return res

    def check_activity_level_constraints(self):
        filtered_traces = self.get_filtered_traces(self.log, parsed_tasks=self.activities_to_parsed, with_loops=True)
        d4py = Declare(self.config)
        d4py.log = self.clean_log_projection(filtered_traces)
        constraint_strings = self.get_constraint_strings()
        d4py.model = parse_decl(constraint_strings)
        tmp_res = d4py.conformance_checking(consider_vacuity=True)
        res = {"-".join(key): val for key, val in tmp_res.items()}
        return res

    def check_resource_level_constraints(self):
        filtered_traces = self.get_filtered_traces(self.log, parsed_tasks=self.activities_to_parsed, with_loops=True)
        d4py = Declare(self.config)
        d4py.log = self.clean_log_projection(filtered_traces)
        constraint_strings = self.get_constraint_strings()
        d4py.model = parse_decl(constraint_strings)
        tmp_res = d4py.conformance_checking(consider_vacuity=True)
        res = {"-".join(key): val for key, val in tmp_res.items()}
        return res

    def has_loop(self, trace):
        return trace[self.config.XES_NAME].nunique() > len(trace)

    def get_parsed_tasks(self, log: DataFrame, resource_handler, only_relevant_labels=True):
        relevant_tasks = set(
            [event[self.config.XES_NAME] for trace_id, trace in log.groupby(self.config.XES_CASE) for event_idx, event in trace.iterrows() if
             is_relevant_label(event[self.config.XES_NAME])]) if only_relevant_labels else set(
            [x[self.config.XES_NAME] for trace in log for x in trace])
        return {t: resource_handler.get_parsed_task(t) for t in relevant_tasks}

    def get_filtered_traces(self, log: DataFrame, parsed_tasks=None, with_loops=False, with_resources=False):
        if parsed_tasks is not None:
            res = [
                [parsed_tasks[event[self.config.XES_NAME]] if event[self.config.XES_NAME] in parsed_tasks else get_dummy(self.config, event[self.config.XES_NAME], self.config.EN)
                 for event_index, event in
                 trace.iterrows()] for trace_id, trace in log.groupby(self.config.XES_CASE) if
                not with_loops and not self.has_loop(trace)]
            return res
        else:
            res = [[event[self.config.XES_NAME] for event_idx, event in trace.iterrows()] for trace_id, trace in
                    log.groupby(self.config.XES_CASE)
                    if not with_loops and not self.has_loop(trace)]
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
        for i, trace in enumerate(traces):
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = str(i)
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
        for i, trace in enumerate(traces):
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = str(i)
            last = ""
            for parsed in trace:
                if parsed.main_object not in self.config.TERMS_FOR_MISSING and parsed.main_object != last:
                    event = Event({self.config.XES_NAME: parsed.main_object})
                    tmp_trace.append(event)
                last = parsed.main_object
            projection.append(tmp_trace)
        return projection

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
                    event = Event({self.config.XES_NAME: parsed.label})
                    tmp_trace.append(event)
            if len(tmp_trace) > 0:
                projection.append(tmp_trace)
        return projection





