from collections import Counter

from semconstmining.conformance.violation import Violation
from semconstmining.conformance.similaritycomputer import SimMode
from semconstmining.constraintmining.model.constraint import Observation


class ConformanceChecker:

    def __init__(self, kb, config, parser, sim_computer, log, log_name, event_key, already_simplified=False):
        # configuration options
        self.kb = kb
        self.equal_bos = config.equal_bos
        self.split_loops = config.split_loops
        self.parser = parser
        self.sim_computer = sim_computer

        # log information
        self.log = log
        self.log_name = log_name
        self.event_key = event_key
        self.already_simplified = already_simplified

        # anomaly detection objects
        self.parse_map = {}
        self.anomaly_counter = Counter()
        self.paired_anomaly_map = {}
        self.coocc_pairs = set()
        self.anomalous_variants = []

    def detect_anomalies(self):
        # initialize similarity matrix if necessary
        if self.sim_computer.sim_mode == SimMode.SEMANTIC_SIM and self.sim_computer.compute_sim_per_log:
            self.sim_computer.initialize_similarities(self.kb.get_all_actions(), self.get_all_actions_in_log())

        actions = self.get_all_actions_in_log()
        print("checking for conformance in", self.log_name, "with", len(self.log.get_variants()), "variants")
        # first parse over all variants
        for variant in self.log.get_variants():
            variant_anomalies = self.detect_anomalies_in_trace(variant)
            if variant_anomalies:
                self.anomalous_variants.append(variant)
                for anomaly in variant_anomalies:
                    self.anomaly_counter[anomaly] += self.log.variant_count(variant)

        # determine actual co-occurrence anomalies based on observed co-occurrence pairs
        for variant in self.log.get_variants():
            variant_coocc_anomalies = self.identify_real_cooccurrence_anomalies(variant)
            if variant_coocc_anomalies:
                if variant not in self.anomalous_variants:
                    self.anomalous_variants.append(variant)
                for anomaly in variant_coocc_anomalies:
                    self.anomaly_counter[anomaly] += self.log.variant_count(variant)
        return self.anomaly_counter

    def detect_anomalies_in_trace(self, variant):
        variant_anomalies = set()
        subtraces = [variant]
        if self.split_loops:
            subtraces = split_into_subtraces(variant)
        for subtrace in subtraces:
            for i in range(len(subtrace) - 1):
                for j in range(i + 1, len(subtrace)):
                    variant_anomalies.update(self.detect_anomalies_for_pair(subtrace[i], subtrace[j]))
                    # for anomaly in pair_anomalies:
                    #     self.anomaly_counter[anomaly] += self.simple_log.variant_count(variant)
        return variant_anomalies

    def detect_anomalies_for_pair(self, event1_name, event2_name):
        pair_anomalies = set()
        # if pair already observed, return stored violations
        if (event1_name, event2_name) in self.paired_anomaly_map:
            return self.paired_anomaly_map[(event1_name, event2_name)]

        if self.pair_should_be_checked(event1_name, event2_name):
            verb1, verb2 = self.parser.get_action(event1_name), self.parser.get_action(event2_name)
            # check for exclusion violations
            if self.kb.has_xor_violation(verb1, verb2, self.sim_computer):
                # sort exclusion anomalies in a consistent manner
                if verb1 > verb2:
                    pair_anomalies.add(Violation(Observation.ACT_XOR, event1_name, event2_name,
                                                 verb1, verb2,
                                                 self.kb.get_record_count(verb1, verb2, Observation.ACT_XOR)))
                else:
                    pair_anomalies.add(Violation(Observation.ACT_XOR, event1_name, event2_name,
                                                 verb2, verb1,
                                                 self.kb.get_record_count(verb1, verb2, Observation.ACT_XOR)))
            #  check for ordering violation
            if self.kb.has_order_violation(verb1, verb2, self.sim_computer):
                pair_anomalies.add(Violation(Observation.ACT_ORDER, event1_name,
                                             event2_name, verb1, verb2,
                                             self.kb.get_record_count(verb2, verb1, Observation.ACT_ORDER)))
            # check for potential co-occurrence violation
            if self.kb.has_cooc_dependency(verb1, verb2, self.sim_computer):
                self.coocc_pairs.add((event1_name, event2_name))
                self.coocc_pairs.add((event2_name, event1_name))
        self.paired_anomaly_map[(event1_name, event2_name)] = pair_anomalies
        return pair_anomalies

    def pair_should_be_checked(self, event1_name, event2_name):
        # Parse events
        e1_parse = self.parser.parse_label(event1_name)
        e2_parse = self.parser.parse_label(event2_name)
        # check if conditions on business object match are met
        if not self.equal_bos or e1_parse.bos == e2_parse.bos:
            # check if there are actual actions detected
            # if len(e1_parse.actions) > 0 and len(e2_parse.actions) > 0:
            verb1, verb2 = self.parser.get_action(event1_name), self.parser.get_action(event2_name)
            # check if non-empty verbs and make sure they are not equivalent
            if verb1 and verb2 and not verb1 == verb2:
                return True
        return False

    def identify_real_cooccurrence_anomalies(self, variant):
        variant_anomalies = set()
        # store all events in trace
        for event_name in variant:
            # check if event is part of a co-occurrence pair
            for (event1_name, event2_name) in self.coocc_pairs:
                if event_name == event1_name and event2_name not in variant:
                    verb1, verb2 = self.parser.get_action(event1_name), self.parser.get_action(event2_name)
                    count = self.kb.get_record_count(verb1, verb2, Observation.ACT_CO_OCC)
                    if event1_name < event2_name:
                        variant_anomalies.add(
                            Violation(Observation.ACT_CO_OCC, event1_name, event2_name, verb1, verb2, count))
                    else:
                        variant_anomalies.add(
                            Violation(Observation.ACT_CO_OCC, event1_name, event2_name, verb2, verb1, count))
        return variant_anomalies

    def get_all_actions_in_log(self):
        actions = {self.parser.get_action(event) for variant in self.log.get_variants() for event in variant}
        if None in actions:
            actions.remove(None)
        return actions


def split_into_subtraces(variant):
    variants = []
    current = []
    for event in variant:
        if event not in current:
            current.append(event)
        else:
            variants.append(current)
            current = [event]
    variants.append(current)
    return variants
