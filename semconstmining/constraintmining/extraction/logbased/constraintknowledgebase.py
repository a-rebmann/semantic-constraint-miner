from semconstmining.constraintmining.bert_parser import label_utils
from semconstmining.conformance.similaritycomputer import SimMode
from semconstmining.constraintmining.model.constraint import Observation, Constraint


class ConstraintKnowledgeBase:

    def __init__(self,config):
        self.config= config
        self.record_map = {}
        self.object_constraints = {}
        self.actions = None
        self.min_support = 1
        self.apply_filter_heuristics = False
        self.label_util = label_utils.LabelUtil(config)

    def get_record(self, action1, action2, record_type):
        action1 = self.label_util.lemmatize_word(action1)
        action2 = self.label_util.lemmatize_word(action2)
        # symmetric XOR records are always stored in lexical order
        if record_type == Observation.ACT_XOR and action2 > action1:
            action1, action2 = action2, action1
        if (action1, action2, record_type) in self.record_map:
            return self.record_map[(action1, action2, record_type)]
        return None

    def get_record_count(self, action1, action2, record_type):
        record = self.get_record(action1, action2, record_type)
        if record is None:
            return 0
        return record.count

    def add_observation(self, action1, action2, record_type, model_name="", count=1):
        record = self.get_record(action1, action2, record_type)
        if record:
            record.increment_count(count)
            if model_name and len(model_name) > 0:
                record.add_model_name(model_name)
        else:
            self.add_new_record(action1, action2, record_type, count, model_name)

    def add_new_record(self, action1, action2, record_type, count, model_name):
        action1 = self.label_util.lemmatize_word(action1)
        action2 = self.label_util.lemmatize_word(action2)
        # ensure consistent ordering for symmetric XOR records
        if record_type == Observation.ACT_XOR and action2 > action1:
            action1, action2 = action2, action1
        self.record_map[(action1, action2, record_type)] = Constraint(const=(action1, action2), const_type=record_type, count=count, model_names={
            model_name})

    def get_similar_records(self, action1, action2, record_type, sim_computer):
        sim_verbs1 = self._get_sim_verbs(action1, sim_computer)
        sim_verbs2 = self._get_sim_verbs(action2, sim_computer)
        records = []
        if not sim_computer.match_one:
            # both verbs in a record may differ from original ones
            for sim_action1 in sim_verbs1:
                for sim_action2 in sim_verbs2:
                    if self.get_record(sim_action1, sim_action2, record_type):
                        records.append(self.get_record(sim_action1, sim_action2, record_type))
        else:
            #     requires that at least one verb in record corresponds to original one
            for sim_action1 in sim_verbs1:
                if self.get_record(sim_action1, action2, record_type):
                    records.append(self.get_record(sim_action1, action2, record_type))
            for sim_action2 in sim_verbs2:
                if self.get_record(action1, sim_action2, record_type):
                    records.append(self.get_record(action1, sim_action2, record_type))
        return records

    def _get_sim_verbs(self, verb, sim_computer):
        verb = self.label_util.lemmatize_word(verb)
        sim_verbs = []
        if sim_computer.sim_mode == SimMode.SYNONYM:
            sim_verbs = sim_computer.get_synonyms(verb)
        if sim_computer.sim_mode == SimMode.SEMANTIC_SIM:
            sim_verbs = sim_computer.compute_semantic_sim_verbs(verb, self.get_all_actions())
        # filter out any verb that opposeses the original verb
        sim_verbs = [sim_verb for sim_verb in sim_verbs if not self.get_record(verb, sim_verb, Observation.ACT_XOR)]
        return sim_verbs

    def filter_out_conflicting_records(self):
        new_map = {}
        for (action1, action2, record_type) in self.record_map:
            # filter out conflicting exclusion constraints
            # logic: if there is a cooccurrence or order constraint involving two verbs, then they cannot be exclusive
            if record_type == Observation.ACT_XOR:
                record_count = self.get_record_count(action1, action2, record_type)
                other_counts = self.get_record_count(action1, action2, Observation.ACT_CO_OCC) + \
                               self.get_record_count(action1, action2, Observation.ACT_ORDER) + \
                               self.get_record_count(action2, action1, Observation.ACT_ORDER)
                if record_count > other_counts:
                    new_map[(action1, action2, record_type)] = self.record_map[(action1, action2, record_type)]
                else:
                    print('removing', (action1, action2, record_type, record_count), "from kb. Other count:", other_counts)
            # filter out conflicting ordering constraints
            # logic: only keep this order constraint if the reverse is less common
            if record_type == Observation.ACT_ORDER:
                order_count = self.get_record_count(action1, action2, record_type)
                reverse_count = self.get_record_count(action2, action1, record_type)
                if order_count > reverse_count:
                    new_map[(action1, action2, record_type)] = self.record_map[(action1, action2, record_type)]
                else:
                    print('removing', (action1, action2, record_type, order_count), "from kb. Reverse count:",
                          reverse_count)
            # filter out conflicting co-occurrence constraints
            if record_type == Observation.ACT_CO_OCC:
                new_map[(action1, action2, record_type)] = self.record_map[(action1, action2, record_type)]
        self.record_map = new_map

    @property
    def get_all_actions(self):
        if not self.actions:
            res = set()
            for record in self.record_map.values():
                res.add(record.action1)
                res.add(record.action2)
            self.actions = list(res)
        return self.actions

    def get_record_numbers(self):
        count_order = len([record for record in self.record_map.values() if record.const_type == Observation.ACT_ORDER])
        count_xor = len([record for record in self.record_map.values() if record.const_type == Observation.ACT_XOR])
        count_coocc = len([record for record in self.record_map.values() if record.const_type == Observation.ACT_CO_OCC])
        return (count_xor, count_order, count_coocc, len(self.record_map))

    def print_most_common_records(self):
        newlist = sorted(self.record_map.values(), key=lambda x: x.count, reverse=True)
        for i in range(0, 20):
            print(newlist[i])
