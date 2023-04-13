import json
import logging
import operator

_logger = logging.getLogger(__name__)


class ActionClassifier:

    def __init__(self, config, actions, embeddings):
        self.config = config
        self.actions = actions
        self.embeddings = embeddings
        action_taxonomy = self.config.mitphb
        # all unique actions
        unique_actions_taxonomy = set()
        # a mapping from all unique actions to their top most ancestor(s)
        child_to_upper_level = dict()
        # all upper level actions
        upper_acts = set()
        self.unique_actions_from_taxonomy(action_taxonomy, unique_actions_taxonomy, child_to_upper_level,
                                          upper_acts)
        self.unique_actions_taxonomy = unique_actions_taxonomy
        self.child_to_upper_level = child_to_upper_level
        self.upper_acts = upper_acts

    def classify_actions(self):
        return {act: self.get_action_type_for_action(act) for act in self.actions}

    def unique_actions_from_taxonomy(self, action_taxonomy, unique_actions, child_to_upper_level, upper_acts,
                                     upper_level=None):
        for act, children in action_taxonomy.items():
            unique_actions.add(act)
            if upper_level is None:
                child_to_upper_level[act] = {act}
                upper_acts.add(act)
                ul = act
            else:
                if act in child_to_upper_level:
                    child_to_upper_level[act].add(upper_level)
                else:
                    child_to_upper_level[act] = {upper_level}
                ul = upper_level
            for child in children:
                self.unique_actions_from_taxonomy(child, unique_actions, child_to_upper_level, upper_acts,
                                                  upper_level=ul)

    def produce_gs(self):
        with open('./gt_actions.json') as json_file:
            gt = json.load(json_file)
            action_taxonomy = json.load(json_file)
            # all unique actions
            unique_actions_taxonomy = set()
            # a mapping from all unique actions to their top most ancestor(s)
            child_to_upper_level = dict()
            # all upper level actions
            upper_acts = set()
            self.unique_actions_from_taxonomy(action_taxonomy, unique_actions_taxonomy, child_to_upper_level,
                                              upper_acts)
            for action in self.actions:
                ms = self.get_most_similar(action, unique_actions_taxonomy, child_to_upper_level, upper_acts)
                if action in gt:
                    continue
                else:
                    gt[action] = ms
            with open(self.config.resource_dir + 'gt_actions.json', 'w') as outfile:
                json.dump(gt, outfile)

    def get_most_similar(self, action, taxonomy_actions, child_to_upper_level, upper_acts):
        if len(action) < 3:
            return "None"
        sims = {}
        upper_level_sims = {}
        for tax_action in taxonomy_actions:
            try:
                sim = self.embeddings.similarity(action, tax_action)
                # print(action, tax_action, sim)
                if tax_action in upper_acts:
                    upper_level_sims[tax_action] = sim
                sims[tax_action] = sim
            except KeyError as e:
                _logger.warning("KeyError: " + str(e))
        if len(sims) == 0:
            return "None"
        max_sim = max(sims.items(), key=operator.itemgetter(1))[0]
        max_sim_upper = max(upper_level_sims.items(), key=operator.itemgetter(1))[0]
        max_sim_upper_ini = str(max_sim_upper)

        if len(child_to_upper_level[max_sim]) == 1:
            max_sim = list(child_to_upper_level[max_sim])[0]
        else:
            max_sim_upper = -1
            for upper_level_act in child_to_upper_level[max_sim]:
                if upper_level_sims[upper_level_act] > max_sim_upper:
                    max_sim = upper_level_act
                    max_sim_upper = upper_level_sims[upper_level_act]

        return max_sim if sims[max_sim] > 0 else max_sim_upper_ini

    def get_action_type_for_action(self, action):
        return self.get_most_similar(action, self.unique_actions_taxonomy, self.child_to_upper_level, self.upper_acts)
