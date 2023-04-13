import json
import uuid
import warnings
from typing import List, Dict

import pandas as pd
import logging

from semconstmining.constraintmining.extraction.declareextractor import _get_constraint_template
from semconstmining.parsing.bert_parser import label_utils
from semconstmining.parsing.conversion.bpmnjsonanalyzer import process_bpmn_shapes
from semconstmining.constraintmining.model.constraint import Observation
from semconstmining.parsing.conversion import bpmnjsonanalyzer as bpmn_analyzer
from semconstmining.declare.enums import Template
from semconstmining.parsing.resource_handler import ResourceHandler

warnings.simplefilter('ignore')

_logger = logging.getLogger(__name__)


def _get_json_from_row(row_tuple):
    return json.loads(row_tuple.model_json)


def process_pools_and_lanes(shapes, pools, lanes):
    follows = {}
    labels = {}
    tasks = set()
    # Analyze shape list and store all shapes and activities
    # PLEASE NOTE: the code below ignores BPMN sub processes
    for shape in shapes:
        # If current shape is a pool or a lane, we have to go a level deeper
        shapeID = shape['resourceId']
        if shape['stencil']['id'] == 'Pool' or shape['stencil']['id'] == 'Lane':
            if 'name' in shape['properties'] and not shape['properties']['name'] == "":
                if shape['stencil']['id'] == 'Pool':
                    result = process_pools_and_lanes(shape['childShapes'], pools, lanes)
                    pools[shapeID] = {"name": shape['properties']['name'], "follows": result[0],
                                      "labels": result[1], "tasks": result[2]}
                if shape['stencil']['id'] == 'Lane':
                    result = process_pools_and_lanes(shape['childShapes'], pools, lanes)
                    lanes[shapeID] = {"name": shape['properties']['name'], "follows": result[0],
                                      "labels": result[1], "tasks": result[2]}

        outgoingShapes = [s['resourceId'] for s in shape['outgoing']]
        if shapeID not in follows:
            follows[shapeID] = outgoingShapes

        # Save all tasks and respective labels separately
        if shape['stencil']['id'] == 'Task':
            if not shape['properties']['name'] == "":
                tasks.add(shapeID)
                labels[shapeID] = shape['properties']['name'].replace('\n', ' ').replace('\r', '').replace('  ',
                                                                                                           ' ')
            else:
                labels[shapeID] = 'Task'
        else:
            if 'name' in shape['properties'] and not shape['properties']['name'] == "":
                labels[shapeID] = shape['stencil']['id'] + " (" + shape['properties']['name'].replace('\n',
                                                                                                      ' ').replace(
                    '\r',
                    '').replace(
                    '  ', ' ') + ")";
            else:
                labels[shapeID] = shape['stencil']['id']
    return follows, labels, tasks


def _create_mp_declare_const_with_role_condition(task, role):
    # As a proxy for the role constraint we use an Absence1 constraint, which by itself means that the
    # respective task never occurs. We add a data condition on the activation that specifies the role that
    # should execute the task. For instance, Absence1(approve order)|A.org:role!=manager|. If we translate this, we get
    # 'approve order' should never occur if its executing role is not 'manager'
    tmp_str = Template.ABSENCE.templ_str
    n = "1"
    constraint_str = tmp_str + n + "[" + task + "]" + " |A.org:role is not " + role + " |"
    return constraint_str


def _create_mp_declare_const_with_decision_condition(left, right):
    # As a proxy for the decision constraint which corresponds to exclusive choice
    tmp_str = Template.EXCLUSIVE_CHOICE.templ_str
    constraint_str = tmp_str + "[" + left + ", " + right + "]" + " | | |"
    return constraint_str


class ModelExtractor:

    def __init__(self, config, resource_handler: ResourceHandler, types_to_ignore):
        self.config = config
        self.resource_handler = resource_handler
        self.types_to_ignore = [] if not types_to_ignore else types_to_ignore

    def _traverse_and_extract_decisions(self, follows, labels, tasks):
        choice_sets = {}
        irrelevant_shapes = ()
        for s in follows.keys():
            # Only check relevant shapes
            if bpmn_analyzer.is_relevant(s, labels, irrelevant_shapes):
                # Get postset of considered element "s"
                postset = set(follows[s])
                # ++++++++++++++++++++++++++++++
                # Source = choice gateway
                # ++++++++++++++++++++++++++++++
                if bpmn_analyzer.get_type(s, labels, irrelevant_shapes) == "Gateway":
                    if bpmn_analyzer.is_choice(s, labels):
                        choice_sets[s] = {"name": labels[s], "choices": set()}
                        for elem in postset:
                            seq = labels[elem]
                            first = seq.split(' ')[0]
                            if first in self.config.NON_TASKS:
                                seq = seq.replace(first, '')
                            choice_sets[s]["choices"].add(seq)
        return choice_sets

    def _get_observations_for_decision_perspective(self, json_str, types_to_ignore):
        observations = []
        if Template.EXCLUSIVE_CHOICE.templ_str not in types_to_ignore:
            try:
                follows, labels, tasks = process_bpmn_shapes(json_str['childShapes'])
                choice_sets = self._traverse_and_extract_decisions(follows, labels, tasks)
                for gateway_id, gateway_payload in choice_sets.items():
                    for choice1 in gateway_payload["choices"]:
                        if label_utils.sanitize_label(choice1) != "":
                            for choice2 in gateway_payload["choices"]:
                                if label_utils.sanitize_label(choice2) != "" and not choice1 == choice2:
                                    decision_left, decision_right = label_utils.sanitize_label(
                                        choice1), label_utils.sanitize_label(choice2)
                                    if decision_left != decision_right and \
                                            decision_left not in self.config.IRRELEVANT_CONSTRAINTS[
                                        Template.EXCLUSIVE_CHOICE.templ_str] and \
                                            decision_right not in self.config.IRRELEVANT_CONSTRAINTS[
                                        Template.EXCLUSIVE_CHOICE.templ_str]:
                                        observation = (
                                            decision_left, decision_right,
                                            _create_mp_declare_const_with_decision_condition(decision_left,
                                                                                             decision_right), self.config.DECISION,
                                            self.config.BINARY, "")
                                        observations.append(observation)
                                        # observations[frozenset([decision_left, decision_right])] = observation
            except Exception as ex:
                _logger.warning(ex)

        return observations  # list(observations.values())

    def get_perspectives_from_models(self):
        dfs = [self._get_observations_from_model(t, self.types_to_ignore)
               for t in self.resource_handler.bpmn_models.reset_index().itertuples()]
        dfs = [df for df in dfs if df is not None]
        return pd.concat(dfs).astype({self.config.LEVEL: "category"})  # .set_index([RECORD_ID])

    def _get_observations_from_model(self, row_tuple, types_to_ignore):
        if row_tuple.model_json is None:
            return None
        observations = []
        json_str = _get_json_from_row(row_tuple)
        # The resource perspective
        observations.extend(self._get_observations_for_resource_perspective(json_str, types_to_ignore))
        # The data perspective
        observations.extend(self._get_observations_for_decision_perspective(json_str, types_to_ignore))
        # The object perspective
        #observations.extend(self._get_observations_for_object_perspective(json_str, types_to_ignore))
        return (
            pd.DataFrame.from_records(self.get_observations_flat(observations)).assign(model_id=row_tuple.model_id).assign(
                model_name=row_tuple.name)
        )

    def _get_observations_for_resource_perspective(self, json_str, types_to_ignore):
        observations = []
        pools = {}
        lanes = {}
        try:
            process_pools_and_lanes(json_str['childShapes'], pools, lanes)
        except Exception as ex:
            _logger.debug(ex)
            return observations
        if Observation.RESOURCE_TASK_EXISTENCE not in types_to_ignore:
            for _, lane_info in lanes.items():
                for shape_id, label in lane_info["labels"].items():
                    if shape_id in lane_info["tasks"]:
                        lan_str = label_utils.sanitize_label(lane_info["name"])
                        task_str = label_utils.sanitize_label(label)
                        if lan_str == 'lane' or lan_str in self.config.TERMS_FOR_MISSING or task_str in self.config.TERMS_FOR_MISSING:
                            continue
                        observation = (task_str, lan_str,
                                       _create_mp_declare_const_with_role_condition(task_str, lan_str), self.config.RESOURCE, self.config.UNARY,
                                       "")
                        observations.append(observation)
            for _, pool_info in pools.items():
                for shape_id, label in pool_info["labels"].items():
                    if shape_id in pool_info["tasks"]:
                        pool_str = label_utils.sanitize_label(pool_info["name"])
                        task_str = label_utils.sanitize_label(label)
                        if pool_str == 'pool' or pool_str in self.config.TERMS_FOR_MISSING or task_str in self.config.TERMS_FOR_MISSING:
                            continue
                        observation = (task_str, pool_str,
                                       _create_mp_declare_const_with_role_condition(task_str, pool_str), self.config.RESOURCE,
                                       self.config.UNARY,
                                       "")
                        observations.append(observation)

        # DEPRECATED: TODO find proper way to represent this in DECLARE or SiGNAL
        if Observation.RESOURCE_CONTAINMENT not in types_to_ignore:
            for pool_id, pool_info in pools.items():
                for shape_id, label in pool_info["labels"].items():
                    if shape_id in lanes:
                        pool_str = label_utils.sanitize_label(pool_info["name"])
                        task_str = label_utils.sanitize_label(label)
                        if pool_str == 'pool' or pool_str in self.config.TERMS_FOR_MISSING or task_str in self.config.TERMS_FOR_MISSING:
                            continue
                        observation = (pool_str, task_str,
                                       Observation.RESOURCE_CONTAINMENT, self.config.RESOURCE, self.config.BINARY, "")
                        observations.append(observation)

        return observations

    def get_observations_flat(self, observations) -> List[Dict[str, str]]:
        res = [{self.config.RECORD_ID: str(uuid.uuid4()),
                self.config.CONSTRAINT_STR: obs[-4],
                self.config.LEVEL: obs[-3],
                self.config.OPERATOR_TYPE: obs[-2],
                self.config.OBJECT: obs[-1],
                self.config.DICTIONARY: set(),
                self.config.DATA_OBJECT: set(),
                self.config.TEMPLATE: _get_constraint_template(obs[-4])
                } for obs in observations]
        res = self.add_operands(res)
        return res

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


