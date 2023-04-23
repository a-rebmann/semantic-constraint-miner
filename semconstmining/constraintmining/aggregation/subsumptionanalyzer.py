"""
Class to analyze a set of constraints wrt subsumption, i.e., determining which constrains cover which other ones.
@author Adrian Rebmann <adrianrebmann@gmail.com>
"""
import logging
import uuid
from copy import deepcopy

import pandas as pd
from pandas import DataFrame

from semconstmining.declare.enums import Template, relation_based_on
from semconstmining.declare.parsers import parse_single_constraint

_logger = logging.getLogger(__name__)


def _construct_constraint(constraint, other: Template):
    constraint_str = other.templ_str
    if constraint['template'].supports_cardinality:
        constraint_str += str(constraint['n'])
    constraint_str += '[' + ", ".join(constraint["activities"]) + '] |' + ' |'.join(
        constraint["condition"])
    return constraint_str


def _reverse_constraint(constraint):
    constraint_str = constraint['template'].templ_str
    if constraint['template'].supports_cardinality:
        constraint_str += str(constraint['n'])
    constraint["activities"].reverse()
    constraint_str += '[' + ", ".join(constraint["activities"]) + '] |' + ' |'.join(
        constraint["condition"])
    return constraint_str


class SubsumptionAnalyzer:

    def __init__(self, config, constraints: DataFrame):
        self.config = config
        self.constraints = constraints
        self.constraints[self.config.REDUNDANT] = False

    def check_subsumption(self):
        # First, we get all possible operands (activities, objects, actions)
        #all_operands = set()
        #all_operands.update(self.constraints[self.config.LEFT_OPERAND].dropna().unique())
        #all_operands.update(self.constraints[self.config.RIGHT_OPERAND].dropna().unique())
        #for operand in all_operands:

            # get all constraints that involve the current operand

        constraints = self.constraints[(~self.constraints[
                 self.config.REDUNDANT])]  # | (self.constraints[RIGHT_OPERAND] == operand)
        non_red_const = len(constraints)
        counter = 0
        for constraint_tuple in constraints.itertuples():
            constraint = parse_single_constraint(constraint_tuple.constraint_string)
            counter += 1
            if constraint is None:
                continue
            # for every constraint, check subsumption
            if self.config.POLICY == self.config.EAGER_ON_SUPPORT_OVER_HIERARCHY:
                tmp = constraint["template"]
                # check if there is a higher-level constraint
                while tmp.templ_str in relation_based_on and relation_based_on[tmp.templ_str] is not None:
                    tmp_str = relation_based_on[tmp.templ_str]
                    constraint_str = tmp_str
                    if constraint['template'].supports_cardinality:
                        constraint_str += str(constraint['n'])
                    constraint_str += '[' + ", ".join(constraint["activities"]) + '] |' + ' |'.join(
                        constraint["condition"])
                    to_mark = constraints[(constraints[self.config.CONSTRAINT_STR] == constraint_str) & (
                            constraints[self.config.OBJECT] == constraint_tuple.Object) & (
                                                  self.constraints[self.config.LEVEL] == constraint_tuple.Level)]
                    for i, row in to_mark.iterrows():
                        # check if the support of the higher-level constraint is the same and if so,
                        # mark that one as redundant!
                        if row[self.config.SUPPORT] == constraint_tuple.support:
                            self.constraints.at[i, self.config.REDUNDANT] = True
                    tmp = Template.get_template_from_string(tmp_str)
            #print("Checked " + str(counter) + " of " + str(non_red_const) + " constraints.")

    def check_equal(self):
        """
        Goes through all provided constraints and marks redundant, i.e., equal constraints.
        :return: Nothing, but, changes the constraint DataFrame
        """
        done = set()
        for idx, constraint_row in self.constraints.iterrows():
            if idx not in done and not constraint_row[self.config.REDUNDANT]:
                constraint = parse_single_constraint(constraint_row[self.config.CONSTRAINT_STR])
                if constraint["template"].templ_str in [Template.CHOICE.templ_str, Template.EXCLUSIVE_CHOICE.templ_str,
                                                        Template.CO_EXISTENCE.templ_str]:

                    reversed_const_str = _reverse_constraint(constraint)
                    to_mark = self.constraints[(self.constraints[self.config.CONSTRAINT_STR] == reversed_const_str) & (
                            self.constraints[self.config.OBJECT] == constraint_row[self.config.OBJECT]) & (
                                                       self.constraints[self.config.LEVEL] == constraint_row[
                                                   self.config.LEVEL])]
                    if len(to_mark) > 0:
                        if len(to_mark) > 1:
                            _logger.warning("More than one constraint matched " + reversed_const_str)
                        for i, row in to_mark.iterrows():
                            if row[self.config.SUPPORT] == constraint_row[self.config.SUPPORT]:
                                self.constraints.at[
                                    i, self.config.REDUNDANT] = True  # sets the reverse constraint as redundant
                                done.add(i)

    def check_refinement(self):
        """
        Goes through all provided constraints and identifies if the set can be refined.
        Here 'refining' means that two individual constraints can be represented by one (more specific) one.
        These more specific constraints are hence added to the list of constraints, while the individual ones
        are marked as redundant.
        :return: Nothing, but, changes the constraint DataFrame
        """
        new_constraints = pd.DataFrame()
        for idx, constraint_row in self.constraints.iterrows():
            if not constraint_row[self.config.REDUNDANT]:
                constraint = parse_single_constraint(constraint_row[self.config.CONSTRAINT_STR])
                if constraint is None:
                    continue
                # ChainSuccession(a, b) == ChainResponse(a, b) & & ChainPrecedence(a, b)
                if constraint["template"].templ_str == Template.CHAIN_RESPONSE.templ_str:
                    other_const_str = _construct_constraint(constraint, Template.CHAIN_PRECEDENCE)
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, other_const_str,
                                                          new_constraints,
                                                          Template.CHAIN_SUCCESSION)
                    continue
                if constraint["template"].templ_str == Template.CHAIN_PRECEDENCE.templ_str:
                    other_const_str = _construct_constraint(constraint, Template.CHAIN_RESPONSE)
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, other_const_str,
                                                          new_constraints,
                                                          Template.CHAIN_SUCCESSION)
                    continue
                # AlternateSuccession(a, b) == AlternateResponse(a, b) & & AlternatePrecedence(a, b)#
                if constraint["template"].templ_str == Template.ALTERNATE_RESPONSE.templ_str:
                    other_const_str = _construct_constraint(constraint, Template.ALTERNATE_PRECEDENCE)
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, other_const_str,
                                                          new_constraints,
                                                          Template.ALTERNATE_SUCCESSION)
                    continue
                if constraint["template"].templ_str == Template.ALTERNATE_PRECEDENCE.templ_str:
                    other_const_str = _construct_constraint(constraint, Template.ALTERNATE_RESPONSE)
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, other_const_str,
                                                          new_constraints,
                                                          Template.ALTERNATE_SUCCESSION)
                    continue
                # Succession(a, b) == Response(a, b) & & Precedence(a, b)
                if constraint["template"].templ_str == Template.RESPONSE.templ_str:
                    other_const_str = _construct_constraint(constraint, Template.PRECEDENCE)
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, other_const_str,
                                                          new_constraints,
                                                          Template.SUCCESSION)
                    continue
                if constraint["template"].templ_str == Template.PRECEDENCE.templ_str:
                    other_const_str = _construct_constraint(constraint, Template.RESPONSE)
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, other_const_str,
                                                          new_constraints,
                                                          Template.SUCCESSION)
                    continue
                # CoExistence(a, b) == RespondedExistence(a, b) & & RespondedExistence(b, a)
                if constraint["template"].templ_str == Template.RESPONDED_EXISTENCE.templ_str:
                    other_const_str = constraint["template"].templ_str
                    if constraint['template'].supports_cardinality:
                        other_const_str += str(constraint['n'])
                    other_const_str += '[' + ", ".join(reversed(constraint["activities"])) + '] |' + ' |'.join(
                        constraint["condition"])
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, other_const_str,
                                                          new_constraints,
                                                          Template.CO_EXISTENCE)
                    continue

                # Init(a) && End(a)
                if constraint["template"].templ_str == Template.INIT.templ_str:
                    other_const_str = _construct_constraint(constraint, Template.END)
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, other_const_str,
                                                          new_constraints,
                                                          Template.EXISTENCE, add_const=False)
                    continue
                if constraint["template"].templ_str == Template.END.templ_str:
                    other_const_str = _construct_constraint(constraint, Template.INIT)
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, other_const_str,
                                                          new_constraints,
                                                          Template.EXISTENCE, add_const=False)
                    continue
                # Existence(a, 1) && Absence(a, 2) == Exactly(a, 1)
                if constraint["template"].templ_str == Template.EXACTLY.templ_str:
                    first_const_str = Template.ABSENCE.templ_str
                    first_const_str += str(int(constraint['n']) + 1)
                    first_const_str += '[' + ", ".join(constraint["activities"]) + '] |' + ' |'.join(
                        constraint["condition"])
                    second_const_str = Template.EXISTENCE.templ_str
                    second_const_str += str(constraint['n'])
                    second_const_str += '[' + ", ".join(constraint["activities"]) + '] |' + ' |'.join(
                        constraint["condition"])
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, first_const_str,
                                                          new_constraints,
                                                          Template.EXACTLY, add_const=False)
                    new_constraints = self._check_and_add(idx, constraint_row, constraint, second_const_str,
                                                          new_constraints,
                                                          Template.EXACTLY, add_const=False)
                    continue

        # merge the new constraints into the existing ones
        print("New constraints: " + str(len(new_constraints)))

        self.constraints = pd.concat([self.constraints.reset_index(), new_constraints]).set_index(["obs_id"])

    def _check_and_add(self, constraint_row_idx, constraint_row, constraint, other_const_str, new_constraints,
                       template, add_const=True):
        to_mark = self.constraints[(self.constraints[self.config.CONSTRAINT_STR] == other_const_str) & (
                self.constraints[self.config.OBJECT] == constraint_row[self.config.OBJECT]) & (
                                           self.constraints[self.config.LEVEL] == constraint_row[self.config.LEVEL])]
        if len(to_mark) > 0:
            row_copy = deepcopy(constraint_row)
            if len(to_mark) > 1:
                _logger.warning("More than one constraint matched " + other_const_str)
            for i, row in to_mark.iterrows():
                if not add_const:
                    # check if the support of the other constraint is the same or smaller and if so,
                    # mark the other as redundant.
                    if row[self.config.SUPPORT] <= constraint_row[self.config.SUPPORT]:
                        self.constraints.at[i, self.config.REDUNDANT] = True
                        continue
                # check if the support of the higher-level constraint is the same and if so,
                # mark both as redundant and add to new one!
                if row[self.config.SUPPORT] == constraint_row[self.config.SUPPORT]:
                    self.constraints.at[i, self.config.REDUNDANT] = True  # sets the opponent constraint as redundant
                    if add_const:
                        self.constraints.at[
                            constraint_row_idx, self.config.REDUNDANT] = True  # sets the current constraint as redundant
                        new_const_str = _construct_constraint(constraint, template)
                        row_copy[self.config.CONSTRAINT_STR] = new_const_str
                        row_copy[self.config.RECORD_ID] = str(uuid.uuid4())
                        return new_constraints.append(row_copy, ignore_index=True)
        return new_constraints

    # def _check_and_add(self, constraint_row_idx, constraint_row, constraint, other_const_str, new_constraints,
    #                    template):
    #     # if template.templ_str in self.config.CONSTRAINT_TYPES_TO_IGNORE:
    #     #     return new_constraints
    #     to_mark = self.constraints[(self.constraints[self.config.CONSTRAINT_STR] == other_const_str) & (
    #             self.constraints[self.config.OBJECT] == constraint_row[self.config.OBJECT])
    #                                & (self.constraints[self.config.LEVEL] == constraint_row[self.config.LEVEL])]
    #     if len(to_mark) > 0:
    #         row_copy = deepcopy(constraint_row)
    #         if len(to_mark) > 1:
    #             _logger.warning("More than one constraint matched " + other_const_str)
    #         for i, row in to_mark.iterrows():
    #             # check if the support of the higher-level constraint is the same and if so,
    #             # mark both as redundant and add to new one!
    #             if row[self.config.SUPPORT] == constraint_row[self.config.SUPPORT]:
    #                 self.constraints.at[i, self.config.REDUNDANT] = True  # sets the opponent constraint as redundant
    #                 self.constraints.at[constraint_row_idx, self.config.REDUNDANT] = True  # sets the current constraint as redundant
    #                 new_const_str = _construct_constraint(constraint, template)
    #                 row_copy[self.config.CONSTRAINT_STR] = new_const_str
    #                 row_copy[self.config.RECORD_ID] = str(uuid.uuid4())
    #                 new_constraints.append(row_copy, ignore_index=True)
    #                 return new_constraints
    #     return new_constraints
