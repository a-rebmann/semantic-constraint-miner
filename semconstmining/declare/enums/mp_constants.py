from enum import Enum

from semconstmining.mining.model.constraint import Observation


class Template(str, Enum):

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = str.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, templ_str: str, is_binary: bool, is_negative: bool, supports_cardinality: bool):
        self.templ_str = templ_str
        self.is_binary = is_binary
        self.is_negative = is_negative
        self.supports_cardinality = supports_cardinality

    EXISTENCE   = "Existence",  False, False, True
    ABSENCE     = "Absence",    False, False, True
    EXACTLY     = "Exactly",    False, False, True

    INIT        = "Init",       False, False, False
    END         = "End",        False, False, False

    CHOICE              = "Choice",                 True, False, False
    EXCLUSIVE_CHOICE    = "Exclusive Choice",       True, False, False
    RESPONDED_EXISTENCE = "Responded Existence",    True, False, False
    RESPONSE            = "Response",               True, False, False
    ALTERNATE_RESPONSE  = "Alternate Response",     True, False, False
    CHAIN_RESPONSE      = "Chain Response",         True, False, False
    PRECEDENCE          = "Precedence",             True, False, False
    ALTERNATE_PRECEDENCE = "Alternate Precedence",  True, False, False
    CHAIN_PRECEDENCE    = "Chain Precedence",       True, False, False

    SUCCESSION              = "Succession",             True, False, False
    ALTERNATE_SUCCESSION    = "Alternate Succession",   True, False, False
    CHAIN_SUCCESSION        = "Chain Succession",       True, False, False
    CO_EXISTENCE            = "Co-Existence",           True, False, False

    NOT_CO_EXISTENCE        = "Not Co-Existence",       True, True, False
    NOT_RESPONDED_EXISTENCE = "Not Responded Existence",    True, True, False
    NOT_RESPONSE            = "Not Response",               True, True, False
    NOT_CHAIN_RESPONSE      = "Not Chain Response",         True, True, False
    NOT_PRECEDENCE          = "Not Precedence",             True, True, False
    NOT_CHAIN_PRECEDENCE    = "Not Chain Precedence",       True, True, False

    NOT_SUCCESSION          = "Not Succession",             True, True, False
    NOT_ALTERNATE_SUCCESSION= "Not Alternate Succession",   True, True, False
    NOT_CHAIN_SUCCESSION    = "Not Chain Succession",       True, True, False

    @classmethod
    def get_template_from_string(cls, template_str):
        return next(filter(lambda t: t.templ_str == template_str, Template), None)

    @classmethod
    def get_unary_templates(cls):
        return tuple(filter(lambda t: not t.is_binary, Template))

    @classmethod
    def get_binary_templates(cls):
        return tuple(filter(lambda t: t.is_binary, Template))

    @classmethod
    def get_positive_templates(cls):
        return tuple(filter(lambda t: not t.is_negative, Template))

    @classmethod
    def get_negative_templates(cls):
        return tuple(filter(lambda t: t.is_negative, Template))

    @classmethod
    def get_cardinality_templates(cls):
        return tuple(filter(lambda t: t.supports_cardinality, Template))


existence_based_on = {
    Template.ABSENCE.templ_str: None,
    Template.EXISTENCE.templ_str: None,
    Template.EXACTLY.templ_str: None,
    Template.INIT.templ_str: None,
    Template.END.templ_str: None
}

relation_based_on = {
    Template.RESPONSE.templ_str: Template.RESPONDED_EXISTENCE.templ_str,
    Template.ALTERNATE_RESPONSE.templ_str: Template.RESPONSE.templ_str,
    Template.CHAIN_RESPONSE.templ_str: Template.ALTERNATE_RESPONSE.templ_str,
    Template.ALTERNATE_PRECEDENCE.templ_str: Template.PRECEDENCE.templ_str,
    Template.CHAIN_PRECEDENCE.templ_str: Template.ALTERNATE_PRECEDENCE.templ_str,

    Template.NOT_RESPONDED_EXISTENCE.templ_str: Template.NOT_SUCCESSION.templ_str,
    Template.NOT_RESPONSE.templ_str: Template.NOT_CHAIN_RESPONSE.templ_str,
    Template.NOT_PRECEDENCE.templ_str: Template.NOT_CHAIN_PRECEDENCE.templ_str,
    Template.NOT_SUCCESSION.templ_str: Template.NOT_ALTERNATE_SUCCESSION.templ_str,
    Template.NOT_ALTERNATE_SUCCESSION.templ_str: Template.NOT_CHAIN_SUCCESSION.templ_str,
}

opponent_constraint = {
    Template.NOT_CHAIN_PRECEDENCE.templ_str: Template.CHAIN_PRECEDENCE.templ_str,
    Template.NOT_PRECEDENCE.templ_str: Template.PRECEDENCE.templ_str,
    Template.NOT_RESPONSE.templ_str: Template.RESPONSE.templ_str,
    Template.NOT_CHAIN_RESPONSE.templ_str: Template.CHAIN_RESPONSE.templ_str,
    Template.NOT_RESPONDED_EXISTENCE: Template.RESPONDED_EXISTENCE.templ_str,

    #Template.NOT_SUCCESSION: Template.SUCCESSION
}

nat_lang_templates = {
    Template.ABSENCE.templ_str: "{1} does not occur more than {n} times",
    Template.EXISTENCE.templ_str: "{1} occurs at least {n} times",
    Template.EXACTLY.templ_str: "{1} occurs exactly {n} times",
    Template.INIT.templ_str: "{1} is the first to occur",
    Template.END.templ_str: "{1} is the last to occur",

    Template.CHOICE.templ_str: "{1} or {2} or both eventually occur in the same process instance",
    Template.EXCLUSIVE_CHOICE.templ_str: "{1} or {2} occurs, but never both in the same process instance",
    Template.RESPONDED_EXISTENCE.templ_str: "if {1} occurs in the process instance, then {2} occurs as well",
    Template.RESPONSE.templ_str: "if {1} occurs, then {2} occurs at some point after {1}",
    Template.ALTERNATE_RESPONSE.templ_str: "each time {1} occurs, then {2} occurs afterwards, and no other "
                                           "{1} recurs in between",
    Template.CHAIN_RESPONSE.templ_str: "each time {1} occurs, then {2} occurs immediately afterwards",
    Template.PRECEDENCE.templ_str: "{2} occurs only if it is preceded by {1}",
    Template.ALTERNATE_PRECEDENCE.templ_str: "each time {2} occurs, it is preceded by {1} and no other "
                                             "{2} can recur in between",
    Template.CHAIN_PRECEDENCE.templ_str: "each time {2} occurs, then {1} occurs immediately beforehand",

    Template.SUCCESSION.templ_str: "{1} occurs if and only if it is followed by {2}",
    Template.ALTERNATE_SUCCESSION.templ_str: "{1} and {2} occur if and only if the latter follows the former, "
                                             "and they alternate in a process instance",
    Template.CHAIN_SUCCESSION.templ_str: "{1} and {2} occur if and only if the latter immediately follows the former",
    Template.CO_EXISTENCE.templ_str: "if {1} occurs, then {2} occurs as well, and vice versa",

    Template.NOT_RESPONDED_EXISTENCE.templ_str: "only one of {1} and {2} can occur in a process instance, but not both",
    Template.NOT_CHAIN_PRECEDENCE.templ_str: "when {2} occurs, {1} did not occur immediately beforehand",
    Template.NOT_PRECEDENCE.templ_str:  "when {2} occurs, {1} did not occur beforehand",
    Template.NOT_RESPONSE.templ_str: "when {1} occurs, {2} cannot not occur afterwards",
    Template.NOT_CHAIN_RESPONSE.templ_str: "when {1} occurs, {2} cannot not occur immediately afterwards",
    Template.NOT_SUCCESSION.templ_str: "{2} cannot occur after {1}",
    Template.NOT_CO_EXISTENCE.templ_str: "if {1} occurs, then {2} cannot occur, and vice versa",
    Observation.RESOURCE_TASK_EXISTENCE.value: "{2} is performed by {1}",
    Observation.RESOURCE_CONTAINMENT.value: "{2} is part of {1}",
    "": ""
}

regex_representations = {
    Template.ABSENCE.templ_str: r'\[^a]*(a[^a]*){0,m}+[^a]*',
    Template.EXISTENCE.templ_str: r'[^a]*(a[^a]*){n,}+[^a]*',
    Template.EXACTLY.templ_str: r'[^a]*(a[^a]*)+[^a]*',
    Template.INIT.templ_str: r'a.*',
    Template.END.templ_str: r'.*a',

    Template.CHOICE.templ_str: r'',
    Template.EXCLUSIVE_CHOICE.templ_str: r'[^ab]*((a[^b]*)|(b[^a]*))?',
    Template.RESPONDED_EXISTENCE.templ_str: r'[^a]*((a.*b.*)|(b.*a.*))*[^a]*',
    Template.RESPONSE.templ_str: r'[^a]*(a.*b)*[^a]*',
    Template.ALTERNATE_RESPONSE.templ_str: r'[^a]*(a[^a]*b[^a]*)*[^a]*',
    Template.CHAIN_RESPONSE.templ_str: r'[^a]*(ab[^a]*)*[^a]*',
    Template.PRECEDENCE.templ_str: r'[^b]*(a.*b)*[^b]*',
    Template.ALTERNATE_PRECEDENCE.templ_str: r'[^b]*(a[^b]*b[^b]*)*[^b]*',
    Template.CHAIN_PRECEDENCE.templ_str: r'[^b]*(ab[^b]*)*[^b]*',

    Template.SUCCESSION.templ_str: r'[^ab]*(a.*b)*[^ab]*',
    Template.ALTERNATE_SUCCESSION.templ_str: r'[^ab]*(a[^ab]*b[^ab]*)*[^ab]*',
    Template.CHAIN_SUCCESSION.templ_str: r'[^ab]*(ab[^ab]*)*[^ab]*',
    Template.CO_EXISTENCE.templ_str: r'[^ab]*((a.*b.*)|(b.*a.*))*[^ab]*',

    Template.NOT_RESPONDED_EXISTENCE.templ_str: r'',
    Template.NOT_CHAIN_PRECEDENCE.templ_str: r'',
    Template.NOT_PRECEDENCE.templ_str:  r'',
    Template.NOT_RESPONSE.templ_str: r'',
    Template.NOT_CHAIN_RESPONSE.templ_str: r'',
    Template.NOT_SUCCESSION.templ_str: r'[^a]*(a[^b]*)*[^ab]*',
    Template.NOT_CO_EXISTENCE.templ_str: r'',
    Observation.RESOURCE_TASK_EXISTENCE.value: r'',
    Observation.RESOURCE_CONTAINMENT.value: r'',
    "": ""
}


class TraceState(str, Enum):
    VIOLATED = "Violated"
    SATISFIED = "Satisfied"
    POSSIBLY_VIOLATED = "Possibly Violated"
    POSSIBLY_SATISFIED = "Possibly Satisfied"
