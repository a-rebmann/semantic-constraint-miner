import uuid
from typing import List, Dict

import pandas as pd

from semconstmining.declare.enums import Template

rel_to_observation = {"antonymy": Template.EXCLUSIVE_CHOICE.templ_str,
                      "opposite-of": Template.EXCLUSIVE_CHOICE.templ_str,
                      # "can-result-in": Observation.ACT_CO_OCC,
                      "happens-before": Template.PRECEDENCE.templ_str}


def extract(config):
    input_file = config.DATA_RESOURCES / config.VERB_OCEAN_FILE
    # Parses csv and stores content in knowledge base
    candidates = set()
    observations = []
    with open(input_file) as f:
        line = f.readline()
        while line:
            if not line.startswith("#"):
                (verb1, rel, verb2, conf) = _line_to_tuple(line)
                if rel in rel_to_observation:
                    observation_type = rel_to_observation[rel]
                    candidates.add((verb1, verb2, observation_type))
            line = f.readline()
    added = set()
    # filter out false antonyms
    for (verb1, verb2, observation_type) in candidates:
        if observation_type in [Template.PRECEDENCE.templ_str]:  # , Observation.ACT_CO_OCC):
            # print('adding VO_rel:', verb1, verb2)
            observations.append((verb1, verb2, observation_type, config.LINGUISTIC, config.BINARY, ""))
        if observation_type is Template.EXCLUSIVE_CHOICE.templ_str:
            if (verb1, verb2, Template.PRECEDENCE.templ_str) not in candidates and (
                    verb2, verb1, Template.PRECEDENCE.templ_str) not in candidates:
                if (verb2, verb1, observation_type) not in added:
                    # print('adding VO_rel:', verb1, verb2)
                    observations.append((verb1, verb2, observation_type, config.LINGUISTIC, config.BINARY, ""))
                    added.add((verb1, verb2, observation_type, config.LINGUISTIC, config.BINARY, ""))
    print('finished populating based on VerbOcean')
    return (
        pd.DataFrame.from_records(get_observations_flat(observations)).assign(model_id="").assign(
            model_name="row_tuple.name")
    )


def get_observations_flat(config, observations) -> List[Dict[str, str]]:
    res = [{config.RECORD_ID: str(uuid.uuid4()),
            config.CONSTRAINT_STR: obs[-4],
            config.LEVEL: obs[-3],
            config.OPERATOR_TYPE: obs[-2],
            config.OBJECT: obs[-1]} for obs in observations]
    res = config.add_operands(res)
    return res


def add_operands(config, res):
    for rec in res:
        if rec[config.OPERATOR_TYPE] == config.UNARY:
            op_l = rec[config.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").strip()
            rec[config.config.LEFT_OPERAND] = op_l if op_l not in config.TERMS_FOR_MISSING else ""
        if rec[config.OPERATOR_TYPE] == config.BINARY:
            ops = rec[config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").split(",")
            op_l = ops[0].strip()
            op_r = ops[1].strip()
            rec[config.LEFT_OPERAND] = op_l if op_l not in config.TERMS_FOR_MISSING else ""
            rec[config.RIGHT_OPERAND] = op_r if op_r not in config.TERMS_FOR_MISSING else ""
    return res


def _line_to_tuple(line):
    start_br = line.find('[')
    end_br = line.find(']')
    conf_delim = line.find('::')
    verb1 = line[:start_br].strip()
    rel = line[start_br + 1: end_br].strip()
    verb2 = line[end_br + 1: conf_delim].strip()
    conf = line[conf_delim: len(line)].strip()
    return (verb1, rel, verb2, conf)
