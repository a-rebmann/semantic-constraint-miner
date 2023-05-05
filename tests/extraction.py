import json
from pathlib import Path

import pandas as pd

from semconstmining.config import Config
from semconstmining.main import get_resource_handler, get_or_mine_constraints, get_context_sim_computer
from semconstmining.parsing.label_parser.nlp_helper import NlpHelper

config = Config(Path(__file__).parents[2].resolve(), "test_data")


def test_example_extraction():
    nlp_helper = NlpHelper(config)
    resource_handler = get_resource_handler(config, nlp_helper)
    all_constraints = get_or_mine_constraints(config, resource_handler, min_support=1)
    get_context_sim_computer(config, all_constraints, nlp_helper, resource_handler)
    print("Done!")

