from pathlib import Path

from semconstmining.config import Config
from semconstmining.main import get_resource_handler, get_or_mine_constraints
from semconstmining.parsing.label_parser.nlp_helper import NlpHelper
from semconstmining.selection.instantiation.filter_config import FilterConfig
from semconstmining.selection.instantiation.recommendation_config import RecommendationConfig

conf = Config(Path(__file__).parents[2].resolve(), "opal")
filt_config = FilterConfig(conf)
rec_config = RecommendationConfig(conf)

priority_list = [conf.SUPPORT]
max_per_category = 1000


def apply_on_real_data():
    nlp_helper = NlpHelper(conf)
    resource_handler = get_resource_handler(conf, nlp_helper=nlp_helper)
    constraints = get_or_mine_constraints(conf, resource_handler)
