from pathlib import Path

from semconstmining.config import Config
from semconstmining.main import get_resource_handler

conf = Config(Path(__file__).parents[2].resolve(), "sap_sam_2022_preprocessed")


def preprocess_sap_sam():
    resource_handler = get_resource_handler(conf)



def generate_logs_from_sap_sam():
    pass


def run_configurations():
    pass
