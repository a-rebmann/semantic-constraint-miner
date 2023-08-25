import logging
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from semconstmining.config import Config
from semconstmining.declare.enums import Template
from semconstmining.main import get_resource_handler
from semconstmining.parsing.label_parser.nlp_helper import NlpHelper
from semconstmining.parsing.parser import BpmnModelParser

conf = Config(Path(__file__).parents[2].resolve(), "sap_sam_filtered")
conf.CONSTRAINT_TYPES_TO_IGNORE.extend(conf.NEGATIVE_TEMPLATES)
conf.CONSTRAINT_TYPES_TO_IGNORE.remove(Template.NOT_CO_EXISTENCE.templ_str)
conf.CONSTRAINT_TYPES_TO_IGNORE.append(Template.INIT.templ_str)
conf.CONSTRAINT_TYPES_TO_IGNORE.append(Template.END.templ_str)

_logger = logging.getLogger(__name__)


def prepare_subset():
    max_models = 2500
    nlp_helper = NlpHelper(conf)
    resource_handler = get_resource_handler(conf, nlp_helper)
    model_ids = random.sample(list(resource_handler.components.all_objects_per_model.keys()), max_models)
    parser = BpmnModelParser(conf)
    paths = parser.get_csv_paths()
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        dfs.append(df[df["Model ID"].isin(model_ids)])
    df = pd.concat(dfs)
    df.to_csv(conf.DATA_EVAL / "sap_sam_filtered_2500.csv", index=False)


def prepare_semantic_dataset():
    max_labels_in_sentence = 3
    max_models = 2500
    nlp_helper = NlpHelper(conf)
    resource_handler = get_resource_handler(conf, nlp_helper)
    model_sentences = {}
    for model_id in tqdm(resource_handler.components.all_objects_per_model):
        # for model_id, group in tqdm(resource_handler.bpmn_model_elements.groupby(conf.MODEL_ID)):
        # labels = list(group[group[conf.ELEMENT_CATEGORY] == "Task"][conf.CLEANED_LABEL].dropna().unique())
        labels = resource_handler.components.all_objects_per_model[model_id]
        labels = [label for i, label in enumerate(labels) if
                  label not in conf.TERMS_FOR_MISSING and i < max_labels_in_sentence]
        model_sentences[model_id] = " and ".join(labels)
    cluster_to_ids = nlp_helper.cluster(model_sentences)
    _logger.info(f"Number of clusters: {len(cluster_to_ids)}")
    # pick the same number of ids from each cluster such that the maximum number of ids is max_models
    ids = []
    for cluster, cluster_ids in cluster_to_ids.items():
        end_idx = min(max_models // len(cluster_to_ids), len(cluster_ids) - 1)
        ids.extend(cluster_ids[:end_idx])
    _logger.info(f"Number of ids: {len(ids)}")
    parser = BpmnModelParser(conf)
    paths = parser.get_csv_paths()
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        dfs.append(df[df["Model ID"].isin(ids)])
    df = pd.concat(dfs)
    df.to_csv(conf.DATA_EVAL / "semantic_sap_sam_filtered.csv", index=False)


if __name__ == '__main__':
    # prepare_subset()
    prepare_semantic_dataset()
