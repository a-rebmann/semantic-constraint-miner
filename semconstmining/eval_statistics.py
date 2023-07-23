from pathlib import Path
from statistics import mean

import pandas as pd

from semconstmining.config import Config
from semconstmining.declare.enums import Template
from semconstmining.evaluate import load_or_generate_logs_from_sap_sam
from semconstmining.main import get_resource_handler, get_or_mine_constraints
from semconstmining.parsing.label_parser.nlp_helper import NlpHelper

conf = Config(Path(__file__).parents[2].resolve(), "semantic_sap_sam_filtered")
conf.CONSTRAINT_TYPES_TO_IGNORE.extend(conf.NEGATIVE_TEMPLATES)
conf.CONSTRAINT_TYPES_TO_IGNORE.append(Template.INIT.templ_str)
conf.CONSTRAINT_TYPES_TO_IGNORE.append(Template.END.templ_str)


def compute_constraint_statistics():
    nlp_helper = NlpHelper(conf)
    resource_handler = get_resource_handler(conf, nlp_helper)
    base_constraints = pd.read_pickle(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "_eval_constraints.pkl"))
    types = []
    counts = []
    avg_count_per_model = []
    min_count_per_model = []
    max_count_per_model = []
    for const_type, constraints in base_constraints.groupby(by=conf.LEVEL):
        types.append(const_type)
        counts.append(len(constraints))
        avg_count_per_model.append(constraints[conf.MODEL_ID].value_counts().mean())
        min_count_per_model.append(constraints[conf.MODEL_ID].value_counts().min())
        max_count_per_model.append(constraints[conf.MODEL_ID].value_counts().max())
    df = pd.DataFrame({
        "number_of_models": len(base_constraints[conf.MODEL_ID].unique()),
        "type": types,
        "count": counts,
        "avg_count_per_model": avg_count_per_model,
        "min_count_per_model": min_count_per_model,
        "max_count_per_model": max_count_per_model
    })
    df.to_csv(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "constraint_statistics_per_model.csv"), index=False)
    agg_constraints = get_or_mine_constraints(conf, resource_handler, min_support=1)
    agg_constraints["unique"] = agg_constraints.apply(lambda x: x[conf.SUPPORT] == 1, axis=1)
    types = []
    counts = []
    avg_support = []
    min_support = []
    max_support = []
    count_unique = []
    count_not_unique = []
    for const_type, constraints in agg_constraints.groupby(by=conf.LEVEL):
        types.append(const_type)
        counts.append(len(constraints))
        avg_support.append(constraints[conf.SUPPORT].mean())
        min_support.append(constraints[conf.SUPPORT].min())
        max_support.append(constraints[conf.SUPPORT].max())
        count_unique.append(len(constraints[constraints["unique"]]))
        count_not_unique.append(len(constraints[~constraints["unique"]]))
    df = pd.DataFrame({
        "number_of_models": len(base_constraints[conf.MODEL_ID].unique()),
        "type": types,
        "count": counts,
        "avg_support": avg_support,
        "min_support": min_support,
        "max_support": max_support,
        "count_unique": count_unique,
        "count_not_unique": count_not_unique
    })
    df.to_csv(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "constraint_statistics.csv"), index=False)
    # for index, row in agg_constraints.iterrows():
    #     model_ids = [x.strip() for x in row[conf.MODEL_ID].split("|")]


def compute_log_statistics():
    nlp_helper = NlpHelper(conf)
    resource_handler = get_resource_handler(conf, nlp_helper)
    noisy_logs = load_or_generate_logs_from_sap_sam(resource_handler)
    # get number of logs
    log_ids = list(noisy_logs[conf.MODEL_ID].unique())
    # get average number of event classes per log
    avg_event_classes_orig = noisy_logs[conf.LOG].apply(
        lambda x: len(set([e[conf.XES_NAME] for t in x for e in t]))).mean()
    # get max number of event classes per log
    max_event_classes_orig = noisy_logs[conf.LOG].apply(
        lambda x: len(set([e[conf.XES_NAME] for t in x for e in t]))).max()
    # get average number of distinct objects per log
    avg_objects_orig = noisy_logs.apply(
        lambda x: len(resource_handler.components.all_objects_per_model[x[conf.MODEL_ID]]), axis=1).mean()
    # get max number of distinct objects per log
    max_objects_orig = noisy_logs.apply(
        lambda x: len(resource_handler.components.all_objects_per_model[x[conf.MODEL_ID]]), axis=1).max()
    # get average number of variants
    avg_variants_orig = noisy_logs[conf.LOG].apply(
        lambda x: len(set(tuple([e[conf.XES_NAME] for e in t]) for t in x))).mean()
    # get max number of variants
    max_variants_orig = noisy_logs[conf.LOG].apply(
        lambda x: len(set(tuple([e[conf.XES_NAME] for e in t]) for t in x))).max()
    # get average variant length
    avg_variant_length_orig = noisy_logs[conf.LOG].apply(
        lambda x: mean(len(tuple([e[conf.XES_NAME] for e in t])) for t in x) if len(x) > 0 else 0).mean()
    # get max variant length
    max_variant_length_orig = noisy_logs[conf.LOG].apply(
        lambda x: max(len(tuple([e[conf.XES_NAME] for e in t])) for t in x) if len(x) > 0 else 0).max()
    # get average number of event classes per log
    avg_event_classes_noisy = noisy_logs[conf.NOISY_LOG].apply(
        lambda x: len(set([e[conf.XES_NAME] for t in x for e in t]))).mean()
    # get max number of event classes per log
    max_event_classes_noisy = noisy_logs[conf.NOISY_LOG].apply(
        lambda x: len(set([e[conf.XES_NAME] for t in x for e in t]))).max()
    # get average number of distinct objects per log
    avg_objects_noisy = noisy_logs.apply(
        lambda x: len(resource_handler.components.all_objects_per_model[x[conf.MODEL_ID]]), axis=1).mean()
    # get max number of distinct objects per log
    max_objects_noisy = noisy_logs.apply(
        lambda x: len(resource_handler.components.all_objects_per_model[x[conf.MODEL_ID]]), axis=1).max()
    # get average number of variants
    avg_variants_noisy = noisy_logs[conf.NOISY_LOG].apply(
        lambda x: len(set(tuple([e[conf.XES_NAME] for e in t]) for t in x))).mean()
    # get max number of variants
    max_variants_noisy = noisy_logs[conf.NOISY_LOG].apply(
        lambda x: len(set(tuple([e[conf.XES_NAME] for e in t]) for t in x))).max()
    # get average variant length
    avg_variant_length_noisy = noisy_logs[conf.NOISY_LOG].apply(
        lambda x: mean(len(tuple([e[conf.XES_NAME] for e in t])) for t in x) if len(x) > 0 else 0).mean()
    # get max variant length
    max_variant_length_noisy = noisy_logs[conf.NOISY_LOG].apply(
        lambda x: max(len(tuple([e[conf.XES_NAME] for e in t])) for t in x) if len(x) > 0 else 0).max()
    stats_df_dict = {
        "Collection": ["Original", "Noisy"],
        "Number of Logs": [len(log_ids), len(log_ids)],
        "Average Number of Event Classes": [avg_event_classes_orig, avg_event_classes_noisy],
        "Max Number of Event Classes": [max_event_classes_orig, max_event_classes_noisy],
        "Average Number of Distinct Objects": [avg_objects_orig, avg_objects_noisy],
        "Max Number of Distinct Objects": [max_objects_orig, max_objects_noisy],
        "Average Number of Variants": [avg_variants_orig, avg_variants_noisy],
        "Max Number of Variants": [max_variants_orig, max_variants_noisy],
        "Average Variant Length": [avg_variant_length_orig, avg_variant_length_noisy],
        "Max Variant Length": [max_variant_length_orig, max_variant_length_noisy],
    }
    stats_df = pd.DataFrame(stats_df_dict)
    stats_df.to_csv(conf.DATA_EVAL / (conf.MODEL_COLLECTION + "log_statistics.csv"), index=False)


if __name__ == "__main__":
    compute_constraint_statistics()
    compute_log_statistics()
