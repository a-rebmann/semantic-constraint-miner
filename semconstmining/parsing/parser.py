import json
import logging
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from semconstmining.parsing.conversion.bpmnjsonanalyzer import fromJSON, is_relevant, get_type, \
    get_full_postset

_logger = logging.getLogger(__name__)
warnings.simplefilter('ignore')

max_rec = 0x100000
sys.setrecursionlimit(max_rec)
_logger.info(sys.getrecursionlimit())


def parse_dict_csv_raw(csv_path: Path, **kwargs):
    df = (
        pd.read_csv(csv_path, dtype={"Type": "category", "Category": "category"}, **kwargs)
        .rename(columns=lambda s: s.replace(" ", "_").lower())
        .set_index("id")
    )
    if df.index.duplicated().any():
        _logger.warning("csv has %d duplicate model ids", df.index.duplicated().sum())
    assert not df["category"].isna().any(), "csv has NA category entries, this should not happen."
    return df


def parse_csv_raw(csv_path: Path, **kwargs):
    df = (
        pd.read_csv(csv_path, dtype={"Type": "category", "Namespace": "category"}, **kwargs)
        .rename(columns=lambda s: s.replace(" ", "_").lower())
        .set_index("model_id")
    )
    if df.index.duplicated().any():
        _logger.warning("csv has %d duplicate model ids", df.index.duplicated().sum())
    assert not df["namespace"].isna().any(), "csv has NA namespace entries, this should not happen."
    return df

def _traverse_and_extract_data_object_relations(follows, labels):
    data_objects = {}
    for s in follows.keys():
        if is_relevant(s, labels, ()):
            postset = get_full_postset(labels, follows, s)
            for elem in postset:
                if elem not in follows.keys():
                    continue
                ty = get_type(elem, labels, ())
                if ty == "Object":
                    if s in data_objects:
                        data_objects[s].append(elem)
                    else:
                        data_objects[s] = list()
                        data_objects[s].append(elem)
    return data_objects


class BpmnModelParser:
    def __init__(self, config, parse_outgoing=False, parse_parent=False):
        self.config = config
        self.parse_outgoing = parse_outgoing
        self.parse_parent = parse_parent

    def parse_model_elements(self) -> pd.DataFrame:
        csv_paths = self.get_csv_paths()
        _logger.info("Starting to parse %d cvs", len(csv_paths))
        dfs = [self._parse_bpmn_model_elements_csv(p) for p in tqdm(csv_paths)]
        df = pd.concat(dfs)
        return df

    def _parse_bpmn_model_elements_csv(self, csv_path: Path) -> pd.DataFrame:
        df = parse_csv_raw(csv_path)
        df_bpmn = df.query(f"namespace == '{self.config.BPMN2_NAMESPACE}'")
        model_dfs = [self._parse_df_row(t) for t in df_bpmn.reset_index().itertuples()]
        return (
            pd.concat(model_dfs)
            .set_index([self.config.ELEMENT_ID])
            .astype({self.config.ELEMENT_CATEGORY: "category"})  # convert column category to dtype categorical to save memory
        )

    def parse_models(self, csv_paths=None, filter_df=None):
        if csv_paths is None:
            csv_paths = self.get_csv_paths()
        _logger.info("Starting to parse %d CSVs", len(csv_paths))
        dfs = [self._parse_models(p, filter_df) for p in tqdm(csv_paths)]
        df = pd.concat(dfs)
        return df

    def _parse_models(self, csv_path, filter_df):
        df = parse_csv_raw(csv_path)
        if filter_df is not None:
            df_bpmn = df.query(f"namespace == '{self.config.BPMN2_NAMESPACE}'")
            # print(df_bpmn.index.values)
            _logger.info("There are %d BPMNs", len(df_bpmn))
            df_bpmn = df_bpmn[df_bpmn.index.isin(list(filter_df[self.config.MODEL_ID].unique()))]
            # print(filter_df.index.values)
            _logger.info("There are %d BPMNs after filtering out non-english ones.", len(df_bpmn))
        else:
            df_bpmn = df.query(f"namespace == '{self.config.BPMN2_NAMESPACE}'")
        return df_bpmn

    def _parse_df_row(self, row_tuple):
        # print(row_tuple.model_json)
        model_dict = json.loads(row_tuple.model_json)
        # print(model_dict)
        elements = self._get_elements_flat(model_dict, row_tuple.model_id)
        return (
            pd.DataFrame.from_records(elements)
            .assign(model_id=row_tuple.model_id)
        )

    def _get_elements_flat(self, model_dict, model_id) -> List[Dict[str, str]]:
        """
        Parses the recursive childShapes and produces a flat list of model elements with the most important attributes
        such as id, category, label, outgoing, and parent elements.
        """
        elements_flat = []
        try:
            f, l, _ = fromJSON(model_dict)
        except KeyError as e:
            _logger.warning("Could not parse model %s, skipping", model_id)
            return elements_flat
        follows = {}
        labels = {}
        for e in l:
            labels[model_id+str(e)] = l[e]
        for e in f:
            follows[model_id+str(e)] = [model_id+str(e) for e in f[e]]
        stack = deque([model_dict])
        data_object_relations = _traverse_and_extract_data_object_relations(follows, labels)
        while len(stack) > 0:
            element = stack.pop()
            for c in element.get("childShapes", []):
                c["parent"] = element["resourceId"]
                stack.append(c)

            # don't append root as element
            if element["resourceId"] == model_dict["resourceId"]:
                continue
            element_id = model_id + str(element["resourceId"])
            # NOTE: it's possible to add other attributes here, such as the bounds of an element
            record = {
                self.config.ELEMENT_ID: element_id,
                self.config.ELEMENT_ID_BACKUP: element_id,
                self.config.ELEMENT_CATEGORY: element["stencil"].get("id") if "stencil" in element else None,
                self.config.LABEL: element["properties"].get("name"),
                self.config.GLOSSARY: json.dumps(element["glossaryLinks"]) if "glossaryLinks" in element else "{}",
                self.config.DATA_OBJECT: data_object_relations[element_id] if element_id in data_object_relations else [],
            }
            if self.parse_parent:
                record["parent"] = element.get("parent")
            if self.parse_outgoing:
                record["outgoing"] = [v for d in element.get("outgoing", []) for v in d.values()]

            elements_flat.append(record)

        return elements_flat

    def get_csv_paths(self) -> List[Path]:
        paths = sorted(self.config.DATA_DATASET.glob("*.csv"))
        assert len(paths) > 0, f"Could not find any csv in {self.config.DATA_DATASET.absolute()}, have you downloaded the dataset?"
        _logger.info("Found %d csvs", len(paths))
        return paths

    def parse_model_metadata(self, csv_paths=None) -> pd.DataFrame:
        if csv_paths is None:
            csv_paths = self.get_csv_paths()
        _logger.info("Starting to parse %d csv excluding model json", len(csv_paths))
        # exclude "Model JSON" column to speed up import and reduce memory usage
        dfs = [parse_csv_raw(p, usecols=lambda s: s != "Model JSON") for p in tqdm(csv_paths)]
        df = pd.concat(dfs)
        _logger.info("Parsed %d models", len(df))
        return df
