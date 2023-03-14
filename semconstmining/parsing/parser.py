import concurrent.futures
import warnings

from func_timeout import func_timeout, FunctionTimedOut
import gc
import json
import logging
import os
import sys
import time
from collections import deque
from os.path import exists
from pathlib import Path
from typing import List, Dict
import signal

import numpy as np
import pandas as pd
import pm4py
import psutil
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.simulation.playout.petri_net.algorithm import Variants
from pm4py.objects.log.obj import EventLog
from tqdm import tqdm

from semconstmining.constraintmining.conversion.jsontopetrinetconverter import JsonToPetriNetConverter
from semconstmining.constraintmining.conversion.bpmnjsonanalyzer import fromJSON

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


def create_log_without_loops(log):
    log_no_loops = EventLog()
    for trace in log:
        trace_labels = [x["concept:name"] for x in trace]
        if 0 < len(trace_labels) == len(set(trace_labels)):
            log_no_loops.append(trace)
    return log_no_loops


class BpmnModelParser:
    def __init__(self, config, parse_outgoing=False, parse_parent=False):
        self.config=config
        self.parse_outgoing = parse_outgoing
        self.parse_parent = parse_parent
        self.converter = JsonToPetriNetConverter()
        self.done = 0

    def soundness_check(self, row):
        if row.pn:
            _logger.info("Soundness check. " + row.model_id + "; Number " + str(self.done))
            net, im, fm = row.pn
            start = time.time()
            try:
                res = func_timeout(self.config.TIMEOUT, woflan.apply,
                                   args=(net, im, fm, {woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                       woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                       woflan.Parameters.RETURN_DIAGNOSTICS: False}))
                _logger.info("Result: " + row.model_id + " sound? " + str(res))
            except FunctionTimedOut as ex:
                _logger.warning("Time out during soundness checking.")
                res = False
            except IndexError as ex:
                _logger.warning("Error during soundness checking.")
                res = False
            finally:
                stop = time.time()
                completed_in = round(stop - start, 2)
                if completed_in > 2 * self.config.TIMEOUT:
                    _logger.error("Timeout not working!! " + row.model_id)
            self.done += 1
            if self.done % 1000 == 0:
                _logger.info("Collect garbage...")
                gc.collect()
                _logger.info("GC done.")
            return res
        else:
            return False

    def convert_models_to_pn_df(self, df_bpmn):
        success = 0
        failed = 0
        ids = []
        pns = []
        follows = []
        labels = []
        tasks = []
        names = []
        for row in df_bpmn.reset_index().itertuples():
            # add the id and name of the model to the new frame for future reference
            ids.append(row.model_id)
            names.append(row.name)
            json_str = self._get_json_from_row(row)
            try:
                f, l, t = fromJSON(json_str)
                net, initial_marking, final_marking = self.converter.convert_from_parsed(f, l, t)
                follows.append(f)
                labels.append(l)
                tasks.append(t)
                pns.append((net, initial_marking, final_marking))
                success += 1
            except Exception:
                _logger.debug("Error during conversion from bpmn to Petri net.")
                pns.append(None)
                follows.append(None)
                labels.append(None)
                tasks.append(None)
                failed += 1
        _logger.info(str(success + failed) + " jsons done. Success: " + str(success) + " failed: " + str(failed))
        return pd.DataFrame(
            {"model_id": ids, "pn": pns, "follows": follows, "labels": labels, "tasks": tasks, "name": names})

    def generate_logs_from_petri(self, df_petri):
        df_petri["sound"] = False
        success = 0
        done = 0
        ids = []
        logs = []
        start = time.time()
        for i, row in df_petri.iterrows():
            _logger.info('Started parsing:' + row.model_id)
            is_sound = self.soundness_check(row)
            if is_sound:
                df_petri.at[i, "sound"] = True
                _logger.info('model is sound. Generating traces')
                logs.append(self.log_creation_check(row))
                success += 1
            else:
                logs.append(None)
            ids.append(row.model_id)
            done += 1
        end = time.time()
        _logger.info("-------------------------------------------")
        _logger.info("PPID %s Completed in %s" % (os.getpid(), round(end - start, 2)))
        _logger.info(f"Number of converted (sound) models: {success} / {done}")
        signal.alarm(0)
        return pd.DataFrame({"model_id": ids, "log": logs})

    def generate_logs_lambda(self, df_petri):
        df_petri["sound"] = False
        df_petri["log"] = None
        df_results = []
        split_df = np.array_split(df_petri, 100)
        start = time.time()
        for i, df in enumerate(split_df):
            file_name = "no_" + str(i) + self.config.LOGS_SER_FILE
            if exists(self.config.PETRI_LOGS_DIR / file_name):
                df = pd.read_pickle(self.config.PETRI_LOGS_DIR / file_name)
            else:
                df['sound'] = df.apply(lambda x: self.soundness_check(x), axis=1)
                df['log'] = df.apply(lambda x: self.log_creation_check(x), axis=1)
                df.to_pickle(self.config.PETRI_LOGS_DIR / file_name)
            df_results.append(df)
        stop = time.time()
        completed_in = round(stop - start, 2)
        _logger.info("-------------------------------------------")
        _logger.info("PPID %s Completed in %s" % (os.getpid(), completed_in))
        return pd.concat(df_results)

    def generate_logs_from_petri_parallel(self, timeout, df_petri):
        df_petri["sound"] = False
        df_petri["log"] = None
        logical = False
        df_results = []
        num_procs = psutil.cpu_count(logical=logical)
        if len(sys.argv) > 1:
            num_procs = int(sys.argv[1])
        _logger.info(f"splitting into {num_procs} processes")
        split_df = np.array_split(df_petri, num_procs)
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
            results = [executor.submit(self._generate_logs_from_petri_parallel, timeout=timeout, df=df) for df in
                       split_df]
            for result in concurrent.futures.as_completed(results):
                try:
                    df_results.append(result.result())
                except Exception as ex:
                    _logger.warning(str(ex))
                    pass
        end = time.time()
        _logger.info("-------------------------------------------")
        _logger.info("PPID %s Completed in %s" % (os.getpid(), round(end - start, 2)))
        df_petri = pd.concat(df_results)
        return df_petri

    def _generate_logs_from_petri_parallel(self, df):
        pid = os.getpid()
        ppid = os.getppid()
        start = time.time()
        _logger.info("PPID %s->%s Started" % (ppid, pid))
        df['sound'] = df.apply(lambda x: self.soundness_check(x), axis=1)
        df['log'] = df.apply(lambda x: self.log_creation_check(x), axis=1)
        stop = time.time()
        completed_in = round(stop - start, 2)
        _logger.info("PPID %s Completed in %s" % (os.getpid(), completed_in))
        return df

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
            .set_index([self.config.MODEL_ID, self.config.ELEMENT_ID])
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
            df_bpmn = df_bpmn.loc[df_bpmn.index.isin(filter_df.index.values)]
            # print(filter_df.index.values)
            _logger.info("There are %d BPMNs after filtering out non-english ones.", len(df_bpmn))
        else:
            df_bpmn = df.query(f"namespace == '{self.config.BPMN2_NAMESPACE}'")
        return df_bpmn

    def _get_json_from_row(self, row_tuple):
        return json.loads(row_tuple.model_json)

    def _parse_df_row(self, row_tuple):
        # print(row_tuple.model_json)
        model_dict = json.loads(row_tuple.model_json)
        # print(model_dict)
        elements = self._get_elements_flat(model_dict)
        return (
            pd.DataFrame.from_records(elements)
            .assign(m_id=row_tuple.model_id)
            .assign(model_id=row_tuple.model_id)
        )

    def _get_elements_flat(self, model_dict) -> List[Dict[str, str]]:
        """
        Parses the recursive childShapes and produces a flat list of model elements with the most important attributes
        such as id, category, label, outgoing, and parent elements.
        """
        stack = deque([model_dict])
        elements_flat = []

        while len(stack) > 0:
            element = stack.pop()
            for c in element.get("childShapes", []):
                c["parent"] = element["resourceId"]
                stack.append(c)

            # don't append root as element
            if element["resourceId"] == model_dict["resourceId"]:
                continue

            # NOTE: it's possible to add other attributes here, such as the bounds of an element
            record = {
                "element_id": element["resourceId"],
                "category": element["stencil"].get("id") if "stencil" in element else None,
                "label": element["properties"].get("name"),
                "glossary": json.dumps(element["glossaryLinks"]) if "glossaryLinks" in element else "{}",
            }
            if self.parse_parent:
                record["parent"] = element.get("parent")
            if self.parse_outgoing:
                record["outgoing"] = [v for d in element.get("outgoing", []) for v in d.values()]

            elements_flat.append(record)

        return elements_flat

    def alarm_handler(self, signum, frame):
        raise Exception("timeout")

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

    def log_creation_check(self, row):
        # _logger.info("Log creation. " + row.model_id)
        log_no_loops = None
        if row.sound:
            start = time.time()
            signal.signal(signal.SIGALRM, self.alarm_handler)
            signal.alarm(self.config.TIMEOUT)
            try:
                net, im, fm = row.pn
                log = pm4py.play_out(net, im, fm, variant=Variants.EXTENSIVE)
                log_no_loops = create_log_without_loops(log)
            except Exception as ex:
                _logger.warning("Time out during log creation.")
                _logger.warning(ex)
            finally:
                signal.alarm(0)
                stop = time.time()
                completed_in = round(stop - start, 2)
                if completed_in > 1.5 * self.config.TIMEOUT:
                    _logger.error("Timeout not working!!")
        return log_no_loops
