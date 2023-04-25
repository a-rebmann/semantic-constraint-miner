import gc
import json
import logging
import os
import signal
import time
from os.path import exists

import pm4py
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.simulation.playout.petri_net.algorithm import Variants
from pm4py.objects.log.obj import EventLog
import pandas as pd
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

from semconstmining.parsing.conversion.bpmnjsonanalyzer import fromJSON
from semconstmining.parsing.conversion.jsontopetrinetconverter import JsonToPetriNetConverter

_logger = logging.getLogger(__name__)


def create_log_without_loops(log):
    log_no_loops = EventLog()
    for trace in log:
        trace_labels = [x["concept:name"] for x in trace]
        if 0 < len(trace_labels) == len(set(trace_labels)):
            log_no_loops.append(trace)
    return log_no_loops


def _get_json_from_row(row_tuple):
    return json.loads(row_tuple.model_json)


class Model2LogConverter:

    def __init__(self, config):
        self.config = config
        self.converter = JsonToPetriNetConverter()
        self.done = 0
        self.loop_counter = 0

    def alarm_handler(self, signum, frame):
        raise Exception("timeout")

    def generate_logs_lambda(self, df_petri, model_elements):
        elms = model_elements.set_index(self.config.ELEMENT_ID_BACKUP)
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
                df[self.config.SOUND] = df.apply(lambda x: self.soundness_check(x), axis=1)
                df[self.config.LOG] = df.apply(lambda x: self.log_creation_check(x, elms), axis=1)
                df.to_pickle(self.config.PETRI_LOGS_DIR / file_name)
            df_results.append(df)
        stop = time.time()
        completed_in = round(stop - start, 2)
        _logger.info("with loops " + str(self.loop_counter))
        _logger.info("-------------------------------------------")
        _logger.info("PPID %s Completed in %s" % (os.getpid(), completed_in))
        return pd.concat(df_results)

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
            except Exception as ex:
                _logger.warning("Error during soundness checking.")
                res = False
            finally:
                stop = time.time()
                completed_in = round(stop - start, 2)
                if completed_in > 2 * self.config.TIMEOUT:
                    _logger.error("Timeout not working!! " + row.model_id)
            self.done += 1
            if self.done % 5000 == 0:
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
        names = []
        for row in df_bpmn.reset_index().itertuples():
            # add the id and name of the model to the new frame for future reference
            ids.append(row.model_id)
            names.append(row.name)
            json_str = _get_json_from_row(row)
            try:
                f, l, _ = fromJSON(json_str)
                follows_ = {}
                labels_ = {}
                for e in l:
                    labels_[row.model_id + str(e)] = l[e]
                for e in f:
                    follows_[row.model_id + str(e)] = [row.model_id + str(e) for e in f[e]]
                net, initial_marking, final_marking = self.converter.convert_from_parsed(follows_, labels_)
                follows.append(follows_)
                labels.append(labels_)
                pns.append((net, initial_marking, final_marking))
                success += 1
            except KeyError:
                _logger.debug("Error during conversion from bpmn to Petri net.")
                pns.append(None)
                follows.append(None)
                labels.append(None)
                failed += 1
            except Exception as ex:
                _logger.debug("Error during conversion from bpmn to Petri net." + str(ex))
                pns.append(None)
                follows.append(None)
                labels.append(None)
                failed += 1
        _logger.info(str(success + failed) + " jsons done. Success: " + str(success) + " failed: " + str(failed))
        return pd.DataFrame(
            {"model_id": ids, "pn": pns, "follows": follows, "labels": labels, "name": names})

    def create_variant_log(self, log):
        variant_log = EventLog()
        seen = set()
        already_counted_loop = False
        for trace in log:
            trace_labels = tuple([x["concept:name"] for x in trace])
            if trace_labels not in seen:
                variant_log.append(trace)
                seen.add(trace_labels)
                if 0 < len(trace_labels) == len(set(trace_labels)) and not already_counted_loop:
                    self.loop_counter += 1
                    already_counted_loop = True
        return variant_log

    def log_creation_check(self, row, model_elements):
        # _logger.info("Log creation. " + row.model_id)
        played_out_log = None
        if row.sound:
            start = time.time()
            signal.signal(signal.SIGALRM, self.alarm_handler)
            signal.alarm(self.config.TIMEOUT)
            try:
                net, im, fm = row.pn
                log = pm4py.play_out(net, im, fm, variant=Variants.EXTENSIVE)
                variant_log = self.create_variant_log(log)
                if not self.config.LOOPS:
                    played_out_log = create_log_without_loops(variant_log)
                else:
                    played_out_log = variant_log
            except Exception as ex:
                _logger.warning(str(ex))
                _logger.warning(ex)
            finally:
                signal.alarm(0)
                stop = time.time()
                completed_in = round(stop - start, 2)
                if completed_in > 1.5 * self.config.TIMEOUT:
                    _logger.error("Timeout not working!!")
        if played_out_log is not None:
            played_out_log = self.replace_attributes(played_out_log, model_elements)
        return played_out_log

    def replace_attributes(self, played_out_log, model_elements):
        for trace in played_out_log:
            for event in trace:
                e_id = event[self.config.XES_NAME]
                event[self.config.DATA_OBJECT] = model_elements.loc[e_id, self.config.DATA_OBJECT]
                event[self.config.ELEMENT_ID] = e_id
                event[self.config.XES_NAME] = model_elements.loc[e_id, self.config.CLEANED_LABEL]
                event[self.config.DICTIONARY] = model_elements.loc[e_id, self.config.DICTIONARY]
        return played_out_log
