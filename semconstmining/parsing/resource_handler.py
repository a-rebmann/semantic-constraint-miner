import logging
import os
import warnings
from os.path import exists
import pandas as pd
from tqdm import tqdm
import json

from semconstmining.parsing.actioncalssification import ActionClassifier
from semconstmining.parsing.label_parser import nlp_helper
from semconstmining.mining.model.parsed_label import ParsedLabel, get_dummy
from semconstmining.parsing import parser
from semconstmining.parsing import detector
from semconstmining.parsing.components import Components
from semconstmining.parsing.model_to_log import Model2LogConverter
from semconstmining.util.io import read_pickle, write_pickle

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

_logger = logging.getLogger(__name__)


class ResourceHandler:
    """
    This class is responsible for loading and saving the resources needed for the parsing and mining constraints.
    It is also responsible for the conversion of the models to logs.
    """

    def __init__(self, config, nlp_helper):
        self.config = config
        self.data_parser = parser.BpmnModelParser(config)
        self.nlp_helper = nlp_helper
        self.model_to_log_converter = Model2LogConverter(config)
        self.elements_ser_file = config.DATA_INTERIM / (self.config.MODEL_COLLECTION + "_" + config.ELEMENTS_SER_FILE)
        self.models_ser_file = config.DATA_INTERIM / (self.config.MODEL_COLLECTION + "_" + config.MODELS_SER_FILE)
        self.languages_ser_file = config.DATA_INTERIM / (self.config.MODEL_COLLECTION + "_" + config.LANG_SER_FILE)
        self.logs_ser_file = config.DATA_INTERIM / (self.config.MODEL_COLLECTION + "_" + config.LOGS_SER_FILE)
        self.tagged_ser_file = config.DATA_INTERIM / (self.config.MODEL_COLLECTION + "_" + config.TAGGED_SER_FILE)
        self.comp_ser_file = config.DATA_INTERIM / (self.config.MODEL_COLLECTION + "_" + config.COMPONENTS_SER_FILE)
        self.dictionary_ser_file = config.DATA_DATASET_DICT
        self.bpmn_model_elements = None
        self.bpmn_models = None
        self.model_languages = None
        self.bpmn_logs = None
        self.bpmn_task_labels = None
        self.dictionary = None
        self.components = Components(config)
        self.filter_options = None

    def get_filter_options(self):
        if self.filter_options is None:
            self.filter_options = {
                self.config.OPERATOR_TYPE: [self.config.UNARY, self.config.BINARY],
                self.config.LEVEL: [self.config.OBJECT, self.config.MULTI_OBJECT, self.config.RESOURCE,
                                    self.config.ACTIVITY],
                self.config.DICTIONARY: self.get_names_of_dictionary_entries(),
                self.config.DATA_OBJECT: self.get_names_of_data_objects(),
                self.config.ACTION_CATEGORY: self.config.ACTION_CATEGORIES,
                self.config.ACTION: list(self.components.all_actions),
                self.config.OBJECT: list(self.components.all_objects),
                self.config.NAME: list(self.bpmn_models[self.config.NAME].unique())
            }
        return self.filter_options

    def get_parsed_task(self, t1):
        """
        This method returns the parsed label for a given label.
        """
        if t1 in self.components.parsed_tasks:
            return self.components.parsed_tasks[t1]
        t1_parse = self.bpmn_model_elements[
            (self.bpmn_model_elements[self.config.CLEANED_LABEL] == t1)
            & (~self.bpmn_model_elements[self.config.SPLIT_LABEL].isna())].reset_index()
        if len(t1_parse) == 0:
            return get_dummy(self.config, t1, self.config.EN)
        label = t1_parse[self.config.CLEANED_LABEL].values[0]
        split = t1_parse[self.config.SPLIT_LABEL].values[0]
        tags = t1_parse[self.config.TAGS].values[0]
        dicts = []
        d_objs = []
        if t1_parse[self.config.DICTIONARY].values[0] != []:
            dicts = t1_parse[self.config.DICTIONARY].values[0]
            dicts = [entry for entry in dicts if entry not in self.config.TERMS_FOR_MISSING]
        if t1_parse[self.config.DATA_OBJECT].values[0] != []:
            d_objs = t1_parse[self.config.DATA_OBJECT].values[0]
            d_objs = [entry for entry in d_objs if entry not in self.config.TERMS_FOR_MISSING]
        if "lang" not in t1_parse.columns:
            lang = self.config.EN
        else:
            lang = t1_parse["lang"].values[0]
        parsed = ParsedLabel(self.config, label, split, tags, self.nlp_helper.find_objects(split, tags),
                             self.nlp_helper.find_actions(split, tags, lemmatize=True), lang,
                             dictionary_entries=dicts, data_objects=d_objs)
        self.components.parsed_tasks[t1] = parsed
        return parsed

    def get_dictionary_entry(self, entry):
        """
        This method returns the name of a dictionary entry.
        """
        if self.dictionary is None:
            self.load_dictionary_if_exists()
        if self.dictionary is None:
            _logger.warning("No dictionary found. Please make sure the dictionary data is available.")
        return self.dictionary.loc[entry]['name']

    def load_bpmn_model_elements(self):
        """
        This method loads the model elements from the pickle file or parses them if the file does not exist.
        """
        if exists(self.config.DATA_INTERIM / self.elements_ser_file):
            _logger.info("Loading elements from " + str(self.config.DATA_INTERIM / self.elements_ser_file) + ".")
            self.bpmn_model_elements = pd.read_pickle(self.config.DATA_INTERIM / self.elements_ser_file)
        else:
            self.bpmn_model_elements = self.data_parser.parse_model_elements()
            self.bpmn_model_elements[self.config.CLEANED_LABEL] = self.bpmn_model_elements[self.config.LABEL].apply(
                lambda x: nlp_helper.sanitize_label(str(x or '')))
            self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.elements_ser_file)
        self.referenced_data_objects = set(self.bpmn_model_elements[self.config.DATA_OBJECT].explode().unique())
        _logger.info("There are " + str(len(self.referenced_data_objects)) + " referenced data objects.")
        _logger.info("These have " + str(len(self.get_names_of_data_objects())) + " unique names.")
        _logger.info("There are " + str(len(self.bpmn_model_elements)) + " elements in total.")

    def get_names_of_data_objects(self, ids=None):
        """
        This method returns the names of the data objects that are actually referenced from other model elements.
        """
        if ids is None:
            return list(self.bpmn_model_elements[self.bpmn_model_elements[
                self.config.ELEMENT_ID_BACKUP].isin(self.referenced_data_objects)][self.config.LABEL].unique())
        else:
            return list(self.bpmn_model_elements[self.bpmn_model_elements[
                                                     self.config.ELEMENT_ID_BACKUP].isin(self.referenced_data_objects)
                                                 & self.bpmn_model_elements[self.config.ELEMENT_ID_BACKUP].isin(ids)]
                        [self.config.LABEL].unique())

    def get_names_of_dictionary_entries(self, ids=None):
        """
        This method returns the names of the dictionary entries that are actually referenced from model elements.
        """
        if self.dictionary is None:
            return []
        if ids is None:
            return list(self.dictionary[self.dictionary[self.config.IS_REFERENCED]][self.config.NAME].unique())
        else:
            return list(self.dictionary[self.dictionary[self.config.IS_REFERENCED] & self.dictionary.index.isin(ids)]
                        [self.config.NAME].unique())

    def load_bpmn_models(self):
        """
        This method loads the models from the pickle file or parses them if the file does not exist.
        """
        if exists(self.config.DATA_INTERIM / self.models_ser_file):
            _logger.info("Loading models from " + str(self.config.DATA_INTERIM / self.models_ser_file) + ".")
            self.bpmn_models = pd.read_pickle(self.config.DATA_INTERIM / self.models_ser_file)
        else:
            self.bpmn_models = self.data_parser.parse_models(filter_df=self.bpmn_model_elements)
            self.bpmn_models[self.config.NAME] = self.bpmn_models[self.config.NAME].astype(str)
            _logger.info("Remove example models from " + str(len(self.bpmn_models)))
            pattern = '|'.join(self.config.EXAMPLE_MODEL_NAMES)
            df_no_example = self.bpmn_models[~self.bpmn_models[self.config.NAME].str.contains(pattern)]
            df_example = self.bpmn_models[self.bpmn_models[self.config.NAME].str.contains(pattern)].drop_duplicates(
                subset=[self.config.NAME])
            self.bpmn_models = pd.concat([df_example, df_no_example])
            _logger.info(str(len(self.bpmn_models)) + " models remaining")
            self.bpmn_models.to_pickle(self.config.DATA_INTERIM / self.models_ser_file)

    def determine_model_languages(self):
        """
        This method determines the natural language of the models.
        """
        if exists(self.config.DATA_INTERIM / self.languages_ser_file):
            _logger.info("Loading detected languages.")
            self.model_languages = pd.read_pickle(self.config.DATA_INTERIM / self.languages_ser_file)
            if self.config.DETECTED_NAT_LANG not in self.bpmn_model_elements.columns:
                self.bpmn_model_elements = pd.merge(self.bpmn_model_elements, self.model_languages, how="left",
                                                    on=self.config.MODEL_ID)
                self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.languages_ser_file)
        else:
            _logger.info("Detect languages.")
            ld = detector.ModelLanguageDetector(self.config, 0.8)
            self.model_languages = ld.get_detected_natural_language_from_bpmn_model(self.bpmn_model_elements)
            self.model_languages.to_pickle(self.config.DATA_INTERIM / self.languages_ser_file)
            if self.config.DETECTED_NAT_LANG not in self.bpmn_model_elements.columns:
                self.bpmn_model_elements = pd.merge(self.bpmn_model_elements, self.model_languages, how="left",
                                                    on=self.config.MODEL_ID)
                self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.elements_ser_file)

    def get_logs_for_sound_models(self):
        """
        This method returns the logs for the sound models.
        """
        # Parse and convert the JSON-BPMNs to Petri nets
        if exists(self.config.DATA_INTERIM / self.logs_ser_file):
            self.bpmn_logs = pd.read_pickle(self.config.DATA_INTERIM / self.logs_ser_file)
        else:
            df_petri = self.model_to_log_converter.convert_models_to_pn_df(self.bpmn_models)
            self.bpmn_logs = self.model_to_log_converter.generate_logs_lambda(df_petri, self.bpmn_model_elements)
            self.bpmn_logs.to_pickle(self.config.DATA_INTERIM / self.logs_ser_file)
            for (dir_path, dir_names, filenames) in os.walk(self.config.PETRI_LOGS_DIR):
                for filename in filenames:
                    os.remove(dir_path + "/" + filename)
        _logger.info("Number of available logs: " + str(len(self.bpmn_logs[self.bpmn_logs[self.config.LOG].notna()])))
        _logger.info("Number of available models: " + str(len(self.bpmn_logs)))
        return

    def tag_task_labels(self):
        """
        This method tags the labels of the tasks. It extracts actions and objects from the labels.
        """
        if exists(self.config.DATA_INTERIM / self.tagged_ser_file):
            _logger.info("Loading tagged labels.")
            self.bpmn_task_labels = pd.read_pickle(self.config.DATA_INTERIM / self.tagged_ser_file)
            if self.config.SPLIT_LABEL not in self.bpmn_model_elements.columns:
                self.bpmn_model_elements = pd.merge(self.bpmn_model_elements, self.bpmn_task_labels, how='left',
                                                    on=self.config.CLEANED_LABEL)
                self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.elements_ser_file)
        else:
            _logger.info("Start tagging labels.")
            all_labs = list(
                self.bpmn_model_elements[(self.bpmn_model_elements[self.config.ELEMENT_CATEGORY] == "Task")][
                    self.config.CLEANED_LABEL].unique())
            _logger.info(str(len(all_labs)) + " labels cleaned. " + "Start parsing.")
            all_labs_split = [nlp_helper.split_label(lab) for lab in tqdm(all_labs)]
            tagged = self.nlp_helper.parse_labels(all_labs)
            self.bpmn_task_labels = pd.DataFrame(
                {self.config.CLEANED_LABEL: all_labs, self.config.SPLIT_LABEL: all_labs_split, "tags": tagged,
                 "lang": ["english" for _ in range(len(all_labs))]})
            self.bpmn_task_labels.to_pickle(self.config.DATA_INTERIM / self.tagged_ser_file)
            if self.config.SPLIT_LABEL not in self.bpmn_model_elements.columns:
                self.bpmn_model_elements = pd.merge(self.bpmn_model_elements, self.bpmn_task_labels, how='left',
                                                    on=self.config.CLEANED_LABEL)
                self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.elements_ser_file)
        _logger.info("We have " + str(len(self.bpmn_task_labels)) + " labels.")

    def filter_only_english(self):
        """
        This method filters the models for which the natural language is english.
        """
        _logger.info("Filtering for english labels.")
        self.bpmn_model_elements = self.bpmn_model_elements[self.bpmn_model_elements[self.config.DETECTED_NAT_LANG]
                                                            == self.config.EN]

    def load_dictionary_if_exists(self):
        """
        This method loads the dictionary if it exists.
        """
        if self.dictionary_ser_file is None:
            return
        paths = sorted(self.dictionary_ser_file.glob("*.csv"))
        if len(paths) > 0:
            _logger.info("Loading dictionary from " + str(paths[-1]))
            self.dictionary = parser.parse_dict_csv_raw(paths[-1])
            if self.config.DICTIONARY not in self.bpmn_model_elements.columns:
                self.bpmn_model_elements[self.config.DICTIONARY] = self.bpmn_model_elements[self.config.GLOSSARY].apply(
                    lambda x: self.get_entries_from_dict(x))
                self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.elements_ser_file)
            for entries in self.bpmn_model_elements[self.config.DICTIONARY]:
                self.components.referenced_dict_entries.update(entries)
            self.dictionary[self.config.IS_REFERENCED] = \
                self.dictionary.index.isin(self.components.referenced_dict_entries)
        if self.config.DICTIONARY not in self.bpmn_model_elements.columns:
            self.bpmn_model_elements[self.config.DICTIONARY] = None

    def load_or_create_components(self):
        """
        This method loads important components, i.e., actions, objects, referenced dictionary entries,
        data objects, and parsed labels, if they have been processed, otherwise creates and saves them.
        """
        if exists(self.config.DATA_INTERIM / self.comp_ser_file):
            self.components = read_pickle(self.config.DATA_INTERIM / self.comp_ser_file)
        else:
            self.handle_all_actions_and_objects()
            write_pickle(self.components, self.config.DATA_INTERIM / self.comp_ser_file)

    def get_entries_from_dict(self, glossary_entries):
        """
        This method returns the entries from the dictionary.
        """
        if glossary_entries == '{}' or glossary_entries == '{"name": [""]}':
            return []
        entries = [entry.replace("/glossary/", "") for entry in json.loads(glossary_entries)[self.config.NAME]]
        return entries

    def handle_all_actions_and_objects(self):
        for index, row in self.bpmn_model_elements[
            (~self.bpmn_model_elements[self.config.SPLIT_LABEL].isna())].iterrows():
            if row[self.config.CLEANED_LABEL] in self.components.parsed_tasks:
                parsed = self.components.parsed_tasks[row[self.config.CLEANED_LABEL]]
            else:
                parsed = ParsedLabel(self.config, row[self.config.CLEANED_LABEL],
                                     row[self.config.SPLIT_LABEL], row[self.config.TAGS],
                                     self.nlp_helper.find_objects(row[self.config.SPLIT_LABEL],
                                                                              row[self.config.TAGS]),
                                     self.nlp_helper.find_actions(row[self.config.SPLIT_LABEL],
                                                                              row[self.config.TAGS],
                                                                              lemmatize=True),
                                     row[self.config.LANG],
                                     [entry for entry in row[self.config.DICTIONARY]
                                      if entry not in self.config.TERMS_FOR_MISSING]
                                     if row[self.config.DICTIONARY] is not None else [],
                                     [entry for entry in row[self.config.DATA_OBJECT]
                                      if entry not in self.config.TERMS_FOR_MISSING]
                                     if row[self.config.DATA_OBJECT] is not None else [])
                self.components.parsed_tasks[row[self.config.CLEANED_LABEL]] = parsed
            self.components.add_action(row[self.config.MODEL_ID], parsed.main_action)
            self.components.add_object(row[self.config.MODEL_ID], parsed.main_object)
        self.categorize_actions()
        _logger.info("Handled main components")

    def categorize_actions(self):
        action_classifier = ActionClassifier(self.config, self.components.all_actions,
                                             self.nlp_helper.glove_embeddings)
        self.components.action_to_category = action_classifier.classify_actions()
        for label in self.components.parsed_tasks:
            if label not in self.components.action_to_category:
                self.components.action_to_category[label] = \
                    self.components.action_to_category[self.components.parsed_tasks[label].main_action]
