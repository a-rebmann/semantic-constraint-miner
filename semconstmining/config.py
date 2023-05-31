import json
from pathlib import Path

from semconstmining.mining.model.constraint import Observation
from semconstmining.declare.enums import Template

example_models = {
  "example_processes": [
    {
      "batch_name": "1st batch",
      "content": [
        "Lieferung-zu-Bezahlung",
        "Bestellung-zu-Lieferung",
        "BANF-zu-Bestellung",
        "Wertschöpfungskette: Beschaffung",
        "Purchase Order-to-Delivery",
        "Delivery-to-Payment",
        "Purchase Requisition-to-Purchase Order",
        "Value chain: Procurement"
      ],
      "comment": ""
    },

    {
      "batch_name": "2nd batch",
      "content": [
        "Lieferung-zu-Bezahlung",
        "Bestellung-zu-Lieferung",
        "BANF-zu-Bestellung",
        "Wertschöpfungskette: Beschaffung",
        "Purchase Order-to-Delivery",
        "Delivery-to-Payment",
        "Purchase Requisition-to-Purchase Order",
        "Value chain: Procurement"
      ],
      "comment": "2nd batch, should be same names as 1st"
    },
    {
      "batch_name": "3rd batch A",
      "content": [
        "Ebene 1 - Prozesslandkarte ACME AG",
        "Ebene 2 - Prozessbereich: Auftragsdurchführung",
        "Ebene 2 - Prozessbereich: Produktentwicklung",
        "Teile beschaffen",
        "Wareneingang",
        "Menge und Qualität überprüfen",
        "Ebene  2 - Prozessbereich: Personalwesen",
        "Arbeitsmittel beschaffen",
        "Bewerbungseingang",
        "Mitarbeiter Onboarding",
        "Bewerber prüfen",
        "Level 1 - Value Chain ACME AG",
        "Level 2 - Process Area: Product Development",
        "Level 2 - Process Area: Order Processing",
        "Procure parts",
        "Check quantity and quality",
        "Receipt of Goods",
        "Level  2 - Process Area: Human Resources",
        "Receipt of Application",
        "Verify applicant",
        "Employee Onboarding",
        "Procurement of Work Equipment",
        "Niveau 1 \u2013 Chaine de valeur d'ACME AG",
        "Niveau 2 - Processus des Ressources Humaines",
        "Donner l'équipement de travail",
        "Vérifier le candidat",
        "Installation d'un employé",
        "Réception d'une candidature",
        "Niveau 2 - Processus de développement produit",
        "Niveau 2 - Processus de gestion des commandes",
        "Contrôler la quantité et la qualité",
        "Commande de pièces",
        "Réception de biens"
      ],
      "comment": "3rd batch A, new!"
    },
    {
      "batch_name": "3rd batch B",
      "content": [
        "Ebene 1 - Prozesslandkarte ACME AG",
        "Ebene 2 - Prozessbereich: Auftragsdurchführung",
        "Ebene 2 - Prozessbereich: Produktentwicklung",
        "Teile beschaffen",
        "Wareneingang",
        "Menge und Qualität überprüfen",
        "Ebene  2 - Prozessbereich: Personalwesen",
        "Arbeitsmittel beschaffen",
        "Bewerbungseingang",
        "Mitarbeiter Onboarding",
        "Bewerber prüfen",
        "Level 1 - Value Chain ACME AG",
        "Level 2 - Process Area: Product Development",
        "Level 2 - Process Area: Order Processing",
        "Procure parts",
        "Check quantity and quality",
        "Receipt of Goods",
        "Level  2 - Process Area: Human Resources",
        "Receipt of Application",
        "Verify applicant",
        "Employee Onboarding",
        "Procurement of Work Equipment",
        "Niveau 1 \u2013 Chaine de valeur d'ACME AG",
        "Niveau 2 - Processus des Ressources Humaines",
        "Donner l'équipement de travail",
        "Vérifier le candidat",
        "Installation d'un employé",
        "Réception d'une candidature",
        "Niveau 2 - Processus de développement produit",
        "Niveau 2 - Processus de gestion des commandes",
        "Contrôler la quantité et la qualité",
        "Commande de pièces",
        "Réception de biens"
      ],
      "comment": "3rd batch B, localized, should be same as 3rd batch A"
    }
  ]
}


class Config:

    def __init__(self, project_root: Path, model_collection="sap_sam_2022"):
        self.PROJECT_ROOT: Path = project_root
        self.MODEL_COLLECTION = model_collection
        self.DATA_ROOT = self.PROJECT_ROOT / "data"
        self.DATA_RAW = self.DATA_ROOT / "raw"
        self.DATA_LOGS = self.DATA_ROOT / "logs"
        self.DATA_RESOURCES = self.DATA_ROOT / "resources"
        self.DATA_DATASET = self.DATA_RAW / self.MODEL_COLLECTION / "models"
        self.DATA_DATASET_DICT = self.DATA_RAW / self.MODEL_COLLECTION / "dict"
        self.DATA_INTERIM = self.DATA_ROOT / "interim"
        self.DATA_EVAL = self.DATA_ROOT / "eval"
        self.SRC_ROOT = self.PROJECT_ROOT / "semantic-constraint-miner"

        self.PETRI_LOGS_DIR = self.DATA_INTERIM / "bpmn_logs"

        self.MODEL_PATH = self.DATA_ROOT / "bert"

        self.ELEMENTS_SER_FILE = "bpmn_elements.pkl"
        self.LANG_SER_FILE = "bpmn_languages.pkl"
        self.MODELS_SER_FILE = "bpmn_models.pkl"
        self.LOGS_SER_FILE = "bpmn_logs.pkl"
        self.TAGGED_SER_FILE = "bpmn_tagged.pkl"
        self.CF_OBSERVATIONS_SER_FILE = "bpmn_cf_observations.pkl"
        self.MP_OBSERVATIONS_SER_FILE = "bpmn_mp_observations.pkl"
        self.CONSTRAINT_KB_SER_FILE = "constraint_kb.pkl"
        self.PREPROCESSED_CONSTRAINTS = "preprocessed_constraints.pkl"
        self.LOG_INFO = "log_info.pkl"
        self.COMPONENTS_SER_FILE = "filter_options.pkl"
        # Known models
        self.MODEL_META_SER_FILE = "model_meta.pkl"
        # Known constraints
        self.DECLARE_CONST = "bpmn_declare_const.pkl"
        # Known similarities
        self.SIM_MAP = "sim_map.pkl"
        # Known embeddings
        self.EMB_MAP = "emb_map.pkl"

        self.NON_TASKS = (
            "SequenceFlow", "MessageFlow", "DataObject", "Pool", "Lane", "TextAnnotation", "Association_Undirected",
            "Association_Bidirectional", "Association_Unidirectional", "Group", "CollapsedPool", "ITSystem",
            "DataStore")

        self.BPMN2_NAMESPACE = "http://b3mn.org/stencilset/bpmn2.0#"
        self.VERB_OCEAN_FILE = "verbocean.txt"

        self.EMAIL_DUMMY = "jane.doe@dummy.com"
        self.NAME_DUMMY = "Jane Doe"
        self.NUMBER_DUMMY = "12345678"

        self.EXAMPLE_MODEL_NAMES = [name for batch in example_models["example_processes"] for name in batch["content"]]

        self.COLORS_SIGNAVIO_HSL = [
            "hsl(309, 88%, 33%)", "hsl(313, 81%, 35%)", "hsl(318, 75%, 36%)", "hsl(322, 70%, 39%)",
            "hsl(326, 65%, 40%)",
            "hsl(331, 60%, 43%)", "hsl(336, 56%, 45%)", "hsl(341, 53%, 46%)", "hsl(352, 47%, 50%)",
            "hsl(358, 48%, 53%)",
            "hsl(4, 51%, 53%)", "hsl(9, 54%, 53%)", "hsl(14, 57%, 52%)", "hsl(18, 60%, 52%)", "hsl(21, 64%, 52%)",
            "hsl(25, 67%, 51%)", "hsl(27, 70%, 51%)", "hsl(30, 73%, 51%)", "hsl(32, 76%, 50%)", "hsl(34, 79%, 50%)",
            "hsl(36, 83%, 50%)", "hsl(38, 87%, 49%)", "hsl(40, 91%, 49%)", "hsl(41, 96%, 48%)", "hsl(43, 100%, 48%)"
        ]

        self.COLORS_SIGNAVIO_ = [
            "#9c0a85", "#a01180", "#a3177a", "#a71e75", "#aa2470", "#ae2b6a", "#b23265", "#b53860", "#b93f5a",
            "#bc4555",
            "#c04c50", "#c4534a", "#c75945", "#cb6040", "#ce663b", "#d26d35", "#d67430", "#d97a2b", "#dd8125",
            "#e08720",
            "#e48e1b", "#e89515", "#eb9b10", "#efa20b", "#f2a805", "#f6af00"
        ]

        # Strings that represent missing values
        self.TERMS_FOR_MISSING = [
            'MISSING', 'UNDEFINED', 'undefined', 'missing', 'nan', 'empty', 'empties', 'unknown',
            'other', 'others', 'na', 'nil', 'null', '', "", ' ', '<unknown>', "0;n/a", "NIL", 'undefined',
            'missing', 'none', 'nan', 'empty', 'empties', 'unknown', 'other', 'others', 'na', 'nil', 'null'
        ]

        # Timeout for soundness check and Petri net play-out
        self.TIMEOUT = 10

        # XES ATTRIBUTE NAMES
        self.XES_NAME = "concept:name"
        self.XES_TIME = "time:timestamp"
        self.XES_LIFECYCLE = "lifecycle:transition"
        self.XES_RESOURCE = "org:resource"
        self.XES_GROUP = "org:group"
        self.XES_ROLE = "org:role"
        self.XES_CASE = "case:" + self.XES_NAME
        self.XES_INST = "concept:instance"
        self.VIOLATION_TYPE = "violation_type"

        # PROPRIETARY ATTRIBUTE NAMES
        self.DETECTED_NAT_LANG = "detected_natural_language"
        self.CLEANED_LABEL = "cleaned label"
        self.SPLIT_LABEL = "split label"
        self.ELEMENT_CATEGORY = "category"
        self.CONSTRAINT_STR = "constraint_string"
        self.LEFT_OPERAND = "left_op"
        self.RIGHT_OPERAND = "right_op"
        self.RECORD_ID = "obs_id"
        self.FITTED_RECORD_ID = "fitted_obs_id"
        self.LEVEL = "Level"
        self.OPERATOR_TYPE = "op_type"
        self.MODEL_NAME = "model_name"
        self.MODEL_ID = "model_id"
        self.ELEMENT_ID = "element_id"
        self.ELEMENT_ID_BACKUP = "e_id"
        self.LABEL = "label"
        self.GLOSSARY = "glossary"
        self.DICTIONARY = "dictionary"
        self.LABEL_LIST = "label_list"
        self.NAME = "name"
        self.SUPPORT = "support"
        self.REDUNDANT = "redundant"
        self.NAT_LANG_OBJ = "nat_lang_obj"
        self.NAT_LANG_TEMPLATE = "nat_lang_template"
        self.DATA_OBJECT = "data_object"
        self.IS_REFERENCED = "is_referenced"
        self.SOUND = "sound"
        self.LOG = "log"
        self.NOISY_LOG = "noisy_log"
        self.ACTION = "action"
        self.ACTION_CATEGORY = "action_category"
        self.TAGS = "tags"
        self.ACTIVATION = "activation"
        self.TEMPLATE = "template"
        self.LTL = "ltl"

        # DIFFERENT TYPES OF GENERALITY SCORES
        self.NAME_BASED_GENERALITY = "name_based_generality"
        self.LABEL_BASED_GENERALITY = "object_based_generality"
        self.OBJECT_BASED_GENERALITY = "label_based_generality"

        # DIFFERENT TYPES OF RELEVANCE SCORES
        self.NAME_BASED_RELEVANCE = "name_based_relevance"
        self.LABEL_BASED_RELEVANCE = "label_based_relevance"
        self.OBJECT_BASED_RELEVANCE = "object_based_relevance"
        self.ACTION_BASED_RELEVANCE = "action_based_relevance"
        self.RESOURCE_BASED_RELEVANCE = "resource_based_relevance"

        self.INDIVIDUAL_RELEVANCE_SCORES = "individual_relevance_scores"
        self.SEMANTIC_BASED_RELEVANCE = "semantic_based_relevance"
        self.CONSTRAINT_BASED_RELEVANCE = "constraint_based_relevance"

        # OVERALL RELEVANCE SCORE
        self.RELEVANCE_SCORE = "relevance_score"

        # ARITY OF CONSTRAINTS
        self.UNARY = "Unary"
        self.BINARY = "Binary"

        # TYPES OF CONSTRAINTS
        self.OBJECT = "Object"
        self.MULTI_OBJECT = "Multi-object"
        self.RESOURCE = "Resource"
        self.DECISION = "Decision"
        self.LINGUISTIC = "Linguistic"
        self.ACTIVITY = "Activity"

        # LANGUAGES
        self.EN = "english"
        self.DE = "german"
        self.FR = "french"
        self.ES = "spanish"
        self.NL = "dutch"

        self.LANGUAGE = self.EN
        self.LANG = "lang"

        # All binary constraint templates
        self.BINARY_TEMPLATES = [bin_temp.templ_str for bin_temp in Template.get_binary_templates()]
        # All unary constraint templates
        self.UNARY_TEMPLATES = [un_temp.templ_str for un_temp in Template.get_unary_templates()]
        # All cardinality constraint templates
        self.CARDINALITY_TEMPLATES = [un_temp.templ_str for un_temp in Template.get_cardinality_templates()]
        # All negative constraint templates
        self.NEGATIVE_TEMPLATES = [un_temp.templ_str for un_temp in Template.get_negative_templates()]

        # Constraints that are really useless
        self.IRRELEVANT_CONSTRAINTS = {
            Template.EXCLUSIVE_CHOICE.templ_str: ["yes", "no"]
        }

        self.CONSTRAINT_TYPES_TO_IGNORE = [Template.CHAIN_RESPONSE.templ_str,
                                           Template.CHAIN_PRECEDENCE.templ_str, Template.CHAIN_SUCCESSION.templ_str,
                                           Template.CHOICE.templ_str, Template.INIT.templ_str, Template.END.templ_str] \
                                          + self.NEGATIVE_TEMPLATES

        self.CONSTRAINT_TEMPLATES_TO_IGNORE_PER_TYPE = {
            self.ACTIVITY: [],
            self.OBJECT: [],
            self.MULTI_OBJECT: [Template.EXACTLY.templ_str, Template.EXISTENCE.templ_str, Template.ABSENCE.templ_str,
                                Template.INIT.templ_str, Template.END.templ_str, Template.EXCLUSIVE_CHOICE.templ_str,
                                Template.CHOICE.templ_str, Template.CHAIN_PRECEDENCE.templ_str,
                                Template.CHAIN_RESPONSE.templ_str, Template.CHAIN_SUCCESSION.templ_str],
            self.RESOURCE: [],
        }

        # REDUNDANCY RESOLUTION STRATEGIES
        # Privileges the hierarchy (eager policy): for example, if the constraint set contains AlternatePrecedence(A,
        # B) and Precedence(A, B), the latter is filtered out.
        self.EAGER_ON_HIERARCHY_OVER_SUPPORT = "hos"
        # Privileges the support (eager policy):
        # for example, if the constraint set contains AlternatePrecedence(A, B) and Precedence(A, B),
        # and AlternatePrecedence(A, B) has a support of 0.89
        # whereas Precedence(A, B) has a support of 0.9, then AlternatePrecedence(A, B) is filtered out.
        self.EAGER_ON_SUPPORT_OVER_HIERARCHY = "soh"

        # Only filters out subsuming constraints, and only if ALL the subsuming ones in the whole hierarchy have the same
        # support
        self.CONSERVATIVE = "conservative"

        # The policy to use for redundancy resolution
        self.POLICY = self.EAGER_ON_SUPPORT_OVER_HIERARCHY
        # Terms referring to objects from schema.org
        self.SCHEMA_ORG_OBJ = [tok for obj in
                               ['Product', 'Products', 'Project', 'Product Group', 'Product Model', 'Vehicle',
                                'Alignment Object', 'Audience', 'Bed Details', 'Brand', 'Broadcast Channel',
                                'Broadcast Frequency Specification', 'Class', 'Computer Language', 'Data Feed Item',
                                'Term', 'Demand', 'Document Permission', 'Program', 'Energy Consumption Details',
                                'Entry Point', 'Enumeration', 'Floor Plan', 'Game Server', 'Geospatial Geometry',
                                'Goods', 'Grant', 'Health Insurance Plan', 'Health Plan Cost Specification',
                                'Health Plan Formulary', 'Health Plan Network', 'Invoice', 'Item List',
                                'Job Posting', 'Language', 'List Item', 'Media Subscription', 'Menu Item',
                                'Merchant Return Policy', 'Merchant Return Policy', 'Observation', 'Occupation',
                                'Experience Requirements', 'Offer', 'Order', 'Order Item', 'Parcel Delivery',
                                'Permit', 'Program Membership', 'Property', 'Property Value Specification',
                                'Quantity', 'Rating', 'Reservation', 'Role', 'Schedule', 'Seat', 'Series', 'Service',
                                'Service Channel', 'Speakable Specification', 'Statistical Population',
                                'Structured Value', 'Ticket', 'Trip', 'Virtual Location']
                               for tok in obj.lower().split(" ")]
        self.SCHEMA_ORG_OBJ_PROP = [tok for obj in ['Age', 'Name', 'Type', 'Price', 'Cost'] for tok in
                                    obj.lower().split(" ")]
        self.ACTION_CATEGORIES = {"create": "An action is categorized as Create if it is about creating a new object.",
                                  "transform": "An action is categorized as Transform if it is about changing the "
                                               "state of an object.",
                                  "move": "An action is categorized as Move if it is about moving an object from one "
                                          "place to another.",
                                  "preserve": "An action is categorized as Preserve if it is about preserving an "
                                              "object.",
                                  "destroy": "An action is categorized as Destroy if it is about destroying an object.",
                                  "separate": "An action is categorized as Separate if it is about separating an "
                                              "object from another object.",
                                  "combine": "An action is categorized as Combine if it is about combining two objects.",
                                  "communicate": "An action is categorized as Communicate if it is about "
                                                 "communicating with an object.",
                                  "decide": "An action is categorized as Decide if it is about deciding about an "
                                            "object.",
                                  "manage": "An action is categorized as Manage if it is about managing an object."}

        # Language models used in the project
        self.SPACY_MODEL = "en_core_web_sm"
        self.SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"
        self.WORD_EMBEDDINGS = "glove-wiki-gigaword-50"

        # Do we consider loops when mining constraints?
        self.LOOPS = True

        self.DECLARE_SUPPORT = 0.99

        # Server for MQI sets
        self.MQI_SERVER = "http://141.26.82.70:3000/"
        self.MQI_CONSTRAINTS = [Template.RESPONDED_EXISTENCE.templ_str, Template.CO_EXISTENCE.templ_str,
                                Template.RESPONSE.templ_str, Template.PRECEDENCE.templ_str, Template.RESPONSE.templ_str,
                                Template.CHAIN_RESPONSE.templ_str, Template.CHAIN_PRECEDENCE.templ_str,
                                Template.CHAIN_SUCCESSION, Template.SUCCESSION.templ_str,
                                Template.ALTERNATE_PRECEDENCE.templ_str,
                                Template.ALTERNATE_RESPONSE.templ_str, Template.ALTERNATE_SUCCESSION.templ_str,
                                Template.NOT_CO_EXISTENCE.templ_str, Template.NOT_RESPONSE.templ_str,
                                Template.NOT_SUCCESSION.templ_str, Template.NOT_CHAIN_SUCCESSION.templ_str]

        self.ACTION_IDX_TO_LABEL = {0: "create",
                                    1: "transform",
                                    2: "move",
                                    3: "preserve",
                                    4: "destroy",
                                    5: "separate",
                                    6: "combine",
                                    7: "communicate",
                                    8: "decide",
                                    9: "assess",
                                    10: "manage"
                                    }
        self.mitphb = {
            "create": [
                {
                    "build":
                        []
                },
                {
                    "develop":
                        [
                            {
                                "document":
                                    []
                            },
                            {
                                "discuss":
                                    []
                            }
                        ]
                },
                {
                    "perform":
                        [

                        ]
                },
                {
                    "calculate":
                        [
                            {
                                "compute":
                                    []
                            }
                        ]
                },
                {
                    "duplicate":
                        [
                            {
                                "copy":
                                    []
                            }
                        ]
                },
                {
                    "forecast":
                        []
                },
                {
                    "make":
                        []
                },
                {
                    "design":
                        [
                            {
                                "plan": []
                            }
                        ]
                },
                {
                    "produce":
                        []
                }
            ],
            "communicate": [
                {
                    "transfer":
                        [
                            {
                                "inform":
                                    []
                            },
                            {
                                "communicate":
                                    []
                            }
                        ]
                },
                {
                    "send":
                        [
                            {
                                "request":
                                    []
                            }
                        ]
                },
                {
                    "notify":
                        []
                }
                ,
                {
                    "receive":
                        []
                }
            ],
            "transform": [
                {
                    "modify":
                        []
                },
                {
                    "edit":
                        []
                },
                {
                    "improve":
                        [
                            {
                                "update":
                                    []
                            },
                            {
                                "complete":
                                    []
                            }
                        ]
                },
                {
                    "prepare":
                        [
                            {"cook": []}
                        ]
                },
                {
                    "process":
                        [

                        ]
                },
                {
                    "reduce":
                        [
                            {"depreciate": []}
                        ]
                },
                {
                    "evolve":
                        []
                }
            ],

            "move":
                [
                    {
                        "rotate":
                            []
                    },
                    {
                        "exchange":
                            []
                    },
                    {
                        "get":
                            []
                    },
                    {
                        "obtain":
                            []
                    },
                    {
                        "register":
                            []
                    },
                    {
                        "give":
                            [

                            ]
                    }
                ],
            "preserve": [
                {
                    "wait":
                        []
                },
                {
                    "continue":
                        [
                            {
                                "maintain":
                                    []
                            }
                        ]
                },
                {
                    "retain":
                        [
                            {
                                "package":
                                    []
                            }
                        ]
                },
                {
                    "store":
                        [
                            {
                                "document":
                                    [
                                        {
                                            "record":
                                                []
                                        }
                                    ]
                            },
                            {
                                "enter":
                                    []
                            }
                        ]
                }
            ],
            "destroy": [
                {
                    "retire":
                        [
                            {
                                "dispose": []
                            },
                            {
                                "dissolve": []
                            }
                        ]
                },
                {
                    "eliminate":
                        [
                            {
                                "obliterate": []
                            },
                            {
                                "delete": []
                            }
                        ]
                }
            ],
            "combine": [
                {
                    "group":
                        [
                            {
                                "organize":
                                    []
                            },
                            {
                                "match":
                                    []
                            },
                            {
                                "retrieve":
                                    []
                            },
                            {
                                "aggregate":
                                    []
                            },
                            {
                                "link":
                                    []
                            },
                            {
                                "align":
                                    []
                            }
                        ]
                },
                {
                    "integrate":
                        []
                },
                {
                    "connect":
                        []
                }
            ],
            "separate": [
                {
                    "disaggregate":
                        [
                            {
                                "classify":
                                    []
                            }
                        ]
                },
                {
                    "divide":
                        []
                },
                {
                    "segment":
                        [
                            {
                                "clarify":
                                    []
                            },
                            {
                                "diversify":
                                    []
                            }
                        ]
                },
                {
                    "extract":
                        [
                            {
                                "filter":
                                    []
                            },
                            {
                                "distill":
                                    []
                            }
                        ]
                }
            ],
            "decide": [
                {
                    "classify":
                        [
                            {
                                "sort":
                                    []
                            },
                            {
                                "score":
                                    []
                            },
                            {
                                "rank":
                                    []
                            }
                        ]
                },
                {
                    "select":
                        [
                            {
                                "determine":
                                    []
                            },
                            {
                                "check":
                                    []
                            }
                        ]
                },
                {
                    "test":
                        [
                            {
                                "verify":
                                    []
                            },
                            {
                                "assess":
                                    []
                            },
                            {
                                "control":
                                    []
                            }
                        ]
                },
                {
                    "allocate":
                        [
                            {
                                "budget":
                                    []
                            }
                        ]
                },
                {
                    "assign":
                        [
                            {
                                "match":
                                    []
                            },
                            {
                                "arrange":
                                    []
                            },
                            {
                                "schedule":
                                    []
                            },
                            {
                                "approve":
                                    []
                            },
                            {
                                "accept":
                                    []
                            },
                            {
                                "reject":
                                    []
                            }
                        ]
                }
            ],
            "manage": [
                {
                    "assign":
                        [
                            {
                                "match":
                                    []
                            },
                            {
                                "arrange":
                                    []
                            },
                            {
                                "schedule":
                                    []
                            }
                        ]
                },
                {
                    "allocate":
                        [
                            {
                                "budget":
                                    []
                            }
                        ]
                },
                {
                    "organize":
                        [
                            {
                                "cluster":
                                    []
                            },
                            {
                                "involve":
                                    []
                            },
                            {
                                "assemble":
                                    []
                            }
                        ]
                }
            ]
        }

