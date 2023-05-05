import json
from pathlib import Path

import pandas as pd

from semconstmining.config import Config

config = Config(Path(__file__).parents[0].resolve(), "test_data")


def prepare_test_data():
    f = open("test_data/models/Credit_quote_creation_simplified_SAP Signavio.json")
    model = json.load(f)
    # new test data frame with these columns:
    # Revision ID,Model ID,Organization ID,Datetime,Model JSON,Description,Name,Type,Namespace
    df = pd.DataFrame(columns=["Revision ID", "Model ID", "Organization ID", "Datetime", "Model JSON", "Description",
                               "Name", "Type", "Namespace"])
    df = df.append({"Revision ID": "1a", "Model ID": "1a", "Organization ID": "1a", "Datetime": "1a", "Model JSON": json.dumps(model),
                    "Description": "1a", "Name": "Credit_quote_creation_simplified_SAP Signavio", "Type": "BPMN",
                    "Namespace": config.BPMN2_NAMESPACE}, ignore_index=True)
    df.to_csv("test_data/models/testmodels.csv", index=False)
    f.close()
