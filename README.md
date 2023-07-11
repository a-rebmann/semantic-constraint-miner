# Extracting Generic Execution Constraints from Reference Process Models for Semantic Conformance Checking


## License

The source code in this repository is licensed as follows. **Note that a different license applies to the dataset itself!**

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

The following license applies to the SAP-SAM dataset.

```
Copyright (c) 2022 by SAP.

SAP grants to Recipient a non-exclusive copyright license to the Model Collection to use the Model Collection for Non-Commercial Research purposes of evaluating Recipient’s algorithms or other academic research artefacts against the Model Collection. Any rights not explicitly granted herein are reserved to SAP. For the avoidance of doubt, no rights to make derivative works of the Model Collection is granted and the license granted hereunder is for Non-Commercial Research purposes only.

"Model Collection" shall mean all files in the archive (which are JSON, XML, or other representation of business process models or other models).

"Recipient" means any natural person receiving the Model Collection.

"Non-Commercial Research" means research solely for the advancement of knowledge whether by a university or other learning institution and does not include any commercial or other sales objectives.
```

## Setup
You have the following options to set up the project:

### Via pip

```shell
pip install .
```


### If you have issues with Macs with M1/M2 chip

We provide a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) environment.yml file that can be used to create a new environment and install the required dependencies.

You can use the following conda command to create the environment:
```shell
conda env create -f environment.yml  
```


### How to organize the data folder in the project you use this package in

    ├── data
    │   ├── bert              <- The bert model for tagging the event labels.
    │   ├── eval              <- Optional, if you want to use the evaluation.py script to run the full evaluation.
    │   ├── logs              <- Event logs you want to use.
    │   ├── interim           <- Intermediate data that has been preprocessed.
    │   ├── output            <- Folder to save results in.
    │   └── raw               <- The raw dataset should be placed in this folder.

You need to download the [dataset](insert link) and place it into the folder `./data/raw` such that the models are in `./data/raw/sap_sam_2022/models`.

After installing the package you need to download the Spacy language model for English and nltk stopwords

```shell
python -m spacy download en_core_web_sm
```

```shell
python -m nltk.downloader stopwords
```

Lastly, we use a custom Tagger to extract objects, actions, and various other semantic information from labels.
You need to download the four files from [here](https://gitlab.uni-mannheim.de/processanalytics/semantic-event-log-annotation/-/tree/main/.model/main) and put them into <code>data/bert/</code>
    
    
    
    
