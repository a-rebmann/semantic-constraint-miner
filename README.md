# Mining Constraints from Reference Process Models for Detecting Best-Practice Violations in Event Logs

## About
This repository contains the implementation, data, evaluation scripts, and results as described in the manuscript
<i>Mining Constraints from Reference Process Models for Detecting Best-Practice Violations in Event Logs</i>.



## Setup
Setup a virtual env and then run the following command in the project root folder:
```shell
pip install .
```


### How to organize the data folder of the project

Create a datafolder on the same level, i.e., in the same directory, as the project root folder. 
The folder structure should look like this:

    ├── data
    │   ├── eval              <- Optional, if you want to use the evaluation.py script to run the full evaluation.
    │   ├── logs              <- Event logs you want to use.
    │   ├── interim           <- Intermediate data that has been preprocessed.
    │   ├── output            <- Folder to save results in.
    │   └── raw               <- The raw dataset should be placed in this folder.

The full process model dataset (a subset of that was used in our experiments) can be downloaded from here: [dataset](https://zenodo.org/record/7012043); place it into the folder `./data/raw` 
such that the models are in `./data/raw/sap_sam_2022/models`. Note that for the evaluation experiments we created a 
filtered version of the dataset, which is included in `project_root/eval_data`.

After installing the package you need to download the Spacy language model for English and nltk stopwords

```shell
python -m spacy download en_core_web_sm
```

```shell
python -m nltk.downloader stopwords
```


## Usage
1. Place the event logs you want to use in the `data/logs` folder.
2. Adapt the CURRENT_LOG_FILE variable in `main.py` to the name of the log file you want to use.
3. Make sure thr name of the model collection you want to use is set in the `MODEL_COLLECTION` variable in `main.py`.
4. Run the `main.py` script 

    
## Evaluation
### Results from the paper and additional results
The results reported in the paper can be obtained using a provided Python notebook  (<code>notebooks/evaluation_results.pynb</code>). 
It also contains additional results that we could not include in the paper due to space reasons.

### Reproduce
Proceed as explained in this section to reproduce the results from scratch.

#### Data
As explained above the data used must be stored in the data folder `data/raw`.
We provide the data used in the experiments in `project_root/eval_data`. 
1. Model collection: Move the entire `semantic_sap_sam_filtered` folder inside `project_root/eval_data` to `data/raw` to use it.
2. Noisy logs: Move the `semantic_sap_sam_filtered_noisy.pkl` file inside `project_root/eval_data` to `data/eval` to use it.

#### Run the experiments
The experiments can be run using the `evaluate.py` script.
There are many configuration options that can be defined in the `eval_configurations` dictionary in the script. There are already some examples in there.
