from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="semconstmining",
    packages=find_packages(),
    author='Adrian Rebmann',
    author_email='rebmann@uni-mannheim.de',
    version="0.1.6",
    description="long description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'tokenizers',
        'networkx',
        'deprecation',
        'scipy',
        'matplotlib',
        'psutil',
        'requests==2.28.1',
        'transformers==4.9.2',
        'pytorch-transformers==1.2.0',
        'torch==1.9.0',
        'six==1.16.0',
        'tqdm==4.64.1',
        'joblib==1.2.0',
        'pandas==1.5.3',
        'numpy==1.24.1',
        'matplotlib==3.6.2',
        'seaborn==0.12.1',
        'spacy==3.4.1',
        'stringcase==1.2.0',
        'nltk==3.8.1',
        'spacy_langdetect==0.1.2',
        'language_data==1.1',
        'metric-temporal-logic==0.4.0',
        'sentence_transformers==2.2.0',
        'mlxtend==0.21.0',
        'func-timeout==4.3.5',
        'pm4py==2.5.1',
        'gensim==4.1.2',
        "pylogics==0.1.1",
        "inflect==6.0.4",
    ]
)
