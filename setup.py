from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="semconstmining",
    packages=find_packages(),
    author='Adrian Rebmann',
    author_email='adrianrebmann@gmmail.com',
    version="0.2.0",
    description="long description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        'tokenizers',
        'networkx',
        'deprecation',
        'scipy',
        'matplotlib',
        'psutil',
        'transformers',
        'torch',
        'sentence-transformers',
        'requests',
        'textblob',
        'typing-extensions',
        'jupyter',
        'pandas',
        'tqdm',
        'pylogics',
        'numpy==1.24.1',
        'gensim',
        'joblib==1.2.0',
        'matplotlib==3.6.2',
        'seaborn==0.12.1',
        'spacy>=3.7.5',
        'stringcase==1.2.0',
        'nltk==3.8.1',
        'spacy_langdetect==0.1.2',
        'language_data==1.1',
        'metric-temporal-logic==0.4.0',
        'mlxtend==0.21.0',
        'func-timeout==4.3.5',
        'pm4py==2.5.1',
        'typer==0.4.2',

    ]
)
