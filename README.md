# Retrieve-and-Rank-Multi-stage-pipeline-for-Skill-Extraction
This repository contains a multi-stage pipeline for skill extraction. The pipeline includes:

- **Stage 1**: A curriculum-based **bi-encoder retriever**, trained to retrieve relevant skill candidates for job description sentences.
- **Stage 2**: A **cross-encoder ranker**, trained with a binary ranking objective to determine which of the retrieved candidates are truly relevant.

Below is a high-level overview of the proposed framework:

![High Level Overview of Multi-Stage Skill Extraction Pipeline](Diagram.png)

## Instructions on use
Below, you are provided with a complete instruction on how to use the pipeline. For the most optimal training, use a CUDA-enabled environment with access to a GPU. After downloading the repository and extracting the Dataset folder, refer to the following steps:

### Installation
After creating a virtual Python environment with CUDA 12.1. Install the required packages:
<pre> pip install -r ./requirements.txt </pre>
### Stage 1- The bi-encoder retriever
To train a bi-encoder retriever following our curriculum training strategy, simply run:
<pre> bash ./train_retriever.sh </pre>
in your terminal. This should initiate the full training for the Retriever. You can test the retriever using:
<pre> bash ./test_retriever.sh </pre>
You can also specify different training sets in this file. The performance metric will be displayed after the algorithm runs. You can access the retrieved datasets from the output repository.
### Stage 2- The cross-encoder ranker
To train the cross-encoder, you should use:
<pre> bash ./train_ranker.sh </pre>
Then, for testing use:
<pre> bash ./test_ranker.sh </pre>
### LLM ranker baseline
If you want to test an LLM ranking baseline, run:
<pre> bash ./test_llm.sh </pre>
Here, you need to make sure to provide a designated OpenAI API key. 

We redistribute all the training and test data to ensure a smoother implementation. However, if you wish to use original data sources, refer to:
- [DECORTE](https://huggingface.co/datasets/TechWolf/Synthetic-ESCO-skill-sentences)
- [HOUSE](https://huggingface.co/datasets/TechWolf/skill-extraction-house)
- [TECH](https://huggingface.co/datasets/TechWolf/skill-extraction-tech)
- [TECHWOLF](https://huggingface.co/datasets/TechWolf/skill-extraction-techwolf)
- [SKILLSKAPE](https://github.com/magantoine/JobSkape/tree/main)

We thank the original authors for making these datasets publicly available. Please remember to cite them appropriately in any work using these resources.

### Cite
If you use this resource, please use the following citation:
'''
@inproceedings{bielinski2025retrieval,
  articleno    = {{5}},
  author       = {Bielinski, Aleksander and Brazier, David},
  booktitle    = {{Proceedings of the 5th Workshop on Recommender Systems for Human Resources (RecSys-in-HR 2025), in conjunction with the 16th ACM Conference on Recommender Systems}},
  editor       = {{Bogers, Toine and Bied, Guillaume and Decorte, Jean-Joris and Johnson, Chris and Kaya, Mesut}},
  issn         = {{1613-0073}},
  language     = {{eng}},
  location     = {{Prague, Czech Republic}},
  pages        = {{10}},
  publisher    = {{CEUR}},
  title        = {{From Retrieval to Ranking: A Two-Stage Neural Framework for Automated Skill Extraction}},
  url          = {{https://ceur-ws.org/Vol-4046/RecSysHR2025-paper_5.pdf}},
  volume       = {{4046}},
  year         = {{2025}},
}
'''
