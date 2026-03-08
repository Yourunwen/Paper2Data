# Paper2Data & UrbanDataMiner

[![Website](https://img.shields.io/badge/Website-UrbanDataMiner-blue)](https://urbandataminer.github.io/)
[![Paper](https://img.shields.io/badge/Paper-KDD%202026-green)](#) 

Here is the code repo for the paper **"Paper2Data: Large-Scale LLM Extraction and Metadata Structuring of Global Urban Data from Scientific Literature"** (KDD 2026).

Our system uses Large Language Models (LLMs) to automatically identify dataset mentions in scientific papers and structure them using a unified urban data metadata schema. Based on this pipeline, we curate an open urban data discovery portal, **UrbanDataMiner**, which supports dataset-level search and filtering over more than 60,000 urban datasets extracted from over 15,000 Nature-affiliated publications.

🌐 **Try our data portal here:** [https://urbandataminer.github.io/](https://urbandataminer.github.io/)

## System Architecture

The system architecture is shown as follows:

![System Architecture](Framework2.png) 

The Paper2Data pipeline consists of six automated steps:
1. **Literature Curation:** Constructing a large-scale corpus of publications.
2. **Schema-guided Metadata Extraction:** Transforming unstructured mentions into structured records using LLMs.
3. **Evidence-aware Data Verification:** Grounding extracted metadata in the original text to mitigate hallucination.
4. **Metadata Refinement and Harmonization:** Standardizing temporal, geographic, and categorical fields.
5. **External Resource Linking:** Retrieving and verifying accessible URLs via a web search API.
6. **Data Portal Construction:** Indexing verified data cards into the UrbanDataMiner portal.

## Getting Started

### 1. Requirements
Install the required Python dependencies:
```bash
pip install -r requirements.txt
