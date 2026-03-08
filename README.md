# Paper2Data

[cite_start]Here is the code repo for the paper "Paper2Data: Large-Scale LLM Extraction and Metadata Structuring of Global Urban Data from Scientific Literature"[cite: 53, 54].

[cite_start]Our system uses Large Language Models (LLMs) to automatically identify dataset mentions in scientific papers and structure them using a unified urban data metadata schema[cite: 64]. [cite_start]Based on this pipeline, we curate an open urban data discovery portal, **UrbanDataMiner**, which supports dataset-level search and filtering over more than 60,000 urban datasets extracted from over 15,000 Nature-affiliated publications[cite: 63].

## System Architecture

The system architecture is shown as follows:

![System Architecture](framework2.png) 

[cite_start]The Paper2Data pipeline consists of six automated steps[cite: 442, 443]:
1. [cite_start]**Literature Curation:** Constructing a large-scale corpus of publications[cite: 457].
2. [cite_start]**Schema-guided Metadata Extraction:** Transforming unstructured mentions into structured records using LLMs[cite: 467].
3. [cite_start]**Evidence-aware Data Verification:** Grounding extracted metadata in the original text to mitigate hallucination[cite: 752].
4. [cite_start]**Metadata Refinement and Harmonization:** Standardizing temporal, geographic, and categorical fields[cite: 760, 761, 762, 763].
5. [cite_start]**External Resource Linking:** Retrieving and verifying accessible URLs via a web search API[cite: 769, 771].
6. [cite_start]**Data Portal Construction:** Indexing verified data cards into the UrbanDataMiner portal[cite: 776].
