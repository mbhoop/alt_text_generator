# AI Alt Text Generation

Evaluating whether vision-language models can generate accurate, accessibility-focused alt text for crowd-sourced ecological observations.

<!-- icons -->
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini%203.5%20Flash-4285F4?logo=googlegemini&logoColor=white)
![Claude](https://img.shields.io/badge/Anthropic-Claude%20Opus%20as%20Judge-D97757?logo=anthropic&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-LLaVA-black?logo=ollama&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Multi--Provider-1C3C3C?logo=langchain&logoColor=white)
![Accessibility](https://img.shields.io/badge/Accessibility-Alt%20Text-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents
- [Purpose](#purpose)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Setup](#setup)
- [Evaluation](#evaluation)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Contributors](#contributors)

## Purpose 
Citizen science platforms such as iNaturalist contain thousands of biodiversity observations that are invaluable to researchers, educators, and the public. Yet much of this content remains inaccessible to users with visual impairments because observation images rarely include meaningful alt text. When alt text is present, it often doesn't capture the ecological interaction that makes the image valuable.

This project explores whether modern vision-language models (VLMs) can help close that accessibility gap. It evaluates how reliably models can identify species and produce alt text that captures feeding interactions in real-world ecological observations.

The work was developed to support the Who-Eats-Whom project, an iNaturalist-based platform for exploring predator–prey relationships. alt-text-generator generates alt text describing the feeding interaction in each observation and evaluates its quality using an LLM judge. Beyond improving accessibility for one application, this research serves to better understand the role of VLMs in generating domain-specific, accessibility-focused alt-text at scale.

## Key Results
#### Scores by Sample Size
| Model             | Scores from Sample 3x | Scores from Sample 1x |
|-------------------|------------------------|------------------------|
| LLaVA             | 41/161 (25.5%)         | 14/63 (22.2%)          |
| Gemini 2.5 Flash  | 122/161 (75.8%)        | 47/63 (74.6%)          |
| Gemini 3.5 Flash  | 144/161 (89.4%)        | 58/63 (92.1%)          |

#### Scores by prompt version
| Model                | prompt_01     | prompt_02    | prompt_03    |
|----------------------|---------------|--------------|--------------|
| LLaVA                | 34/63 (54%)   | 12/63 (19%)  | 17/63 (27%)  |
| Gemini 2.5 Flash     | 58/63 (92%)   | 48/63 (76%)  | 48/63 (76%)  |
| Qwen3.5-397B-A17B    | 52/63 (83%)   | —            | —            |
| Gemini 3.5 Flash     | —             | —            | 49/63 (78%)  |


* LLaVA isn’t a viable model (~25% accuracy).
* Gem3.5 > Gem2.5 > LLaVA
* Gemini 3.5 has a worthwhile increase in quality. The difference is surfaced by the 3x sample where the models differ by ~14%.  
* Trends are stable across sample sizes. Patterns seen in the 1x sample hold in the 3x sample.
* A generic prompt like prompt_01, sometimes doesn’t provide a relevant enough alt text. But forcing a "what is feeding on what" framing induces hallucinated predator/prey relationships on images, especially in fungi where that framing doesn’t fit. 

Current VLMs are capable of generating useful accessibility-focused alt text for ecological observations, but reliability is highly dependent on prompt design and the biological context of the image. 

### Internal Documentation
Development notes, experiment logs, and findings are documented here in more detail for personal research tracking:
https://docs.google.com/document/d/1Eghxl1vRZRlYIQ-DTI1csZWmxLN8iAG8DbyL7SgbGoY/edit?tab=t.0#heading=h.tl72j0jwb20u


## Repository Structure
```text
alt-text-generator/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- config.py                     # prompt + models for a run 
|-- prompts.py                    # generation and judge prompts
|-- data/
|   |-- dataset.csv               # full Who-Eats-Whom records
|   |-- sample_1x.csv             # 1x stratified sample
|   |-- sample_3x.csv             # 3x stratified sample
|   |-- sample_1x_stripped.csv    # sample with metadata to use for judging
|-- images/                       # downloaded images 
|-- notebooks/
|   |-- sample_generator.ipynb    # generates samples and shows diversity distributions
|   |-- prompt_engineering.ipynb  # environment for testing models and refining prompts 
|-- src/
|   |-- download_images.py        # loads images from sample
|   |-- generate.py               # runs models to generate alt text
|   |-- render_evaluations.py     # renders evaluation of alt text
|   |-- render_confidence_eval.py # renders judge confidence breakdown
|   |-- utils.py
|-- results/
|   |-- 01_raw/                   # raw alt-text, one row per (image, model)
|   |-- 02_comparisons/           # same data, one row per image, models side by side
|   |-- 03_evaluations/           # judge scores 
|   |-- 04_renderings/            # HTML report

```

## Pipeline Overview

Each step uses the previous step's output, so running them in order
reproduces a full run. The diagram shows the data flow; the table shows what
to run and what each step produces.

```text
data/dataset.csv  
        |
        | stratified-by-phylum sample
        v  
data/sample_*.csv
        |
        | image download
        v  
images/<image_name>.jpg
        |
        | VLM alt-text generation
        v  
results/01_raw 
results/02_comparisons
        |
        | LLM-as-judge scoring
        v  
results/03_evaluations
        | 
        | results rendered
        v 
results/04_renderings
```

| # | Step | Command | Output |
|---|---|---|---|
| 1 | Generate a representative, phylum-stratified sample | run `notebooks/sample_generator.ipynb` | `data/sample_*.csv` |
| 2 | Download the sample's observation images | `python -m src.download_images` | `images/<image_name>.jpg` |
| 3 | Specify the prompt and models to run | edit `config.py` (`PROMPT_NAME = prompt_*`, `MODELS = [...]`) | — |
| 4 | Generate alt-text for each image | `python -m src.generate` | `results/01_raw/`, `results/02_comparisons/` (wide layout) |
| 5 | Score each alt-text with the LLM judge | paste the judge prompt into Claude Code (see [Evaluation](#evaluation)) | `results/03_evaluations/` |
| 6 | Render the evaluations as an HTML report | `python -m src.render_evaluations` | `results/04_renderings/` |


## Setup
`git clone https://github.com/mbhoop/alt-text-generator.git`
`cd alt-text-generator`
`pip install -r requirements.txt`

Load API keys for the models as outlined in `.env.example`.


## Evaluation
To quantify each model’s performance, LLM-as-judge approach was used. A top-tier vision and reasoning model, Claude Opus 4.8 via Claude Code, evaluated every alt-text. 

The judge was given the folder of images, the prompt each model had been given, and a CSV of the alt-text results.

For each row in the results CSV:
1. Open the observation image.
2. Look up the observation's metadata.
3. Read its alt text and score it per the rubric, with confidence levels.

## Claude Code as a Judge and its Limitations
This project is currently in an exploratory phase focused on understanding how well Opus 4.8 performs as an evaluation judge and getting baseline metrics on the alt-text quality of models. At this stage, evaluations are run through the Claude Code terminal rather than the API to experiment with different prompting strategies while keeping costs manageable. Running a single evaluation through the API is estimated to cost approximately $5–$10.

Although this setup is still experimental, several steps were taken to make the evaluation process as consistent as possible. Each run starts in a fresh Claude Code conversation to minimize leftover context, uses the same carefully developed judging prompt, and includes a review of the model’s reasoning to help refine the prompt over time. As the project progresses and the judging approach becomes more established, moving to the API will provide more consistent, repeatable scoring while building on the lessons learned during this exploratory phase.

## Future Scope
* Re-run evaluations with a deterministic scoring script.
* Have ecologists verify both the judgements and the alt-text.
* Improve handling of fungi, small organisms, parasites, and visually ambiguous taxa.
* Scale to larger samples (10x) to check whether the same trends replicate.
* Run multiple trials of both the judgements and the alt-text to see variability across trials.
* Re-run generations at the same image quality (medium_url).

## Contributors
- **Maithili Bhoop**
- Under supervision of **Dr. Aditi Mallavarapu**

This project builds on the BioVerify Who-Eats-Whom project, which used iNaturalist datasets for vision-based species identity verification. The dataset sampling and image download pipeline were initially adapted from that repository: https://github.com/surabhi1914/BioVerify.

