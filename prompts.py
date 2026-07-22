
# prompts given to models to generate alt-text
PROMPTS = {
    "prompt_01": "Generate a one sentence alt text for this image.",
    "prompt_02": "Write one concise alt-text sentence describing what is feeding on what",
    "prompt_03": """Write one concise alt-text sentence describing what is feeding on what.
                    Examples:
                    A white ermine stands in the snow, holding a captured grey rodent in its mouth.
                    A close-up of caterpillar chewing through the edge of a green leaf.
                    A brown frog sitting on rocky ground, swallowing a spider with its legs protruding from its mouth.
                    A grey, crested bird perched in a leafy green bush feeds on a yellow tulip.
                    A leopard gripping a horned antelope by the neck in green grass.""",
}

# initial judge model prompt
JUDGE_PROMPT = {
'''
You are an LLM judge for an alt-text quality experiment. Do only what is described here.

Use these files — do not open or reference anything else in the repo.
(ignore config.py, prompts.py, src/, and other results files):
- results/prompt_03_sample_1x_comparison.csv
  Columns: Image_name, image_url, then one column per model, each holding
  that model's alt text for the image.
- data/sample_1x.csv — only for its `uri` column, keyed by Image_name.
- images/<Image_name>.jpg — the actual photos. Open and look at each one.

Use the following rubric to judge each response:
Score 1 if correctly identifies the main subjects and the feeding relationship, or correctly says "no feeding" when nothing is feeding.
Score 0 if = wrong subject, hallucinated/reversed relationship, or too vague to be useful.

This was the prompt given to the models:
    "prompt_03": """Write one concise alt-text sentence describing what is feeding on what.
                    Examples:
                    A white ermine stands in the snow, holding a captured grey rodent in its mouth.
                    A close-up of caterpillar chewing through the edge of a green leaf.
                    A brown frog sitting on rocky ground, swallowing a spider with its legs protruding from its mouth.
                    A grey, crested bird perched in a leafy green bush feeds on a yellow tulip.
                    A leopard gripping a horned antelope by the neck in green grass.""",

TASK: For each row, open its image and score every model column 1 or 0 per the
rubric. For a 0, give a reason (<=12 words); for a 1, leave the reason blank.

Output a CSV at results/prompt_03_sample_1x_evaluations_2.csv:
- First two columns: Image_name, observation_url (observation_url = the row's
  `uri` from data/sample_1x.csv).
- Then, for EACH model column in the comparison file, three columns:
  <model>_alt_text, <model>_score, <model>_reason.
Write rows incrementally so the run is resumable.

Format example (fake data):
Image_name,observation_url,<model>_alt_text,<model>_score,<model>_reason
EXAMPLE,https://inaturalist.org/observations/0,A heron swallowing a fish.,1,
'''
}

# judge model prompt with confidence levels
JUDGE_PROMPT_CONFIDENCE = {
    '''
You are an LLM judge for an alt-text quality experiment. Do only what is described here.

Use these files. Do not open or reference anything else in the repo.
- @results/02_comparisons/p03_1x_comp.csv
  Columns: Image_name, image_url, then one column per model, each holding
  that model's alt text for the image.
- @data/sample_1x.csv — only for its `uri` column, keyed by Image_name.
- @images/<Image_name>.jpg — the actual photos. Open and view each one.

Use the following rubric to judge each response:
Score 1 if correctly identifies the main subjects and the feeding relationship, or correctly says "no feeding" when nothing is feeding.
Score 0 if = wrong subject, hallucinated/reversed relationship, or too vague to be useful.

For each score, also assign a confidence level for how confident you are in your scoring:
* High: The image and alt text are clear, and the score is unambiguous.
* Medium: The score is likely correct, but there is some uncertainty about the subject, species, feeding interaction, or wording.
* Low: The image or alt text is substantially ambiguous, multiple interpretations are plausible, or the score requires human review.

This was the prompt given to the models:
    "prompt_03": """Write one concise alt-text sentence describing what is feeding on what.
                    Examples:
                    A white ermine stands in the snow, holding a captured grey rodent in its mouth.
                    A close-up of caterpillar chewing through the edge of a green leaf.
                    A brown frog sitting on rocky ground, swallowing a spider with its legs protruding from its mouth.
                    A grey, crested bird perched in a leafy green bush feeds on a yellow tulip.
                    A leopard gripping a horned antelope by the neck in green grass.""",

TASK: For each row, open its image and score every model column 1 or 0 per the
rubric. For a 0, give a reason (<=12 words); for a 1, leave the reason blank. 
Assign a confidence level based on how confident you are in your judgement.
For Low or Medium, give explanation for the uncertainty (<=12 words); for High, leave the reason blank.

Output a CSV at results/04_evaluations/p03_1x_confidence_eval.csv:
- First two columns: Image_name, observation_url (observation_url = the row's
  `uri` from data/sample_1x.csv).
- Then, for each model column in the comparison file, five columns:
  <model>_alt_text, <model>_score, <model>_reason, confidence, confidence_reason.

Format example (fake data):
Image_name,observation_url,<model>_alt_text,<model>_score,<model>_reason,confidence,confidence_reason
EXAMPLE,https://...,A heron swallowing a fish.,1,,High,
'''
}

# judge model prompt with confidence levels and observation metadata
JUDGE_PROMPT_METADATA = {
'''
You are an LLM judge for an alt-text quality experiment. Do only what is described here.

Use these files. Do not open or reference anything else in the repo.
- @results/02_comparisons/p03_1x_comp.csv
  Columns: Image_name, image_url, then one column per model, each holding that model's alt text for the image.
- @data/sample_1x.csv
- @images/<Image_name>.jpg — the actual photos. Open and view each one.

Use the following rubric to judge each response:
Score 1 if correctly identifies the main subjects and the feeding relationship, or correctly says "no feeding" when nothing is feeding.
Score 0 if = wrong subject, hallucinated/reversed relationship, or too vague to be useful.

For each score, also assign a confidence level for how confident you are in your scoring:
* High: The image and alt text are clear, and the score is unambiguous.
* Medium: The score is likely correct, but there is some uncertainty about the subject, species, feeding interaction, or wording.
* Low: The image or alt text is substantially ambiguous, multiple interpretations are plausible, or the score requires human review.

This was the prompt given to the models:
    "prompt_03": """Write one concise alt-text sentence describing what is feeding on what.
                    Examples:
                    A white ermine stands in the snow, holding a captured grey rodent in its mouth.
                    A close-up of caterpillar chewing through the edge of a green leaf.
                    A brown frog sitting on rocky ground, swallowing a spider with its legs protruding from its mouth.
                    A grey, crested bird perched in a leafy green bush feeds on a yellow tulip.
                    A leopard gripping a horned antelope by the neck in green grass.""",

TASK: For each row, open its image and read the model's alt text.
@data/sample_1x.csv contains metadata about each observation, which can be treated as ground truth. 
Check the alt text against ONLY these columns (ignore the GroundingDino / bioclip / "Analysis" columns — those are other models' guesses):
scientific_name, common_name, the taxon_* taxonomy, pred_prey (Pred = this organism is feeding;
Prey = it is being fed upon), special_type_of_feeding. 
Score every model column 1 or 0 per the rubric.
For a 0, give a reason (<=12 words); for a 1, leave the reason blank. 
Assign a confidence level based on how confident you are in your judgement.
For Low or Medium, give explanation for the uncertainty (<=12 words); for High, leave the reason blank.

Output a CSV at results/04_evaluations/p03_1x_ground_truth_eval.csv:
- First two columns: Image_name, observation_url (observation_url = the row's
  `uri` from data/sample_1x.csv).
- Then, for each model column in the comparison file, five columns:
  <model>_alt_text, <model>_score, <model>_reason, confidence, confidence_reason.

Format example (fake data):
Image_name,observation_url,<model>_alt_text,<model>_score,<model>_reason,confidence,confidence_reason
EXAMPLE,https://...,A heron swallowing a fish.,1,,High,
'''
}


JUDGE_PROMPT_METADATA_2 = {
'''
You are an LLM judge for an alt-text quality experiment. Follow only the instructions below — do not open or reference any other file in the repo.

## Files
- ALT_TEXT_RESULTS: @results/01_raw/p03_1x_gem35_only_results.csv — model outputs to score
- SAMPLE: @data/sample_1x_stripped.csv — ground-truth metadata per observation
- IMAGES: @images/<Image_name>.jpg

## Context: prompt given to the models
"prompt_03": """Write one concise alt-text sentence describing what is feeding on what.
Examples:
A white ermine stands in the snow, holding a captured grey rodent in its mouth.
A close-up of caterpillar chewing through the edge of a green leaf.
A brown frog sitting on rocky ground, swallowing a spider with its legs protruding from its mouth.
A grey, crested bird perched in a leafy green bush feeds on a yellow tulip.
A leopard gripping a horned antelope by the neck in green grass."""

## Rubric 
Score 1 if the alt text correctly identifies the main subjects and the feeding relationship, per SAMPLE metadata. Otherwise score 0.
- Score 0 -> reason, ≤12 words
- Score 1 -> leave reason blank

## Confidence 
- High: image and alt text clear, score unambiguous — leave reason blank
- Medium: score likely right, some uncertainty — give reason, ≤12 words
- Low: image ambiguous, multiple readings plausible, or documented relationship not visible; needs human review — give reason, ≤12 words

## Task
For each row in ALT_TEXT_RESULTS:
1. Open the observation image.
2. Look up the observation's metadata in SAMPLE.
3. Read its alt text and score it per the rubric above, with confidence.

## Output
Write a CSV to results/04_evaluations/p03_1x_gem35_only_ground_truth_eval.csv

Columns:
Image_name, observation_url, <model>_alt_text, score, reason, confidence, confidence_reason

- observation_url = SAMPLE's `uri` field for that row

Example row:
Image_name, observation_url, <model>_alt_text, score, reason, confidence, confidence_reason
EXAMPLE,https://...,A heron swallowing a fish.,1,,High,
'''
}
