
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

# Judging criteria for the LLM-as-judge / Claude Code judge. Versioned like
# PROMPTS so a methodology writeup can cite "rubric_01". Criteria only — the
# API script (src/evaluate.py) appends its own "respond as JSON" instruction.
EVAL_RUBRIC = {
    "rubric_01": """You are evaluating alt text written for a photo from a biodiversity dataset. 
    Good alt text names the main organism(s) and describes the feeding relationship 
    — what is eating or feeding on what (predator/prey, herbivory, parasitism, etc.) 
    — or correctly states that no feeding is visible when there is none.

Score 1 (accurate) when the alt text:
- correctly identifies the main subject, AND
- correctly describes the feeding relationship, OR correctly states that no feeding is occurring.

Score 0 (inaccurate) when the alt text:
- names the wrong organism or hallucinates something not in the image,
- reverses predator and prey,
- invents a feeding event that isn't happening,
- misses an obvious feeding interaction, or
- is too vague to convey the relationship.""",
}
