from langchain.chat_models import init_chat_model
from prompts import PROMPTS

# edit as per run
PROMPT_NAME = "prompt_03"                 
SAMPLE_NAME = "sample_3x_research_grade"                 
ACTIVE_MODELS = ["gemini-3.5-flash"]   
JUDGE_MODEL = "claude-opus-4-8"

MODEL_REGISTRY = {
    "llava":            lambda: init_chat_model("llava", model_provider="ollama"),
    "gemini-2.5-flash": lambda: init_chat_model("gemini-2.5-flash", model_provider="google_genai"),
    "gemini-3.5-flash": lambda: init_chat_model("gemini-3.5-flash", model_provider="google_genai"),
    "qwen3.5":          lambda: init_chat_model("Qwen/Qwen3.5-397B-A17B", model_provider="together"),
    "claude-sonnet-4-6": lambda: init_chat_model("claude-sonnet-4-6", model_provider="anthropic"),
}

PROMPT = PROMPTS[PROMPT_NAME]
IMAGE_DIR = "images"
SAMPLE_PATH = f"data/{SAMPLE_NAME}.csv"

# file naming convention
PROMPT_TAG = "p" + PROMPT_NAME.split("_")[-1]
SAMPLE_TAG = SAMPLE_NAME.split("_")[-1]
STEM = f"{PROMPT_TAG}_{SAMPLE_TAG}"

# file paths for results pipeline
RESULTS_PATH    = f"results/01_raw/{STEM}_results.csv"
COMPARISON_PATH = f"results/02_comparisons/{STEM}_comp.csv"
EVAL_PATH       = f"results/03_evaluations/{STEM}_eval.csv"
EVAL_HTML_PATH  = f"results/04_renderings/{STEM}_eval.html"
CONF_EVAL_PATH  = f"results/03_evaluations/{STEM}_confidence_eval.csv"
CONF_HTML_PATH  = f"results/04_renderings/{STEM}_confidence_eval.html"

def load_models():
    return [(name, MODEL_REGISTRY[name]()) for name in ACTIVE_MODELS]

def load_judge():    
    return MODEL_REGISTRY[JUDGE_MODEL]()


