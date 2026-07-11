from langchain.chat_models import init_chat_model
from prompts import PROMPTS

# edit as per run
PROMPT_NAME = "prompt_03"                 
SAMPLE_NAME = "sample_3x"                 
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
RESULTS_PATH = f"results/{PROMPT_NAME}_{SAMPLE_NAME}_results.csv"
COMPARISON_PATH = f"results/{PROMPT_NAME}_{SAMPLE_NAME}_comparison.csv"
EVAL_PATH = f"results/{PROMPT_NAME}_{SAMPLE_NAME}_evaluations.csv"
EVAL_HTML_PATH = f"results/{PROMPT_NAME}_{SAMPLE_NAME}_evaluations.html"

def load_models():    
    return [(name, MODEL_REGISTRY[name]()) for name in ACTIVE_MODELS]

def load_judge():    
    return MODEL_REGISTRY[JUDGE_MODEL]()


