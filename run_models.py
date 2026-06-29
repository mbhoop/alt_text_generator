from dotenv import load_dotenv
import logging
from pathlib import Path

import pandas as pd
import base64
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def load_dataset(csv_path: str = "observations.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)



PROMPTS = {
    "prompt_01": "Generate a one sentence alt text for this image.",
    "prompt_02": "Write one concise alt-text sentence describing what is feeding on what",
}

PROMPT_NAME = "prompt_02"
PROMPT = PROMPTS[PROMPT_NAME]

image_dir = "images"
results_path = f"results/{PROMPT_NAME}_results.csv"


load_dotenv()
models = [
    # ("llava", init_chat_model("llava", model_provider="ollama")),
    ("Qwen3.5-397B-A17B", init_chat_model("Qwen/Qwen3.5-397B-A17B", model_provider="together")),
    # ("gemini-2.5-flash", init_chat_model("gemini-2.5-flash", model_provider="google_genai")),
    # ("claude-sonnet-4-6", init_chat_model("claude-sonnet-4-6", model_provider="anthropic")),
]


def message(label, img_path):
    data = base64.b64encode(Path(img_path).read_bytes()).decode()
    url = f"data:image/jpeg;base64,{data}"
    image_url = url if label == "llava" else {"url": url}
    return [HumanMessage(content=[
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": image_url},
    ])]


def text_of(response):
    c = response.content
    return c if isinstance(c, str) else "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def generate_alt(df):
    Path(results_path).parent.mkdir(exist_ok=True)

    # What's already done: the set of (image, model) pairs in the file.
    done = set()
    if Path(results_path).exists():
        prev = pd.read_csv(results_path)
        done = set(zip(prev["Image_name"], prev["model"]))

    for _, row in df.iterrows():
        name = row["Image_name"]
        img_path = Path(image_dir) / f"{name}.jpg"

        if not img_path.exists():
            log.warning("%s missing", img_path)
            continue

        for label, model in models:
            if (name, label) in done:
                continue

            alt = text_of(model.invoke(message(label, img_path))).strip()
            result = {"Image_name": name,
                      "image_url": row["medium_url"], "alt_text": alt, "model": label}
            pd.DataFrame([result]).to_csv(
                results_path, mode="a", index=False,
                header=not Path(results_path).exists())
            done.add((name, label))
            log.info("Done: %s / %s", name, label)


def build_comparison_table():
    wide = (
        pd.read_csv(results_path)
        .pivot(index=["Image_name", "image_url"],
               columns="model",
               values="alt_text")
        .reset_index()
    )
    wide.to_csv(f"results/{PROMPT_NAME}_comparison.csv", index=False)


if __name__ == "__main__":
    df = load_dataset()
    generate_alt(df)
    log.info("Alt text generated!")
    build_comparison_table()
