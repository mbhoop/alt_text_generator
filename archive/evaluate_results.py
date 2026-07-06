from dotenv import load_dotenv
import logging
from pathlib import Path
import base64
import re

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from PIL import Image

from prompts import PROMPT, PROMPT_NAME


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


image_dir = Path("images")
results_path = Path(f"results/{PROMPT_NAME}_results.csv")
ratings_path = Path(f"results/{PROMPT_NAME}_ratings.csv")
evaluation_path = Path(f"results/{PROMPT_NAME}_evaluation.csv")
summary_path = Path(f"results/{PROMPT_NAME}_model_summary.csv")


load_dotenv()
judge_model = init_chat_model("claude-opus-4-8", model_provider="anthropic")


# Build the evaluator prompt for one image/model result.
def message(row):
    img_path = image_dir / f"{row['Image_name']}.jpg"
    with Image.open(img_path) as img:
        mime_type = Image.MIME.get(img.format, "image/jpeg")

    data = base64.b64encode(img_path.read_bytes()).decode()
    url = f"data:{mime_type};base64,{data}"

    return [HumanMessage(content=[
        {"type": "text", "text": f"""
Rate this generated alt text from 1 to 3.

Original prompt:
{PROMPT}

Generated alt text:
{row["alt_text"]}

Rubric:
1 = accurate and follows the prompt
0 = incorrect, hallucinated, or not useful

Return only:
rating: <number>
reason: <very short reason>
"""},
        {"type": "image_url", "image_url": {"url": url}},
    ])]


# Convert a LangChain response into plain text.
def text_of(response):
    c = response.content
    return c if isinstance(c, str) else "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


# # Pull the numeric 1-3 rating out of the evaluator response.
def parse_rating(text):
    match = re.search(r"[1-3]", text)
    return int(match.group()) if match else None


# # Keep a short explanation only for low ratings.
def parse_reason(text, rating):
    if rating is None or rating == 3:
        return ""
    match = re.search(r"reason:\s*(.*)", text, re.I)
    return match.group(1).strip() if match else text.strip()


# Rate any image/model pairs that are not already in the ratings CSV.
def evaluate(df):
    ratings_path.parent.mkdir(exist_ok=True)

    done = set()
    if ratings_path.exists():
        prev = pd.read_csv(ratings_path)
        done = set(zip(prev["Image_name"], prev["model"]))

    for _, row in df.iterrows():
        name = row["Image_name"]
        model = row["model"]
        img_path = image_dir / f"{name}.jpg"

        if (name, model) in done:
            continue

        if not img_path.exists():
            log.warning("%s missing", img_path)
            continue

        response_text = text_of(judge_model.invoke(message(row))).strip()
        rating = parse_rating(response_text)
        reason = parse_reason(response_text, rating)

        result = {
            "Image_name": name,
            "image_url": row["image_url"],
            "alt_text": row["alt_text"],
            "model": model,
            "rating": rating,
            "reason": reason,
        }
        pd.DataFrame([result]).to_csv(
            ratings_path, mode="a", index=False,
            header=not ratings_path.exists())
        done.add((name, model))
        log.info("Rated: %s / %s", name, model)


# Create a comparison-style CSV with each model's alt text, rating, and reason.
def build_evaluation_table():
    ratings = pd.read_csv(ratings_path)
    wide = ratings[["Image_name", "image_url"]].drop_duplicates()

    for model in ratings["model"].unique():
        model_ratings = (
            ratings[ratings["model"] == model]
            [["Image_name", "alt_text", "rating", "reason"]]
            .rename(columns={
                "alt_text": model,
                "rating": f"{model}_rating",
                "reason": f"{model}_reason",
            })
        )
        wide = wide.merge(model_ratings, on="Image_name", how="left")

    wide.to_csv(evaluation_path, index=False)


# Create one overall score summary row per model.
def build_model_summary():
    summary = (
        pd.read_csv(ratings_path)
        .groupby("model", as_index=False)
        .agg(
            images_evaluated=("rating", "count"),
            average_rating=("rating", "mean"),
            median_rating=("rating", "median"),
        )
        .sort_values("average_rating", ascending=False)
    )
    summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    df = pd.read_csv(results_path)
    evaluate(df)
    build_evaluation_table()
    build_model_summary()
    log.info("All images evaluated.")
