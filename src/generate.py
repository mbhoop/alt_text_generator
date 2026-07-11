import base64
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from .utils import load_dataset 

from config import (
    PROMPT,
    IMAGE_DIR,
    SAMPLE_PATH,
    RESULTS_PATH,
    COMPARISON_PATH,
    load_models,
)

log = logging.getLogger(__name__)

load_dotenv()


def message(label, img_path):
    data = base64.b64encode(Path(img_path).read_bytes()).decode()
    url = f"data:image/jpeg;base64,{data}"
    image_url = url if label == "llava" else {"url": url}
    return [HumanMessage(content=[
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": image_url},
    ])]

# normalizes langchain's response.content to a plain string
def text_of(response):
    c = response.content
    # concatenate the text
    return c if isinstance(c, str) else "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def generate_alt(df, models):
    Path(RESULTS_PATH).parent.mkdir(exist_ok=True)

    # Resume support: skip any (image, model) pair already in the results
    done = set()
    if Path(RESULTS_PATH).exists():
        prev = pd.read_csv(RESULTS_PATH)
        done = set(zip(prev["Image_name"], prev["model"]))

    for _, row in df.iterrows():
        name = row["Image_name"]
        img_path = Path(IMAGE_DIR) / f"{name}.jpg"

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
                RESULTS_PATH, mode="a", index=False,
                header=not Path(RESULTS_PATH).exists())
            done.add((name, label))
            log.info("Done: %s / %s", name, label)


def build_comparison_table():
    wide = (
        pd.read_csv(RESULTS_PATH)
        .pivot(index=["Image_name", "image_url"],
               columns="model",
               values="alt_text")
        .reset_index()
    )
    wide.to_csv(COMPARISON_PATH, index=False)


if __name__ == "__main__":
    models = load_models()
    df = load_dataset(SAMPLE_PATH)
    generate_alt(df, models)
    log.info("Alt text generated for all images!")
    build_comparison_table()
