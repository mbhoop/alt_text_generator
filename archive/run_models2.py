from dotenv import load_dotenv
import logging
import os
from pathlib import Path

import pandas as pd
import requests
import base64
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def load_dataset(csv_path: str = "observations.csv") -> pd.DataFrame:
    """Load the dataset CSV into a pandas DataFrame."""
    # tab seperated csv
    return pd.read_csv(csv_path)


def download_images(
    df: pd.DataFrame,
    output_dir: str | os.PathLike = "images",
    obs_id_col: str = "observation_id",
    photo_id_col: str = "photo_id",
) -> None:
    """Download images from URLs in the dataframe and save them to disk.

    Filenames are of the form: obs_<observation_id>_photo_<photo_id>.jpg

    For each row, the function looks for valid URLs in this order:
    1. ``original_url``
    2. ``large_url``
    3. ``medium_url``
    4. ``square_url``
    """
    output_path = Path(output_dir)

    for idx, row in df.iterrows():
        obs_id = row.get(obs_id_col)
        photo_id = row.get(photo_id_col)

        # using Image_name instead of obs_id and photo_id
        filename = f"{row['Image_name']}.jpg"
        img_path = output_path / filename

        # Skip if already downloaded
        if img_path.exists():
            log.info("Skipping %s because it already exists", filename)
            continue

        # Build candidate URLs in the required priority order
        candidate_urls = []
        for col in ("original_url", "large_url", "medium_url", "square_url"):
            candidate = row.get(col)
            if isinstance(candidate, str) and not pd.isna(candidate):
                candidate_urls.append(candidate)

        if not candidate_urls:
            log.warning("No valid URLs found for %s %s", obs_id, photo_id)
            continue

        img_bytes = None
        for candidate in candidate_urls:
            try:
                resp = requests.get(candidate, timeout=15)
                resp.raise_for_status()
                img_bytes = resp.content
                break
            except requests.RequestException:
                continue

        if img_bytes is None:
            # All attempts failed
            continue

        img_path.write_bytes(img_bytes)
        log.info("Wrote %s to directory %s",
                 filename, img_path.parent.resolve())


load_dotenv()

PROMPT_1 = "Generate a one sentence alt text for this image."
PROMPT = PROMPT_1

image_dir = "images"
results_path = "results/prompt_01_results.csv"

models = [
    ("sonnet_4.6", init_chat_model("claude-sonnet-4-6",
     model_provider="anthropic"), "anthropic"),
    ("gemini_3.5", init_chat_model("gemini-3.5-flash",
     model_provider="google_genai"), "google"),
    ("llava", init_chat_model("llava", model_provider="ollama"), "ollama"),
]

meta_columns = ["scientific_name", "uri", "image_url"]
columns = meta_columns + [f"{label}_alt_text" for label, _, _ in models]
 
def message(provider, img_path):
    """Build the image+prompt message. Ollama and Gemini want slightly different image blocks."""
    data = base64.b64encode(Path(img_path).read_bytes()).decode()
    url = f"data:image/jpeg;base64,{data}"
    image_url = url if provider == "ollama" else {"url": url}
    return [HumanMessage(content=[
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": image_url},
    ])]
 
 
def text_of(response):
    """Get the reply text whether .content is a string or a list of blocks."""
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                   for b in content)
 
 
def save_results(results):
    """Write the full prompt as a title row, then the table beneath it."""
    Path(results_path).parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write(PROMPT + "\n")
    results.to_csv(results_path, mode="a")
 
 
def load_results():
    """Load existing results, or start fresh if the file is missing or in an old format."""
    if Path(results_path).exists():
        existing = pd.read_csv(results_path, skiprows=1)
        if "Image_name" in existing.columns:
            return existing.set_index("Image_name")
        # File is from an older format: set it aside instead of crashing.
        backup = results_path + ".bak"
        Path(results_path).rename(backup)
        log.warning("Old results format found; moved it to %s", backup)
    results = pd.DataFrame(columns=columns)
    results.index.name = "Image_name"
    return results
 
 
def generate_alt(df):
    results = load_results()
 
    for idx, row in df.iterrows():
        name = row["Image_name"]
        img_path = Path(image_dir) / f"{name}.jpg"
        results.loc[name, "scientific_name"] = row["scientific_name"]
        results.loc[name, "uri"] = row["uri"]
        results.loc[name, "image_url"] = row["medium_url"]
 
        for label, model, provider in models:
            col = f"{label}_alt_text"
 
            # Skip if this image already has alt text for this model.
            if col in results.columns and pd.notna(results.at[name, col]):
                log.info("Skipping %s / %s because alt text already exists",
                         name, label)
                continue
 
            response = model.invoke(message(provider, img_path))
            results.at[name, col] = text_of(response).strip()
            save_results(results)   # write after each one so progress is kept
            log.info("Done: %s / %s", name, label)

if __name__ == "__main__":
    df = load_dataset()
    log.info("Loaded dataframe with shape: %s", df.shape)
    download_images(df)
    log.info("Image download complete.")
    generate_alt(df)
    log.info("Alt text generated!")
