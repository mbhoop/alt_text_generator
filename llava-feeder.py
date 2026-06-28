import logging
import os
from pathlib import Path

import pandas as pd
import requests
import ollama

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# /share/ftrscape/mbhoop/dataset/

def load_dataset(csv_path: str = "observations.csv") -> pd.DataFrame:
    """Load the dataset CSV into a pandas DataFrame."""
    # tab seperated csv
    return pd.read_csv(csv_path)

# /share/ftrscape/mbhoop/alt-text-generator/


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



PROMPT = "Generate a one sentence alt text for this image."
# /share/ftrscape/mbhoop/alt-text-generator/
image_dir = "images"
# /share/ftrscape/mbhoop/alt-text-generator/
results_path = "results/llava_results.csv"

def generate_alt(df):
    done = set()
    if Path(results_path).exists():
        done = set(pd.read_csv(results_path)["Image_name"])
    else:
        pd.DataFrame(columns=["Image_name", "scientific_name", "llava_alt_text"]).to_csv(results_path, index=False)
    
    for idx, row in df.iterrows():
        # Skip if already generated
        if row["Image_name"] in done:
            log.info("Skipping %s because alt text already exists", row["Image_name"])
            continue

        img_path = Path(image_dir) / f"{row['Image_name']}.jpg"
        response = ollama.chat(
            model='llava',
            messages=[{"role": "user", "content": PROMPT,
                       "images": [str(img_path)]}]
        )
        result = {"Image_name": row["Image_name"], "scientific_name": row["scientific_name"],
                  "llava_alt_text": response["message"]["content"].strip()}
        pd.DataFrame([result]).to_csv(
            results_path, mode='a', header=False, index=False)
        log.info("Done: %s", row['Image_name'])


if __name__ == "__main__":
    df = load_dataset()
    log.info("Loaded dataframe with shape: %s", df.shape)
    download_images(df)
    log.info("Image download complete.")
    generate_alt(df)
    log.info("Alt text generated!")
