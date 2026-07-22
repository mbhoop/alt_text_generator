import base64
import io
import logging
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from config import IMAGE_DIR, PROMPT, EVAL_PATH, EVAL_HTML_PATH

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

GREEN, RED = "#2e7d32", "#c62828"
THUMB_PX = 360 


def thumb_data_uri(img_path: Path) -> str | None:
    """Downscale an image and return a base64 JPEG data URI (keeps HTML small)."""
    if not img_path.exists():
        return None
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render(df: pd.DataFrame) -> str:
    # model keys are inferred from the "<model>_alt_text" columns, in order.
    models = [re.sub(r"_alt_text$", "", c) for c in df.columns if c.endswith("_alt_text")]

    # header: #, Image, then (model, model score) per model.
    head = ('<th style="padding:6px;">#</th>'
            '<th style="padding:6px;">Image</th>')
    for m in models:
        head += (f'<th style="padding:6px;">{m}</th>'
                 f'<th style="padding:6px;text-align:center;">{m} score</th>')

    body = ""
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        uri = thumb_data_uri(Path(IMAGE_DIR) / f"{r['Image_name']}.jpg")
        img = (f'<img src="{uri}" width="180" style="border-radius:4px;">'
               if uri else '<span style="color:#999;">image missing</span>')
        link = (f'<div style="font-size:10px;margin-top:4px;">'
                f'<a href="{r["observation_url"]}">iNaturalist</a></div>')
        num = (f'<td style="padding:6px;vertical-align:top;color:#888;'
               f'font-weight:600;">{i}</td>')
        cells = f'{num}<td style="padding:6px;vertical-align:top;">{img}{link}</td>'

        for m in models:
            alt = "" if pd.isna(r[f"{m}_alt_text"]) else r[f"{m}_alt_text"]
            score = int(r[f"{m}_score"])
            color = GREEN if score == 1 else RED
            pill = (f'<span style="display:inline-block;padding:2px 10px;'
                    f'border-radius:10px;font-weight:600;color:#fff;'
                    f'background:{color};">{score}</span>')
            if score == 0:  # reason shown only for failures
                reason = "" if pd.isna(r[f"{m}_reason"]) else r[f"{m}_reason"]
                pill += (f'<div style="color:{RED};font-size:11px;'
                         f'margin-top:4px;">{reason}</div>')
            cells += (f'<td style="padding:6px;vertical-align:top;max-width:250px;">{alt}</td>'
                      f'<td style="padding:6px;vertical-align:top;text-align:center;">{pill}</td>')
        body += f'<tr style="border-bottom:1px solid #eee;page-break-inside:avoid;">{cells}</tr>'

    return (
        '<html><head><meta charset="utf-8"><style>'
        '@page{size:landscape;margin:1cm;}body{font-family:sans-serif;}</style></head><body>'
        '<div style="background:#f5f5f5;border-left:4px solid #607d8b;padding:12px 16px;'
        'margin-bottom:16px;white-space:pre-wrap;font-size:12px;">'
        f'<b>Prompt given to the models:</b>\n\n{PROMPT}</div>'
        '<table style="border-collapse:collapse;width:100%;font-size:12px;">'
        '<thead><tr style="border-bottom:2px solid #ddd;text-align:left;background:#fafafa;">'
        f'{head}</tr></thead><tbody>{body}</tbody></table></body></html>'
    )


def main():
    df = pd.read_csv(EVAL_PATH)
    Path(EVAL_HTML_PATH).write_text(render(df), encoding="utf-8")
    log.info("Wrote %s (%d rows)", EVAL_HTML_PATH, len(df))


if __name__ == "__main__":
    main()
