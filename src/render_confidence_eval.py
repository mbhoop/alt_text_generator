import logging
import re
from pathlib import Path

import pandas as pd

from config import IMAGE_DIR, PROMPT, CONF_EVAL_PATH, CONF_HTML_PATH
from .render_evaluations import thumb_data_uri, GREEN, RED

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

CONF_COLORS = {"High": "#2e7d32", "Medium": "#f9a825", "Low": "#9e9e9e"}


def confidence_cols(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    cols = list(df.columns)
    out = {}
    for c in cols:
        if c.endswith("_reason") and f"{c[:-7]}_alt_text" in cols:
            model = c[:-len("_reason")]
            i = cols.index(c)
            out[model] = (cols[i + 1], cols[i + 2])
    return out


def pill(text: str, color: str) -> str:
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:10px;'
            f'font-weight:600;color:#fff;background:{color};">{text}</span>')


def render(df: pd.DataFrame) -> str:
    models = [re.sub(r"_alt_text$", "", c) for c in df.columns if c.endswith("_alt_text")]
    conf_cols = confidence_cols(df)

    head = ('<th style="padding:6px;">#</th>'
            '<th style="padding:6px;">Image</th>')
    for m in models:
        head += (f'<th style="padding:6px;">{m}</th>'
                 f'<th style="padding:6px;text-align:center;">{m}<br>score · confidence</th>')

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
            status = pill(str(score), GREEN if score == 1 else RED)

            level = "" if pd.isna(r[conf_cols[m][0]]) else str(r[conf_cols[m][0]]).strip()
            if level:
                status += " " + pill(level, CONF_COLORS.get(level, "#9e9e9e"))

            # reason for a failed score (red), then confidence caveat (grey italic).
            if score == 0 and not pd.isna(r[f"{m}_reason"]):
                status += (f'<div style="color:{RED};font-size:11px;margin-top:4px;">'
                           f'{r[f"{m}_reason"]}</div>')
            creason = r[conf_cols[m][1]]
            if not pd.isna(creason) and str(creason).strip():
                status += (f'<div style="color:#888;font-size:11px;font-style:italic;'
                           f'margin-top:4px;">{creason}</div>')

            cells += (f'<td style="padding:6px;vertical-align:top;max-width:250px;">{alt}</td>'
                      f'<td style="padding:6px;vertical-align:top;text-align:center;">{status}</td>')
        body += f'<tr style="border-bottom:1px solid #eee;page-break-inside:avoid;">{cells}</tr>'

    return (
        '<html><head><meta charset="utf-8"><style>'
        '@page{size:landscape;margin:1cm;}body{font-family:sans-serif;}</style></head><body>'
        '<div style="background:#f5f5f5;border-left:4px solid #607d8b;padding:12px 16px;'
        'margin-bottom:16px;white-space:pre-wrap;font-size:12px;">'
        f'<b>Prompt given to the models:</b>\n\n{PROMPT}\n\n'
        '<span style="font-style:italic;color:#555;">Confidence = the judge\'s '
        'self-reported certainty in its score (green High / amber Medium / grey Low).</span>'
        '</div>'
        '<table style="border-collapse:collapse;width:100%;font-size:12px;">'
        '<thead><tr style="border-bottom:2px solid #ddd;text-align:left;background:#fafafa;">'
        f'{head}</tr></thead><tbody>{body}</tbody></table></body></html>'
    )


def main():
    df = pd.read_csv(CONF_EVAL_PATH)
    Path(CONF_HTML_PATH).write_text(render(df), encoding="utf-8")
    log.info("Wrote %s (%d rows)", CONF_HTML_PATH, len(df))


if __name__ == "__main__":
    main()
