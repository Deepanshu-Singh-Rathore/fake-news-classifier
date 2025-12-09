import argparse
import os
from typing import List, Optional, Dict
from pathlib import Path

import joblib
import json
import time
import hashlib

import requests

try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None

from ..preprocess import clean_text


def canonical_url(url: str) -> str:
    # Basic canonicalization: strip fragments and common tracking params
    try:
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
        u = urlparse(url)
        # remove fragment and known trackers
        query = [(k, v) for k, v in parse_qsl(u.query, keep_blank_values=True)
                 if k.lower() not in {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}]
        return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(query, doseq=True), ""))
    except Exception:
        return url


def extract_fulltext(url: str) -> Optional[str]:
    if trafilatura is None:
        return None
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            return None
        article = trafilatura.extract(downloaded, include_comments=False, favor_recall=True)
        return article
    except Exception:
        return None


def get_newsapi_articles(api_key: str, query: Optional[str], country: Optional[str], category: Optional[str],
                         page_size: int, pages: int) -> List[Dict]:
    headers = {"X-Api-Key": api_key}
    base = "https://newsapi.org/v2/"
    items: List[Dict] = []
    endpoint = "top-headlines" if (country or category) and not query else "everything"

    for page in range(1, pages + 1):
        params: Dict[str, str] = {"pageSize": str(page_size), "page": str(page)}
        if endpoint == "everything":
            if query:
                params["q"] = query
            params["language"] = "en"
            params["sortBy"] = "publishedAt"
        else:
            if country:
                params["country"] = country
            if category:
                params["category"] = category
        resp = requests.get(base + endpoint, headers=headers, params=params, timeout=20)
        if resp.status_code == 429:
            # backoff simple
            time.sleep(2.0)
            resp = requests.get(base + endpoint, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        for a in articles:
            items.append({
                "source": (a.get("source") or {}).get("name"),
                "author": a.get("author"),
                "title": a.get("title"),
                "description": a.get("description"),
                "content": a.get("content"),
                "url": a.get("url"),
                "publishedAt": a.get("publishedAt"),
            })
        if len(articles) < page_size:
            break
    return items


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def build_record(raw: Dict, fulltext: Optional[str]) -> Dict:
    pieces = [raw.get("title") or "", raw.get("description") or "", raw.get("content") or ""]
    baseline_text = " \n".join(pieces).strip()
    # prefer fulltext when available and reasonably long
    text = fulltext if (fulltext and len(fulltext) > max(500, len(baseline_text))) else baseline_text
    text = text or baseline_text
    cleaned = clean_text(text)
    return {
        "source": raw.get("source"),
        "author": raw.get("author"),
        "title": raw.get("title"),
        "url": raw.get("url"),
        "publishedAt": raw.get("publishedAt"),
        "text": text,
        "clean": cleaned,
        "text_hash": hash_text(text),
    }


def classify_records(model, records: List[Dict], threshold: float, return_proba: bool = True) -> List[Dict]:
    outputs: List[Dict] = []
    X = [r["clean"] for r in records]
    if hasattr(model, "predict_proba") and return_proba:
        probs = model.predict_proba(X)
        classes = getattr(model, "classes_", [0, 1])
        try:
            real_idx = list(classes).index(1)
            fake_idx = list(classes).index(0)
        except ValueError:
            fake_idx, real_idx = 0, 1
        for r, p in zip(records, probs):
            real_p = float(p[real_idx])
            fake_p = float(p[fake_idx])
            label = "REAL" if real_p >= threshold else "FAKE"
            outputs.append({**r, "pred_label": label, "prob_real": real_p, "prob_fake": fake_p, "threshold": threshold})
    else:
        preds = model.predict(X)
        for r, y in zip(records, preds):
            label = "REAL" if int(y) == 1 else "FAKE"
            outputs.append({**r, "pred_label": label})
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Fetch live news (NewsAPI) and classify with trained model")
    parser.add_argument("--model", default=str(Path("models")/"fake_news_model.joblib"), help="Path to trained .joblib model")
    parser.add_argument("--api-key", default=os.getenv("NEWSAPI_KEY"), help="NewsAPI key (env NEWSAPI_KEY if not provided)")
    parser.add_argument("--query", default=None, help="Query for 'everything' endpoint; omit to use top-headlines")
    parser.add_argument("--country", default=None, help="Country code for top-headlines (e.g., us, gb)")
    parser.add_argument("--category", default=None, help="Category for top-headlines (e.g., technology, health)")
    parser.add_argument("--page-size", type=int, default=50, help="Articles per page")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to fetch")
    parser.add_argument("--fetch-full", action="store_true", help="Attempt to extract full text via trafilatura")
    parser.add_argument("--threshold", type=float, default=0.5, help="REAL probability threshold")
    parser.add_argument("--output", default=str(Path("data")/"live_predictions.jsonl"), help="Output JSONL path")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate by canonical URL and text hash")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Provide NewsAPI key via --api-key or env NEWSAPI_KEY")

    model = joblib.load(args.model)

    raw_items = get_newsapi_articles(
        api_key=args.api_key,
        query=args.query,
        country=args.country,
        category=args.category,
        page_size=args.page_size,
        pages=args.pages,
    )

    seen_urls = set()
    seen_hashes = set()
    records: List[Dict] = []

    for it in raw_items:
        url = it.get("url")
        if not url:
            continue
        url_c = canonical_url(url)
        if args.dedup and url_c in seen_urls:
            continue
        fulltext = extract_fulltext(url_c) if args.fetch_full else None
        rec = build_record(it, fulltext)
        h = rec["text_hash"]
        if args.dedup and h in seen_hashes:
            continue
        seen_urls.add(url_c)
        seen_hashes.add(h)
        records.append(rec)

    outputs = classify_records(model, records, threshold=args.threshold)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(outputs)} predictions to {out_path}")


if __name__ == "__main__":
    main()
