#!/usr/bin/env python3
"""Download first-instance criminal judgments from congbobanan.toaan.gov.vn,
up to --per-crime per BLHS crime (paper quota: 20 corpus + 5 test; the
extra covers judgments dropped by build_cases). See
vn_legal_graph/cases/congbobanan.py for the site flow.

    python scripts/crawl_congbobanan.py --articles 175 180     # try two crimes
    python scripts/crawl_congbobanan.py                        # every crime, resumable
    python scripts/crawl_congbobanan.py --count-only           # judgments per crime only

Output in --out (default data/raw/congbobanan/, not committed):
    progress.json    article -> {count, so_tham, pages, done}
    listings.jsonl   one result-list row per line (+ "dieu_tim": searched article)
    txt/<id>.txt     judgment text (pdftotext); the PDF is not kept. Scanned
                     PDFs (no text layer) are saved too but not counted

One request at a time, --delay seconds apart (default 2). No LLM calls.
Then: python scripts/build_cases.py --all-crimes --anle ... --congbobanan data/raw/congbobanan ...
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.cases.congbobanan import MIN_TEXT_WORDS, Client, parse_count, parse_listing, pdf_to_text

PAGE_SIZE = 20


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def fetch_text(client, judgment_id, txt_dir):
    """Judgment text, from txt_dir or downloaded (and saved, even when the
    PDF has no text layer, so it is not fetched again); None if the PDF
    cannot be downloaded."""
    path = os.path.join(txt_dir, f"{judgment_id}.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    try:
        text = pdf_to_text(client.pdf(judgment_id))
    except Exception as e:  # one bad PDF must not stop the crime
        print(f"  {judgment_id}: {e}")
        return None
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(path + ".tmp", path)
    return text


def crawl_crime(client, article, args, txt_dir, listings_path, seen_rows):
    """Read result pages and download first-instance judgments until
    --per-crime have a text layer (scanned PDFs and failed downloads do
    not count), the results end, or --max-pages is reached."""
    page = client.search(article)
    count = parse_count(page)
    if count is None:
        raise RuntimeError("trang kết quả không có số bản án")
    state = {"count": count, "so_tham": 0, "anh_quet": 0, "loi_tai": 0, "pages": 1, "done": False}
    if args.count_only or count == 0:
        state["done"] = True
        return state
    tried = set()
    while True:
        rows = parse_listing(page)
        with open(listings_path, "a", encoding="utf-8") as f:
            for row in rows:
                if (row["id"], article) not in seen_rows:
                    seen_rows.add((row["id"], article))
                    f.write(json.dumps(dict(row, dieu_tim=article), ensure_ascii=False) + "\n")
        for row in rows:
            if state["so_tham"] >= args.per_crime:
                break
            if row["cap_xet_xu"] != "Sơ thẩm" or row["loai_an"] != "Hình sự" or row["id"] in tried:
                continue
            tried.add(row["id"])
            text = fetch_text(client, row["id"], txt_dir)
            if text is None:
                state["loi_tai"] += 1
            elif len(text.split()) < MIN_TEXT_WORDS:
                state["anh_quet"] += 1
            else:
                state["so_tham"] += 1
        last = state["pages"] * PAGE_SIZE >= count or not rows
        if state["so_tham"] >= args.per_crime or last or state["pages"] >= args.max_pages:
            break
        page = client.next_page()
        state["pages"] += 1
    state["done"] = True
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/congbobanan")
    parser.add_argument("--law-json", default="data/processed/law_to_crime_vn.json")
    parser.add_argument("--articles", nargs="*", default=None, help="BLHS articles (default: every crime article).")
    parser.add_argument("--per-crime", type=int, default=30)
    parser.add_argument("--max-pages", type=int, default=10, help="Result pages read per crime at most.")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--redo", action="store_true", help="Crawl crimes already marked done again.")
    args = parser.parse_args()

    with open(args.law_json, encoding="utf-8") as f:
        laws = json.load(f)
    all_articles = [f"{l['id']}{l.get('suffix') or ''}" for l in laws if l["items"][0].get("crime")]
    articles = [str(a) for a in args.articles] if args.articles else all_articles

    txt_dir = os.path.join(args.out, "txt")
    os.makedirs(txt_dir, exist_ok=True)
    progress_path = os.path.join(args.out, "progress.json")
    listings_path = os.path.join(args.out, "listings.jsonl")
    progress = load_json(progress_path, {})
    seen_rows = set()
    if os.path.exists(listings_path):
        with open(listings_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                seen_rows.add((row["id"], row["dieu_tim"]))

    client = Client(delay=args.delay)
    site = client.open()
    missing = [a for a in articles if a not in site]
    if missing:
        print(f"Trang không có trong danh sách tội: {missing}")
    todo = [a for a in articles if a in site and (args.redo or not progress.get(a, {}).get("done"))]
    print(f"{len(articles)} tội, còn {len(todo)} tội phải cào (đã xong {len(articles) - len(todo) - len(missing)})")

    started = time.time()
    for i, article in enumerate(todo, 1):
        for attempt in range(2):
            try:
                state = crawl_crime(client, article, args, txt_dir, listings_path, seen_rows)
                break
            except RuntimeError as e:
                print(f"  Điều {article}: {e}; mở phiên mới")
                state = {"error": str(e), "done": False}
                try:
                    client.open()
                except RuntimeError as e2:
                    print(f"  không mở được phiên: {e2}")
        progress[article] = state
        save_json(progress_path, progress)
        elapsed = (time.time() - started) / 60
        print(f"[{i}/{len(todo)}] Điều {article}: {state.get('count')} bản án, "
              f"tải {state.get('so_tham', 0)} sơ thẩm (+{state.get('anh_quet', 0)} ảnh quét, {state.get('loi_tai', 0)} lỗi tải), "
              f"{state.get('pages', 0)} trang ({elapsed:.0f} phút)", flush=True)

    done = [p for p in progress.values() if p.get("done")]
    print(f"\nXong {len(done)}/{len(progress)} tội; {sum(p.get('so_tham', 0) for p in done)} bản án sơ thẩm; "
          f"{len(os.listdir(txt_dir))} file trong {txt_dir}")


if __name__ == "__main__":
    main()
