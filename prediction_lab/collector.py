"""Observe-only round collector.

Opens a normal (headed) Chromium window. You log in yourself. The collector reads
the round-history strip text, and that is all it does: it never clicks, types,
bets, or cashes out. Each poll:

  read strip -> wait until two identical reads (avoid mid-animation) -> align
  against the stored tail -> store new rounds -> score/issue live predictions.
"""
import asyncio
import json
import time

from . import config, database as db
from .history import align, parse_snapshot

# Finds the element whose direct children look most like a strip of "1.37x" items.
DISCOVER_JS = r"""
(minItems) => {
  const re = /^\s*\d[\d,]*(\.\d{1,2})?\s*[xX×]\s*$/;
  let best = null;
  for (const el of document.querySelectorAll('*')) {
    const kids = el.children;
    if (kids.length < minItems) continue;
    let n = 0;
    for (const k of kids) { if (re.test(k.innerText || '')) n++; }
    if (n >= minItems && n >= 0.9 * kids.length && (!best || n > best.n)) best = {el, n};
  }
  if (!best) return null;
  const path = (e) => {
    const parts = [];
    while (e && e.nodeType === 1 && parts.length < 4) {
      let p = e.tagName.toLowerCase();
      const cls = [...e.classList].filter(c => !/\d{3,}|ng-|_ngcontent/.test(c)).slice(0, 2);
      if (cls.length) p += '.' + cls.join('.');
      parts.unshift(p); e = e.parentElement;
    }
    return parts.join(' > ');
  };
  return {
    container: path(best.el),
    texts: [...best.el.children].map(k => (k.innerText || '').trim()),
  };
}
"""

SELECTOR_JS = "(sel) => [...document.querySelectorAll(sel)].map(e => (e.innerText || '').trim())"


def safe_url(url: str) -> str:
    """Drop query string and fragment: game frame URLs carry session tokens."""
    from urllib.parse import urlsplit
    u = urlsplit(url)
    return f"{u.scheme}://{u.netloc}{u.path}"


async def read_strip(page, selector=None, min_items=10):
    """Return (texts, where) from the first frame that has a history strip, else (None, None)."""
    for frame in page.frames:  # the game usually lives in a cross-origin iframe
        try:
            if selector:
                texts = await frame.evaluate(SELECTOR_JS, selector)
                if texts:
                    return texts, {"frame": safe_url(frame.url), "selector": selector}
            else:
                found = await frame.evaluate(DISCOVER_JS, min_items)
                if found:
                    return found["texts"], {"frame": safe_url(frame.url), "container": found["container"]}
        except Exception:
            continue  # frame detached / navigating
    return None, None


def acquire_single_instance_lock():
    """Two collectors on one DB would record every round twice. Refuse to start."""
    import fcntl
    path = config.DB_PATH.with_suffix(".collector.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another collector is already running on this database. "
                         "Stop it first (Ctrl-C in its terminal).")
    return f  # keep the handle open for the life of the process


class Collector:
    def __init__(self, conn, source=config.PREDICTION_LAB_URL, live=None, order="newest_first"):
        self.conn = conn
        self.source = source
        self.live = live
        self.order = order
        self.segment = db.latest_segment(conn)
        self.last_processed = None

    def process_snapshot(self, values, where=None):
        """Store any new rounds implied by a (stable) snapshot. Returns new round pks."""
        if self.order == "oldest_first":
            values = list(reversed(values))
        if values == self.last_processed:
            return []
        tail = db.segment_tail(self.conn, self.segment, len(values)) if self.segment else []
        tail_vals = [r["multiplier"] for r in tail]
        a = align(values, tail_vals, config.MIN_OVERLAP)
        raw = {"where": where, "snapshot_len": len(values)}
        if a.new_count is None:
            reason = "first run" if self.segment is None else "gap: snapshot does not overlap stored tail"
            if self.segment is not None:
                db.void_pending(self.conn)
            self.segment = db.new_segment(self.conn, self.source, reason)
            pks = db.append_rounds(self.conn, self.segment, list(reversed(values)), self.source,
                                   is_bootstrap=True, raw=raw)
            print(f"[segment {self.segment}] {reason}; bootstrapped {len(pks)} rounds from the strip")
        elif a.new_count == 0:
            pks = []
        else:
            raw.update({"batch": a.new_count, "overlap": a.overlap, "ambiguous": a.ambiguous})
            new_vals = list(reversed(values[:a.new_count]))  # oldest first
            pks = db.append_rounds(self.conn, self.segment, new_vals, self.source, raw=raw)
            for v in new_vals:
                print(f"round: {v:.2f}x" + ("  (ambiguous alignment)" if a.ambiguous else ""))
        self.last_processed = values
        if pks and self.live:
            self.live.on_new_rounds(self.conn, self.segment, pks)
        return pks


async def run(selector=config.HISTORY_ITEM_SELECTOR, url=config.PREDICTION_LAB_URL, live=True,
              order="newest_first"):
    from playwright.async_api import async_playwright
    from .live import LivePredictor

    lock = acquire_single_instance_lock()
    conn = db.connect()
    collector = Collector(conn, url, LivePredictor() if live else None, order)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        print("Prediction Lab collector: OBSERVE ONLY. Log in and open the game manually.")
        print(f"Selector: {selector or 'auto-discover'} | DB: {config.DB_PATH}")
        prev, stable, last_ok, last_hint, announced = None, 0, time.time(), 0.0, None
        while True:
            if page.is_closed():
                print("Browser page closed; stopping.")
                return
            texts, where = await read_strip(page, selector)
            values = parse_snapshot(texts, config.MAX_MULTIPLIER) if texts else None
            now = time.time()
            if values is None:
                if texts and now - last_hint > 15:
                    print(f"[warn] strip found but an item failed to parse: {texts[:5]}...")
                    last_hint = now
                elif now - last_hint > 15:
                    print("[waiting] no history strip visible yet (log in / open the game)")
                    last_hint = now
                if now - last_ok > config.RELOAD_AFTER_SECONDS:
                    print("[recover] no valid strip for a while; reloading page")
                    try:
                        await page.reload(wait_until="domcontentloaded")
                    except Exception as e:
                        print(f"[recover] reload failed: {e}")
                    last_ok = now
                prev, stable = None, 0
            else:
                last_ok = now
                if where != announced:
                    print(f"[strip] reading from {json.dumps(where)}")
                    announced = where
                stable = stable + 1 if values == prev else 1
                prev = values
                if stable >= config.STABLE_READS:
                    collector.process_snapshot(values, where)
            await asyncio.sleep(config.POLL_SECONDS)


async def probe(url=config.PREDICTION_LAB_URL):
    """Open the page, wait for you to log in, then print what the collector would read."""
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        page = await (await browser.new_context(viewport={"width": 1400, "height": 900})).new_page()
        await page.goto(url, wait_until="domcontentloaded")
        while True:
            cmd = await asyncio.to_thread(input, "\nLog in, open the game, then press Enter to probe (q to quit): ")
            if cmd.strip().lower() == "q":
                return
            for frame in page.frames:
                try:
                    found = await frame.evaluate(DISCOVER_JS, 5)
                except Exception:
                    continue
                if found:
                    print(f"\nframe:     {safe_url(frame.url)}")
                    print(f"container: {found['container']}")
                    print(f"items ({len(found['texts'])}): {found['texts'][:15]}")
                    print(f"suggested: PREDICTION_LAB_HISTORY_ITEM_SELECTOR={found['container']} > *")
            print("\nCheck the order: is the FIRST item the most recent round? (default assumption)")
