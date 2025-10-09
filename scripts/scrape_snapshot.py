#!/usr/bin/env python3
import argparse, json, os, sys, re
from datetime import datetime
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from dateutil import tz

def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def berlin_now(tzname):
    tzinfo = tz.gettz(tzname)
    return datetime.now(tzinfo)

def find_netherworld_table(soup, cfg):
    header_tag = cfg["selectors"]["header_tag"]
    header_text = cfg["selectors"]["header_text"]
    header = None
    for h in soup.find_all(header_tag):
        if h.get_text(strip=True) == header_text:
            header = h
            break
    if not header:
        raise RuntimeError("Netherworld Header nicht gefunden")
    table = header.find_next("table")
    if not table:
        raise RuntimeError("Netherworld Tabelle nicht gefunden")
    return table

def parse_table(table):
    rows = []
    for tr in table.find_all("tr"):
        th = tr.find("th")
        tds = tr.find_all("td")
        if not th or len(tds) < 2:
            continue
        try:
            rank = int(th.get_text(strip=True))
        except:
            continue
        name = tds[0].get_text(strip=True)
        kills_text = tds[1].get_text(strip=True).replace(",", "")
        if not re.match(r"^\d+$", kills_text):
            continue
        kills = int(kills_text)
        rows.append({"rank_today": rank, "player": name, "kills_today": kills})
    return rows

def save_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def normalized_key(name):
    return re.sub(r"\s+", "_", name.strip().lower())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # lazy import yaml only here to keep requirements clean for GH previewers
    import yaml  # noqa

    cfg = load_config(args.config)
    tzname = cfg["timezone"]
    server = cfg["server"]
    now = berlin_now(tzname)
    today_str = now.strftime("%Y-%m-%d")
    hour_str = now.strftime("%H")

    url = cfg["base_url"]
    resp = requests.get(url, timeout=30, headers={"User-Agent": "uw-scraper/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = find_netherworld_table(soup, cfg)
    rows = parse_table(table)

    snapshot = {
        "timestamp": now.isoformat(),
        "server": server,
        "data": rows
    }

    raw_dir = cfg["paths"]["raw_dir"]
    hourly_dir = cfg["paths"]["hourly_dir"]
    players_dir = cfg["paths"]["players_dir"]

    raw_day_dir = os.path.join(raw_dir, server, today_str)
    raw_path = os.path.join(raw_day_dir, f"{hour_str}.json")
    save_json(raw_path, snapshot)

    # compute hourly diffs
    # find previous snapshot file within same day
    prev_hour = None
    for h in range(int(hour_str) - 1, -1, -1):
        cand = os.path.join(raw_day_dir, f"{h:02d}.json")
        if os.path.exists(cand):
            prev_hour = cand
            break

    prev_map = {}
    if prev_hour:
        prev = load_json(prev_hour, {})
        for r in prev.get("data", []):
            prev_map[normalized_key(r["player"])] = r["kills_today"]

    current_map = {normalized_key(r["player"]): r for r in rows}

    hourly_records = []
    for key, r in current_map.items():
        prev_val = prev_map.get(key, 0)
        diff = r["kills_today"] - prev_val
        if diff < 0:
            # daily reset detected within day change or site reset
            diff = r["kills_today"]
        hourly_records.append({
            "timestamp": now.replace(minute=0, second=0, microsecond=0).isoformat(),
            "date": today_str,
            "hour_local": int(hour_str),
            "server": server,
            "player": r["player"],
            "player_key": key,
            "kills_hour": int(diff),
            "kills_cum_day": int(r["kills_today"]),
            "rank_today": r["rank_today"]
        })

    hourly_day_path = os.path.join(hourly_dir, server, f"{today_str}.json")
    existing = load_json(hourly_day_path, {"records": []})
    # de-duplicate same hour if rerun
    new_map = {(rec["player_key"], rec["hour_local"]): rec for rec in existing["records"]}
    for rec in hourly_records:
        new_map[(rec["player_key"], rec["hour_local"])] = rec
    merged = {"records": sorted(new_map.values(), key=lambda x: (x["player_key"], x["hour_local"]))}
    save_json(hourly_day_path, merged)

    # update player files with last seen
    for rec in hourly_records:
        pdir = os.path.join(players_dir, server)
        ensure_dir(pdir)
        ppath = os.path.join(pdir, f"{rec['player_key']}.json")
        pdata = load_json(ppath, None)
        if not pdata:
            pdata = {
                "player": rec["player"],
                "player_key": rec["player_key"],
                "server": server,
                "first_seen_date": today_str,
                "last_seen_date": today_str,
                "days_seen": [],
                "name_variants": list({rec["player"]})
            }
        pdata["last_seen_date"] = today_str
        if today_str not in pdata["days_seen"]:
            pdata["days_seen"].append(today_str)
        if rec["player"] not in pdata["name_variants"]:
            pdata["name_variants"].append(rec["player"])
        save_json(ppath, pdata)

    print(f"Snapshots und Stundenwerte gespeichert für {server} am {today_str} um {hour_str}")

if __name__ == "__main__":
    main()
