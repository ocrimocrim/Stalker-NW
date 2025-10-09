#!/usr/bin/env python3
import argparse, json, os
from datetime import datetime, timedelta, date
from dateutil import tz
import numpy as np
import requests
import yaml

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def berlin_now(tzname):
    return datetime.now(tz.gettz(tzname))

def read_hourly_range(hourly_dir, server, start_date, end_date):
    cur = start_date
    records = []
    while cur <= end_date:
        p = os.path.join(hourly_dir, server, cur.isoformat() + ".json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.extend(data.get("records", []))
        cur += timedelta(days=1)
    return records

def group_by_day(records):
    per_day = {}
    for r in records:
        key = (r["player_key"], r["date"])
        per_day.setdefault(key, []).append(r)
    return per_day

def weekday_idx(date_str):
    # date_str kommt als YYYY-MM-DD
    y, m, d = map(int, date_str.split("-"))
    return date(y, m, d).weekday()

def summarize_player(records, cfg):
    thresholds = cfg["thresholds"]
    per_player = {}
    per_day = group_by_day(records)

    valid_day = {}
    for (pkey, d), lst in per_day.items():
        total = sum(max(0, r["kills_hour"]) for r in lst)
        valid_day[(pkey, d)] = total >= thresholds["min_daily_kills"]

    for r in records:
        if not valid_day.get((r["player_key"], r["date"]), False):
            continue
        p = r["player_key"]
        per_player.setdefault(p, {
            "player": r["player"],
            "player_key": p,
            "hours": [[] for _ in range(24)],
            "days": set(),
            "hour_count_active": 0
        })
        per_player[p]["days"].add(r["date"])
        h = int(r["hour_local"])
        k = max(0, int(r["kills_hour"]))
        per_player[p]["hours"][h].append(k)
        if k >= thresholds["inactive_lt"]:
            per_player[p]["hour_count_active"] += 1

    summaries = []
    for pkey, agg in per_player.items():
        med24 = [int(np.median(x)) if x else 0 for x in agg["hours"]]
        p95_24 = [int(np.percentile(x, 95)) if x else 0 for x in agg["hours"]]
        top_hours = sorted(range(24), key=lambda h: med24[h], reverse=True)[:3]
        days_seen = len(agg["days"])
        summaries.append({
            "player": agg["player"],
            "player_key": pkey,
            "days_seen": days_seen,
            "median24": med24,
            "p95_24": p95_24,
            "top_hours": top_hours,
            "active_hours_count": agg["hour_count_active"]
        })
    return summaries

def fmt_hour(h):
    return f"{h:02d}"

def hour_band(hlist):
    if not hlist:
        return "keine Daten"
    hlist = sorted(hlist)
    bands = []
    start = hlist[0]
    prev = hlist[0]
    for h in hlist[1:]:
        if h == prev + 1:
            prev = h
            continue
        bands.append((start, prev))
        start = h
        prev = h
    bands.append((start, prev))
    a, b = bands[0]
    if a == b:
        return f"{fmt_hour(a)} Uhr"
    return f"{fmt_hour(a)} bis {fmt_hour(b)} Uhr"

def classify_hour(k, cfg):
    t = cfg["thresholds"]
    if k < t["inactive_lt"]:
        return "inaktiv"
    if t["normal_ge"] <= k < t["normal_lt"]:
        return "normal"
    if t["mid_ge"] <= k < t["mid_lt"]:
        return "mittelaktiv"
    if k >= t["high_ge"]:
        return "hochaktiv"
    return "inaktiv"

def build_weekly_message(player_sum, cfg, start_date, end_date):
    med = player_sum["median24"]
    p95 = player_sum["p95_24"]
    top = player_sum["top_hours"]
    band_txt = hour_band(top)
    lines = []
    lines.append(f"Netherworld Wochenbericht Spieler {player_sum['player']}")
    lines.append(f"Zeitraum {start_date.isoformat()} bis {end_date.isoformat()} Europe Berlin")
    lines.append("Kurzfazit")
    lines.append(f"Aktivstes Fenster {band_txt}.")
    lines.append(f"Aktive Tage {player_sum['days_seen']}. Stunden mit Aktivität {player_sum['active_hours_count']}.")
    lines.append("Stundenmuster")
    for h in top:
        lines.append(f"{fmt_hour(h)} Uhr Median {med[h]} p95 {p95[h]} Kategorie {classify_hour(med[h], cfg)}")
    msg = "\n".join(lines)
    return msg[:cfg["discord"]["max_length"]]

def build_monthly_message(player_sum, cfg, month_start, month_end):
    med = player_sum["median24"]
    p95 = player_sum["p95_24"]
    top = player_sum["top_hours"]
    band_txt = hour_band(top)
    lines = []
    lines.append(f"Netherworld Monatsbericht Spieler {player_sum['player']}")
    lines.append(f"Zeitraum {month_start.strftime('%Y-%m')}")
    lines.append("Kurzfazit")
    lines.append(f"Aktivstes Fenster {band_txt}.")
    lines.append("Stundenrangfolge")
    rank = 1
    for h in top:
        lines.append(f"Platz {rank} {fmt_hour(h)} Uhr Median {med[h]} p95 {p95[h]}")
        rank += 1
    msg = "\n".join(lines)
    return msg[:cfg["discord"]["max_length"]]

def post_discord(webhook, content):
    if not webhook:
        return
    r = requests.post(webhook, json={"content": f"```\n{content}\n```"}, timeout=20)
    r.raise_for_status()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--discord-webhook", required=False, default="")
    ap.add_argument("--force-weekly", action="store_true")
    ap.add_argument("--force-monthly", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    tzname = cfg["timezone"]
    now = berlin_now(tzname)
    server = cfg["server"]
    hourly_dir = cfg["paths"]["hourly_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    did_anything = False

    # weekly
    do_weekly = args.force_weekly or (
        now.minute in cfg["report_times"]["daily_check_minutes"]
        and now.weekday() == cfg["report_times"]["weekly_post_weekday"]
    )
    if do_weekly:
        end_date = (now.date() - timedelta(days=1))
        start_date = end_date - timedelta(days=6)
        recs = read_hourly_range(hourly_dir, server, start_date, end_date)
        sums = summarize_player(recs, cfg)
        if sums:
            topn = sorted(
                sums,
                key=lambda s: (s["active_hours_count"], sum(s["median24"])),
                reverse=True
            )[:cfg["discord"]["top_players_per_period"]]
            for s in topn:
                msg = build_weekly_message(s, cfg, start_date, end_date)
                player_dir = os.path.join("reports", "weekly", server, s["player_key"])
                os.makedirs(player_dir, exist_ok=True)
                fname = os.path.join(player_dir, f"{start_date}_to_{end_date}.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(msg)
                if args.discord_webhook:
                    post_discord(args.discord_webhook, msg)
                did_anything = True
        else:
            # optional Statuspost
            if args.discord_webhook and args.force_weekly:
                post_discord(args.discord_webhook, "Wochenbericht ohne Daten im Zeitraum")

    # monthly
    do_monthly = args.force_monthly or (
        now.minute in cfg["report_times"]["daily_check_minutes"]
        and now.day == cfg["report_times"]["monthly_post_day"]
    )
    if do_monthly:
        first_of_this = now.replace(day=1).date()
        last_month_end = first_of_this - timedelta(days=1)
        month_start = last_month_end.replace(day=1)
        recs = read_hourly_range(hourly_dir, server, month_start, last_month_end)
        sums = summarize_player(recs, cfg)
        if sums:
            topn = sorted(
                sums,
                key=lambda s: (s["active_hours_count"], sum(s["median24"])),
                reverse=True
            )[:cfg["discord"]["top_players_per_period"]]
            for s in topn:
                msg = build_monthly_message(s, cfg, month_start, last_month_end)
                player_dir = os.path.join("reports", "monthly", server, s["player_key"])
                os.makedirs(player_dir, exist_ok=True)
                fname = os.path.join(player_dir, f"{month_start.strftime('%Y-%m')}.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(msg)
                if args.discord_webhook:
                    post_discord(args.discord_webhook, msg)
                did_anything = True
        else:
            if args.discord_webhook and args.force_monthly:
                post_discord(args.discord_webhook, "Monatsbericht ohne Daten im Zeitraum")

    if not did_anything:
        print("Kein Bericht fällig")

if __name__ == "__main__":
    main()
