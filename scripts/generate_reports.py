#!/usr/bin/env python3
import argparse, json, os
from datetime import datetime, timedelta, date
from dateutil import tz
import numpy as np
import requests
import yaml
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

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
            "hours_all": [[] for _ in range(24)],
            "hours_weekday": [[] for _ in range(24)],
            "hours_weekend": [[] for _ in range(24)],
            "days": set(),
            "hour_count_active": 0
        })
        per_player[p]["days"].add(r["date"])
        h = int(r["hour_local"])
        k = max(0, int(r["kills_hour"]))
        per_player[p]["hours_all"][h].append(k)
        wd = weekday_idx(r["date"])
        if wd < 5:
            per_player[p]["hours_weekday"][h].append(k)
        else:
            per_player[p]["hours_weekend"][h].append(k)
        if k >= thresholds["inactive_lt"]:
            per_player[p]["hour_count_active"] += 1

    summaries = []
    for pkey, agg in per_player.items():
        med_all = [int(np.median(x)) if x else 0 for x in agg["hours_all"]]
        med_wd = [int(np.median(x)) if x else 0 for x in agg["hours_weekday"]]
        med_we = [int(np.median(x)) if x else 0 for x in agg["hours_weekend"]]
        p95_all = [int(np.percentile(x, 95)) if x else 0 for x in agg["hours_all"]]
        top_hours = sorted(range(24), key=lambda h: med_all[h], reverse=True)[:3]
        days_seen = len(agg["days"])
        summaries.append({
            "player": agg["player"],
            "player_key": pkey,
            "days_seen": days_seen,
            "median24_all": med_all,
            "median24_weekday": med_wd,
            "median24_weekend": med_we,
            "p95_24": p95_all,
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

def is_last_day_of_month(d):
    return (d.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1) == d

def render_bar(values, title_text):
    hours = list(range(24))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(hours, values)
    ax.set_title(title_text)
    ax.set_xlabel("Stunde")
    ax.set_ylabel("Median Kills")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours], rotation=0)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

def post_discord(webhook, content, files=None, max_len=2000):
    if not webhook:
        return
    prefix = "```\n"
    suffix = "\n```"
    full = f"{prefix}{content}{suffix}"
    if len(full) > max_len:
        max_body = max_len - len(prefix) - len(suffix)
        content = content[:max_body]
        full = f"{prefix}{content}{suffix}"
    if files:
        multipart_files = []
        for idx, f in enumerate(files, 1):
            filename, filebytes = f
            multipart_files.append((f"file{idx}", (filename, filebytes, "image/png")))
        r = requests.post(webhook, data={"content": full}, files=multipart_files, timeout=30)
    else:
        r = requests.post(webhook, json={"content": full}, timeout=20)
    r.raise_for_status()

def build_weekly_message(player_sum, cfg, start_date, end_date):
    med = player_sum["median24_all"]
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
    return "\n".join(lines)

def build_monthly_message(player_sum, cfg, month_start, month_end):
    med = player_sum["median24_all"]
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
    return "\n".join(lines)

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

    max_msg = int(cfg["discord"].get("max_length", 1920))
    weekly_wd = int(cfg["report_times"].get("weekly_post_weekday", 6))
    weekly_hr = int(cfg["report_times"].get("weekly_post_hour_local", 23))
    monthly_hr = int(cfg["report_times"].get("monthly_post_hour_local", 23))

    do_weekly = args.force_weekly or (now.weekday() == weekly_wd and now.hour == weekly_hr)
    do_monthly = args.force_monthly or (is_last_day_of_month(now.date()) and now.hour == monthly_hr)

    did_anything = False

    if do_weekly:
        end_date = (now.date() - timedelta(days=1))
        start_date = end_date - timedelta(days=6)
        recs = read_hourly_range(hourly_dir, server, start_date, end_date)
        sums = summarize_player(recs, cfg)
        limit = int(cfg["discord"].get("top_players_per_period", 10))
        ordered = sorted(
            sums,
            key=lambda s: (s["active_hours_count"], sum(s["median24_all"])),
            reverse=True
        )
        topn = ordered if limit <= 0 else ordered[:limit]
        if not topn and args.discord_webhook:
            post_discord(args.discord_webhook, "Wochenbericht ohne Daten im Zeitraum")
        for s in topn:
            msg = build_weekly_message(s, cfg, start_date, end_date)
            img_all = render_bar(s["median24_all"], "Woche Stundenverteilung gesamt")
            img_wd = render_bar(s["median24_weekday"], "Woche Werktage")
            img_we = render_bar(s["median24_weekend"], "Woche Wochenende")
            files = [
                (f"{s['player_key']}_weekly_all.png", img_all.getvalue()),
                (f"{s['player_key']}_weekly_weekday.png", img_wd.getvalue()),
                (f"{s['player_key']}_weekly_weekend.png", img_we.getvalue()),
            ]
            outdir = os.path.join("reports", "weekly", server, s["player_key"])
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, f"{start_date}_to_{end_date}.txt"), "w", encoding="utf-8") as f:
                f.write(msg)
            for name, data in files:
                with open(os.path.join(outdir, name), "wb") as f:
                    f.write(data)
            if args.discord_webhook:
                post_discord(args.discord_webhook, msg[:max_msg], files=files, max_len=2000)
            did_anything = True

    if do_monthly:
        month_start = now.replace(day=1).date()
        month_end = now.date()
        recs = read_hourly_range(hourly_dir, server, month_start, month_end)
        sums = summarize_player(recs, cfg)
        limit = int(cfg["discord"].get("top_players_per_period", 10))
        ordered = sorted(
            sums,
            key=lambda s: (s["active_hours_count"], sum(s["median24_all"])),
            reverse=True
        )
        topn = ordered if limit <= 0 else ordered[:limit]
        if not topn and args.discord_webhook:
            post_discord(args.discord_webhook, "Monatsbericht ohne Daten im Zeitraum")
        for s in topn:
            msg = build_monthly_message(s, cfg, month_start, month_end)
            img_all = render_bar(s["median24_all"], "Monat Stundenverteilung gesamt")
            img_wd = render_bar(s["median24_weekday"], "Monat Werktage")
            img_we = render_bar(s["median24_weekend"], "Monat Wochenende")
            files = [
                (f"{s['player_key']}_monthly_all.png", img_all.getvalue()),
                (f"{s['player_key']}_monthly_weekday.png", img_wd.getvalue()),
                (f"{s['player_key']}_monthly_weekend.png", img_we.getvalue()),
            ]
            outdir = os.path.join("reports", "monthly", server, s["player_key"])
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, f"{month_start.strftime('%Y-%m')}.txt"), "w", encoding="utf-8") as f:
                f.write(msg)
            for name, data in files:
                with open(os.path.join(outdir, name), "wb") as f:
                    f.write(data)
            if args.discord_webhook:
                post_discord(args.discord_webhook, msg[:max_msg], files=files, max_len=2000)
            did_anything = True

    if not did_anything:
        print("Kein Bericht fällig")

if __name__ == "__main__":
    main()
