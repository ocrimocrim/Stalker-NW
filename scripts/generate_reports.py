#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
from io import BytesIO
from datetime import datetime, timedelta, date

from dateutil import tz
import numpy as np
import yaml
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa


# ---------------------------
# Config / IO helpers
# ---------------------------

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def berlin_now(tzname):
    return datetime.now(tz.gettz(tzname))

def read_hourly_range(hourly_dir, server, start_date, end_date):
    """Liest data/hourly/<server>/<YYYY-MM-DD>.json für einen Datumsbereich und gibt die records-Liste zurück."""
    cur = start_date
    out = []
    while cur <= end_date:
        p = os.path.join(hourly_dir, server, cur.isoformat() + ".json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                out.extend(data.get("records", []))
        cur += timedelta(days=1)
    return out


# ---------------------------
# Aggregation
# ---------------------------

def weekday_idx(date_str):
    y, m, d = map(int, date_str.split("-"))
    return date(y, m, d).weekday()  # 0=Mo .. 6=So

def summarize_player(records, cfg):
    """
    Aggregiert je Spieler:
      - median je Stunde: all / weekdays / weekend
      - p95 je Stunde (all)
      - sum je Stunde: all / weekdays / weekend
      - Zählwerte (Tage gesehen, aktive Stunden)
    Nur Tage mit >= min_daily_kills gehen in die Auswertung ein.
    """
    t = cfg["thresholds"]
    # Tagesfilter
    per_day = {}
    for r in records:
        key = (r["player_key"], r["date"])
        per_day.setdefault(key, []).append(r)
    day_ok = {}
    for (pkey, d), lst in per_day.items():
        total = sum(max(0, int(x.get("kills_hour", 0))) for x in lst)
        day_ok[(pkey, d)] = (total >= t["min_daily_kills"])

    # Rohsammlung pro Spieler
    per_player = {}
    for r in records:
        pkey = r["player_key"]
        if not day_ok.get((pkey, r["date"]), False):
            continue
        h = int(r["hour_local"])
        k = max(0, int(r.get("kills_hour", 0)))

        agg = per_player.setdefault(pkey, {
            "player": r["player"],
            "player_key": pkey,
            "days": set(),
            # Median-Buckets
            "hours_all": [[] for _ in range(24)],
            "hours_weekday": [[] for _ in range(24)],
            "hours_weekend": [[] for _ in range(24)],
            # Summen-Kanäle
            "sum_all": [0]*24,
            "sum_weekday": [0]*24,
            "sum_weekend": [0]*24,
            # Aktivitätszähler
            "hour_count_active": 0,
        })
        agg["days"].add(r["date"])
        agg["hours_all"][h].append(k)
        if weekday_idx(r["date"]) < 5:
            agg["hours_weekday"][h].append(k)
            agg["sum_weekday"][h] += k
        else:
            agg["hours_weekend"][h].append(k)
            agg["sum_weekend"][h] += k
        agg["sum_all"][h] += k
        if k >= t["normal_ge"]:  # „aktiv“ Schwelle = normal_ge
            agg["hour_count_active"] += 1

    # Abschluss: Kennzahlen je Spieler
    results = []
    for pkey, agg in per_player.items():
        med_all = [int(np.median(x)) if x else 0 for x in agg["hours_all"]]
        med_wd  = [int(np.median(x)) if x else 0 for x in agg["hours_weekday"]]
        med_we  = [int(np.median(x)) if x else 0 for x in agg["hours_weekend"]]
        p95_all = [int(np.percentile(x, 95)) if x else 0 for x in agg["hours_all"]]

        results.append({
            "player": agg["player"],
            "player_key": pkey,
            "days_seen": len(agg["days"]),
            # Median
            "median24_all": med_all,
            "median24_weekday": med_wd,
            "median24_weekend": med_we,
            "p95_24": p95_all,
            # Summen
            "sum24_all": agg["sum_all"],
            "sum24_weekday": agg["sum_weekday"],
            "sum24_weekend": agg["sum_weekend"],
            # Aktivität
            "active_hours_count": agg["hour_count_active"],
        })
    return results


# ---------------------------
# Auswertung „aktivstes Fenster“
# ---------------------------

def longest_active_window(sum_array, threshold):
    """
    sum_array: Liste Länge 24 (Wochensummen pro Stunde, All-Days)
    threshold: int; eine Stunde ist „aktiv“, wenn sum >= threshold
    Rückgabe: (start_h, end_h, total_sum)   (inklusive Grenzen)
    Bei Gleichstand gewinnt Fenster mit höherer total_sum.
    Wenn gar nichts aktiv: gibt (None, None, 0) zurück.
    """
    best = (None, None, 0, 0)  # (s, e, length, total)
    s = None
    cur_total = 0
    for h in range(24):
        v = sum_array[h]
        is_active = v >= threshold
        if is_active and s is None:
            s = h
            cur_total = v
        elif is_active and s is not None:
            cur_total += v
        elif (not is_active) and s is not None:
            e = h - 1
            length = e - s + 1
            cand = (s, e, length, cur_total)
            best = pick_better_window(best, cand)
            s = None
            cur_total = 0
    if s is not None:
        e = 23
        length = e - s + 1
        cand = (s, e, length, cur_total)
        best = pick_better_window(best, cand)

    if best[0] is None:
        return (None, None, 0)
    return (best[0], best[1], best[3])

def pick_better_window(a, b):
    # Vergleicht (s, e, length, total)
    if a[0] is None:
        return b
    if b[2] > a[2]:
        return b
    if b[2] == a[2] and b[3] > a[3]:
        return b
    return a


# ---------------------------
# Plotting (ENGLISH ONLY)
# ---------------------------

def render_grouped_bars(all_series, wd_series, we_series, title_text, ylabel, y_max=None):
    """
    Zeichnet pro Stunde 3 Balken: All / Weekdays / Weekend in einem Plot.
    English labels/titles only (as requested).
    """
    hours = np.arange(24)
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.bar(hours - width, all_series, width, label="All")
    ax.bar(hours,         wd_series,  width, label="Weekdays")
    ax.bar(hours + width, we_series,  width, label="Weekend")

    ax.set_title(title_text)
    ax.set_xlabel("Hour")
    ax.set_ylabel(ylabel)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    if y_max is not None and y_max > 0:
        ax.set_ylim(0, y_max * 1.05)
    ax.legend(loc="upper right", ncols=3, frameon=False)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------
# Discord
# ---------------------------

def post_discord(webhook, content, files=None, max_len=2000):
    if not webhook:
        return
    prefix, suffix = "```\n", "\n```"
    full = f"{prefix}{content}{suffix}"
    if len(full) > max_len:
        max_body = max_len - len(prefix) - len(suffix)
        full = f"{prefix}{content[:max_body]}{suffix}"
    if files:
        mfiles = []
        for idx, (name, data, mime) in enumerate(files, 1):
            mfiles.append((f"file{idx}", (name, data, mime)))
        r = requests.post(webhook, data={"content": full}, files=mfiles, timeout=30)
    else:
        r = requests.post(webhook, json={"content": full}, timeout=20)
    r.raise_for_status()


# ---------------------------
# Messages (Deutsch, ohne Kategorien)
# ---------------------------

def fmt_h(h):
    return f"{h:02d}"

def window_text(s, e):
    if s is None:
        return "kein aktives Fenster"
    if s == e:
        return f"{fmt_h(s)} Uhr"
    return f"{fmt_h(s)}–{fmt_h(e)} Uhr"

def build_weekly_message(s, cfg, start_date, end_date, top_n=3):
    # Top-Stunden nach Wochensumme (All-Days)
    sums = s["sum24_all"]
    p95  = s["p95_24"]
    top_hours = sorted(range(24), key=lambda h: sums[h], reverse=True)[:top_n]

    thr = cfg["thresholds"]["normal_ge"]
    s_h, e_h, win_sum = longest_active_window(sums, thr)

    lines = []
    lines.append(f"Netherworld Wochenbericht Spieler {s['player']}")
    lines.append(f"Zeitraum {start_date.isoformat()} bis {end_date.isoformat()} Europe Berlin")
    lines.append("Kurzfazit")
    lines.append(f"Aktivstes Fenster {window_text(s_h, e_h)} (Summe: {win_sum}).")
    lines.append(f"Aktive Tage {s['days_seen']}. Stunden mit Aktivität {s['active_hours_count']}.")
    lines.append("Top-Stunden (nach Wochen-Summe)")
    for h in top_hours:
        lines.append(f"{fmt_h(h)} Uhr Summe {sums[h]} (p95 {p95[h]})")
    return "\n".join(lines)

def build_monthly_message(s, cfg, month_start, month_end, top_n=3):
    sums = s["sum24_all"]
    p95  = s["p95_24"]
    top_hours = sorted(range(24), key=lambda h: sums[h], reverse=True)[:top_n]

    thr = cfg["thresholds"]["normal_ge"]
    s_h, e_h, win_sum = longest_active_window(sums, thr)

    lines = []
    lines.append(f"Netherworld Monatsbericht Spieler {s['player']}")
    lines.append(f"Zeitraum {month_start.strftime('%Y-%m')}")
    lines.append("Kurzfazit")
    lines.append(f"Aktivstes Fenster {window_text(s_h, e_h)} (Summe: {win_sum}).")
    lines.append(f"Aktive Tage {s['days_seen']}. Stunden mit Aktivität {s['active_hours_count']}.")
    lines.append("Top-Stunden (nach Monats-Summe)")
    for h in top_hours:
        lines.append(f"{fmt_h(h)} Uhr Summe {sums[h]} (p95 {p95[h]})")
    return "\n".join(lines)


# ---------------------------
# Utils
# ---------------------------

def is_last_day_of_month(d):
    return (d.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1) == d

def global_ymax(players, key_triplet):
    """
    Bestimmt globales y-Max über alle Spieler und alle 3 Serien eines Typs.
    key_triplet: ("median24_all","median24_weekday","median24_weekend") ODER ("sum24_all","sum24_weekday","sum24_weekend")
    """
    m = 0
    for s in players:
        for k in key_triplet:
            arr = s.get(k, [])
            if arr:
                m = max(m, max(arr))
    return m


# ---------------------------
# Main
# ---------------------------

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
    limit = int(cfg["discord"].get("top_players_per_period", 10))

    weekly_wd = int(cfg["report_times"].get("weekly_post_weekday", 6))
    weekly_hr = int(cfg["report_times"].get("weekly_post_hour_local", 23))
    monthly_hr = int(cfg["report_times"].get("monthly_post_hour_local", 23))

    do_weekly  = args.force_weekly  or (now.weekday() == weekly_wd and now.hour == weekly_hr)
    do_monthly = args.force_monthly or (is_last_day_of_month(now.date()) and now.hour == monthly_hr)

    did_anything = False

    # ------------- WEEKLY -------------
    if do_weekly:
        did_anything = True
        end_date = (now.date() - timedelta(days=1))
        start_date = end_date - timedelta(days=6)
        recs = read_hourly_range(hourly_dir, server, start_date, end_date)
        sums = summarize_player(recs, cfg)

        # Sortierung: stärkste Gesamtaktivität (Summe aller Stunden-Summen)
        ordered = sorted(
            sums,
            key=lambda s: (s["active_hours_count"], sum(s["sum24_all"])),
            reverse=True
        )
        topn = ordered if limit <= 0 else ordered[:limit]

        if not topn:
            if args.discord_webhook:
                post_discord(args.discord_webhook, f"Wochenbericht ohne Daten im Zeitraum {start_date}–{end_date}")
        else:
            # globales y-Max für Median- und Summenplots
            y_max_med = global_ymax(topn, ("median24_all","median24_weekday","median24_weekend"))
            y_max_sum = global_ymax(topn, ("sum24_all","sum24_weekday","sum24_weekend"))

            for s in topn:
                # Texte
                msg = build_weekly_message(s, cfg, start_date, end_date)

                # Grafiken (EN)
                median_png = render_grouped_bars(
                    s["median24_all"], s["median24_weekday"], s["median24_weekend"],
                    title_text=f"Weekly Medians by Hour — {s['player']}",
                    ylabel="Median Kills",
                    y_max=y_max_med
                )
                sum_png = render_grouped_bars(
                    s["sum24_all"], s["sum24_weekday"], s["sum24_weekend"],
                    title_text=f"Weekly Sums by Hour — {s['player']}",
                    ylabel="Total Kills",
                    y_max=y_max_sum
                )

                # Dateien sichern
                outdir = os.path.join("reports", "weekly", server, s["player_key"])
                os.makedirs(outdir, exist_ok=True)
                with open(os.path.join(outdir, f"{start_date}_to_{end_date}.txt"), "w", encoding="utf-8") as f:
                    f.write(msg)
                with open(os.path.join(outdir, f"{s['player_key']}_weekly_median.png"), "wb") as f:
                    f.write(median_png.getvalue())
                with open(os.path.join(outdir, f"{s['player_key']}_weekly_sum.png"), "wb") as f:
                    f.write(sum_png.getvalue())

                if args.discord_webhook:
                    files = [
                        (f"{s['player_key']}_weekly_median.png", median_png.getvalue(), "image/png"),
                        (f"{s['player_key']}_weekly_sum.png",    sum_png.getvalue(),    "image/png"),
                    ]
                    post_discord(args.discord_webhook, msg[:max_msg], files=files, max_len=2000)

    # ------------- MONTHLY -------------
    if do_monthly:
        did_anything = True
        last_month_end = (now.replace(day=1) - timedelta(days=1)).date()
        month_start = last_month_end.replace(day=1)
        recs = read_hourly_range(hourly_dir, server, month_start, last_month_end)
        sums = summarize_player(recs, cfg)

        ordered = sorted(
            sums,
            key=lambda s: (s["active_hours_count"], sum(s["sum24_all"])),
            reverse=True
        )
        topn = ordered if limit <= 0 else ordered[:limit]

        if not topn:
            if args.discord_webhook:
                post_discord(args.discord_webhook, f"Monatsbericht ohne Daten im Zeitraum {month_start}–{last_month_end}")
        else:
            y_max_med = global_ymax(topn, ("median24_all","median24_weekday","median24_weekend"))
            y_max_sum = global_ymax(topn, ("sum24_all","sum24_weekday","sum24_weekend"))

            for s in topn:
                msg = build_monthly_message(s, cfg, month_start, last_month_end)

                median_png = render_grouped_bars(
                    s["median24_all"], s["median24_weekday"], s["median24_weekend"],
                    title_text=f"Monthly Medians by Hour — {s['player']}",
                    ylabel="Median Kills",
                    y_max=y_max_med
                )
                sum_png = render_grouped_bars(
                    s["sum24_all"], s["sum24_weekday"], s["sum24_weekend"],
                    title_text=f"Monthly Sums by Hour — {s['player']}",
                    ylabel="Total Kills",
                    y_max=y_max_sum
                )

                outdir = os.path.join("reports", "monthly", server, s["player_key"])
                os.makedirs(outdir, exist_ok=True)
                with open(os.path.join(outdir, f"{month_start.strftime('%Y-%m')}.txt"), "w", encoding="utf-8") as f:
                    f.write(msg)
                with open(os.path.join(outdir, f"{s['player_key']}_monthly_median.png"), "wb") as f:
                    f.write(median_png.getvalue())
                with open(os.path.join(outdir, f"{s['player_key']}_monthly_sum.png"), "wb") as f:
                    f.write(sum_png.getvalue())

                if args.discord_webhook:
                    files = [
                        (f"{s['player_key']}_monthly_median.png", median_png.getvalue(), "image/png"),
                        (f"{s['player_key']}_monthly_sum.png",    sum_png.getvalue(),    "image/png"),
                    ]
                    post_discord(args.discord_webhook, msg[:max_msg], files=files, max_len=2000)

    if not did_anything:
        print("Kein Bericht fällig")


if __name__ == "__main__":
    main()
