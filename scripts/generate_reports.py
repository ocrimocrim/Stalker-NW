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
    """Read data/hourly/<server>/<YYYY-MM-DD>.json over a date range and return concatenated 'records'."""
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
    return date(y, m, d).weekday()  # 0=Mon .. 6=Sun

def summarize_player(records, cfg):
    """
    Aggregate per player:
      - per-hour medians: all / weekdays / weekend
      - per-hour 95th percentile (all)
      - per-hour sums: all / weekdays / weekend
      - per-day sums (included days only)
      - counters (days seen, active hours)
    Only days with >= min_daily_kills are considered.
    """
    t = cfg["thresholds"]
    # group by player+date to compute daily totals
    per_day = {}
    for r in records:
        key = (r["player_key"], r["date"])
        per_day.setdefault(key, []).append(r)
    day_ok = {}
    day_sum = {}
    for (pkey, d), lst in per_day.items():
        total = sum(max(0, int(x.get("kills_hour", 0))) for x in lst)
        day_sum[(pkey, d)] = total
        day_ok[(pkey, d)] = (total >= t["min_daily_kills"])

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
            # Median buckets
            "hours_all": [[] for _ in range(24)],
            "hours_weekday": [[] for _ in range(24)],
            "hours_weekend": [[] for _ in range(24)],
            # Sum channels
            "sum_all": [0]*24,
            "sum_weekday": [0]*24,
            "sum_weekend": [0]*24,
            # Per-day sums (only included days)
            "sum_by_day": {},  # date_str -> sum
            # Activity counter
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
        # record per-day sum once
        if r["date"] not in agg["sum_by_day"]:
            agg["sum_by_day"][r["date"]] = day_sum[(pkey, r["date"])]
        if k >= t["normal_ge"]:  # "active" threshold for hour
            agg["hour_count_active"] += 1

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
            # Medians
            "median24_all": med_all,
            "median24_weekday": med_wd,
            "median24_weekend": med_we,
            "p95_24": p95_all,
            # Sums
            "sum24_all": agg["sum_all"],
            "sum24_weekday": agg["sum_weekday"],
            "sum24_weekend": agg["sum_weekend"],
            # Per-day
            "sum_by_day": agg["sum_by_day"],
            # Activity
            "active_hours_count": agg["hour_count_active"],
        })
    return results


# ---------------------------
# Longest active window
# ---------------------------

def longest_active_window(sum_array, threshold):
    """
    sum_array: len=24 (weekly/monthly sums per hour for scope)
    threshold: int; hour is "active" if sum >= threshold
    return: (start_h, end_h, total_sum) inclusive
    tie-breaker: longer window wins; if equal length, higher total_sum wins
    if nothing active: (None, None, 0)
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
    if a[0] is None:
        return b
    if b[2] > a[2]:
        return b
    if b[2] == a[2] and b[3] > a[3]:
        return b
    return a


# ---------------------------
# Plot helpers (ENGLISH, linear axis with gentle compression)
# ---------------------------

def _compress_value(v, linthresh=1000, compress=0.25):
    """Piecewise linear compression: up to linthresh linear, above compressed."""
    if v <= linthresh:
        return float(v)
    return linthresh + (v - linthresh) * compress

def _compress_array(arr, linthresh=1000, compress=0.25):
    return [ _compress_value(float(x), linthresh, compress) for x in arr ]

def _setup_y_ticks(ax, y_max, linthresh=1000, compress=0.25):
    """Set real-number tick labels while plotting compressed heights."""
    if y_max <= 0:
        y_max = 1
    # choose sensible ticks up to y_max
    candidates = [0, 100, 200, 500, 1000, 1500, 2000, 3000, 5000, 8000,
                  10000, 12000, 15000, 20000, 25000, 30000, 40000, 50000]
    ticks = [t for t in candidates if t <= y_max]
    if ticks[-1] < y_max:
        ticks.append(y_max)
    ax.set_yticks([_compress_value(t, linthresh, compress) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_ylim(0, _compress_value(y_max, linthresh, compress) * 1.05)

def render_grouped_bars(all_series, wd_series, we_series, title_text, ylabel,
                        y_max=None, linthresh=1000, compress=0.25):
    """
    Three bars per hour: All / Weekdays / Weekend in one plot.
    Linear axis with gentle compression above 'linthresh';
    tick labels show real kill numbers.
    """
    hours = np.arange(24)
    width = 0.28

    # compress heights for plotting
    all_c = _compress_array(all_series, linthresh, compress)
    wd_c  = _compress_array(wd_series,  linthresh, compress)
    we_c  = _compress_array(we_series,  linthresh, compress)

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.bar(hours - width, all_c, width, label="All")
    ax.bar(hours,         wd_c,  width, label="Weekdays")
    ax.bar(hours + width, we_c,  width, label="Weekend")

    ax.set_title(title_text)
    ax.set_xlabel("Hour")
    ax.set_ylabel(ylabel)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    _setup_y_ticks(ax, y_max if y_max is not None else max(max(all_series), max(wd_series), max(we_series)),
                   linthresh, compress)
    ax.legend(loc="upper right", ncols=3, frameon=False)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf

def render_daily_activity(dates, values, title_text, ylabel,
                          y_max=None, linthresh=1000, compress=0.25):
    """
    Bar chart: one bar per day (dates: list[str], values: list[int]).
    Linear axis with gentle compression; real-number tick labels.
    """
    x = np.arange(len(dates))
    vals_c = _compress_array(values, linthresh, compress)

    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.bar(x, vals_c)
    ax.set_title(title_text)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=0)
    _setup_y_ticks(ax, y_max if y_max is not None else (max(values) if values else 1),
                   linthresh, compress)
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
# Messages (ENGLISH, three windows)
# ---------------------------

def fmt_h(h):
    return f"{h:02d}"

def window_text_en(s, e):
    if s is None:
        return "none"
    if s == e:
        return f"{fmt_h(s)}"
    return f"{fmt_h(s)}–{fmt_h(e)}"

def build_weekly_message_en(s, cfg, start_date, end_date, top_n=3):
    sums_all = s["sum24_all"]
    sums_wd  = s["sum24_weekday"]
    sums_we  = s["sum24_weekend"]
    p95      = s["p95_24"]

    top_hours = sorted(range(24), key=lambda h: sums_all[h], reverse=True)[:top_n]

    thr = cfg["thresholds"]["normal_ge"]
    sA, eA, win_sumA = longest_active_window(sums_all, thr)
    sW, eW, win_sumW = longest_active_window(sums_wd,  thr)
    sE, eE, win_sumE = longest_active_window(sums_we,  thr)

    lines = []
    lines.append(f"Netherworld Weekly Report — Player {s['player']}")
    lines.append(f"Range: {start_date.isoformat()} to {end_date.isoformat()} (Europe/Berlin)")
    lines.append("Summary")
    lines.append(f"Longest active window (All): {window_text_en(sA, eA)} (sum: {win_sumA}).")
    lines.append(f"Longest active window (Weekdays): {window_text_en(sW, eW)} (sum: {win_sumW}).")
    lines.append(f"Longest active window (Weekend): {window_text_en(sE, eE)} (sum: {win_sumE}).")
    lines.append(f"Active days: {s['days_seen']}  |  Hours with activity: {s['active_hours_count']}")
    lines.append("Top hours (by weekly sum)")
    for h in top_hours:
        lines.append(f"{fmt_h(h)}: sum {sums_all[h]} (p95 {p95[h]})")
    return "\n".join(lines)

def build_monthly_message_en(s, cfg, month_start, month_end, top_n=3):
    sums_all = s["sum24_all"]
    sums_wd  = s["sum24_weekday"]
    sums_we  = s["sum24_weekend"]
    p95      = s["p95_24"]

    top_hours = sorted(range(24), key=lambda h: sums_all[h], reverse=True)[:top_n]

    thr = cfg["thresholds"]["normal_ge"]
    sA, eA, win_sumA = longest_active_window(sums_all, thr)
    sW, eW, win_sumW = longest_active_window(sums_wd,  thr)
    sE, eE, win_sumE = longest_active_window(sums_we,  thr)

    lines = []
    lines.append(f"Netherworld Monthly Report — Player {s['player']}")
    lines.append(f"Range: {month_start.isoformat()} to {month_end.isoformat()} (Europe/Berlin)")
    lines.append("Summary")
    lines.append(f"Longest active window (All): {window_text_en(sA, eA)} (sum: {win_sumA}).")
    lines.append(f"Longest active window (Weekdays): {window_text_en(sW, eW)} (sum: {win_sumW}).")
    lines.append(f"Longest active window (Weekend): {window_text_en(sE, eE)} (sum: {win_sumE}).")
    lines.append(f"Active days: {s['days_seen']}  |  Hours with activity: {s['active_hours_count']}")
    lines.append("Top hours (by monthly sum)")
    for h in top_hours:
        lines.append(f"{fmt_h(h)}: sum {sums_all[h]} (p95 {p95[h]})")
    return "\n".join(lines)


# ---------------------------
# Utils
# ---------------------------

def is_last_day_of_month(d):
    return (d.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1) == d

def global_ymax(players, key_triplet):
    """
    Determine global y-axis max across all players and all 3 series of a type.
    key_triplet: ("median24_all","median24_weekday","median24_weekend") or ("sum24_all","sum24_weekday","sum24_weekend")
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

        ordered = sorted(
            sums,
            key=lambda s: (s["active_hours_count"], sum(s["sum24_all"])),
            reverse=True
        )
        topn = ordered if limit <= 0 else ordered[:limit]

        if not topn:
            if args.discord_webhook:
                post_discord(args.discord_webhook, f"Weekly report: no data for {start_date}–{end_date}")
        else:
            # global y-max for hour charts
            y_max_med = global_ymax(topn, ("median24_all","median24_weekday","median24_weekend"))
            y_max_sum = global_ymax(topn, ("sum24_all","sum24_weekday","sum24_weekend"))

            # Precompute per-day vectors & global max for daily chart
            all_dates = [ (start_date + timedelta(days=i)).isoformat() for i in range(7) ]
            for s in topn:
                vec = [ int(s["sum_by_day"].get(d, 0)) for d in all_dates ]
                s["_week_days"] = vec
            y_max_days = max( (max(s["_week_days"]) for s in topn), default=0 )

            for s in topn:
                # English message
                msg = build_weekly_message_en(s, cfg, start_date, end_date)

                # Charts (EN)
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
                days_png = render_daily_activity(
                    dates=[d[5:] for d in all_dates],  # MM-DD
                    values=s["_week_days"],
                    title_text=f"Weekly Activity by Day — {s['player']}",
                    ylabel="Total Kills",
                    y_max=y_max_days
                )

                # Save
                outdir = os.path.join("reports", "weekly", server, s["player_key"])
                os.makedirs(outdir, exist_ok=True)
                with open(os.path.join(outdir, f"{start_date}_to_{end_date}.txt"), "w", encoding="utf-8") as f:
                    f.write(msg)
                with open(os.path.join(outdir, f"{s['player_key']}_weekly_median.png"), "wb") as f:
                    f.write(median_png.getvalue())
                with open(os.path.join(outdir, f"{s['player_key']}_weekly_sum.png"), "wb") as f:
                    f.write(sum_png.getvalue())
                with open(os.path.join(outdir, f"{s['player_key']}_weekly_days.png"), "wb") as f:
                    f.write(days_png.getvalue())

                if args.discord_webhook:
                    files = [
                        (f"{s['player_key']}_weekly_median.png", median_png.getvalue(), "image/png"),
                        (f"{s['player_key']}_weekly_sum.png",    sum_png.getvalue(),    "image/png"),
                        (f"{s['player_key']}_weekly_days.png",   days_png.getvalue(),   "image/png"),
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
                post_discord(args.discord_webhook, f"Monthly report: no data for {month_start}–{last_month_end}")
        else:
            y_max_med = global_ymax(topn, ("median24_all","median24_weekday","median24_weekend"))
            y_max_sum = global_ymax(topn, ("sum24_all","sum24_weekday","sum24_weekend"))

            # monthly: build date sequence for the month
            days = (last_month_end - month_start).days + 1
            all_dates = [ (month_start + timedelta(days=i)).isoformat() for i in range(days) ]
            for s in topn:
                vec = [ int(s["sum_by_day"].get(d, 0)) for d in all_dates ]
                s["_month_days"] = vec
            y_max_days = max( (max(s["_month_days"]) for s in topn), default=0 )

            for s in topn:
                msg = build_monthly_message_en(s, cfg, month_start, last_month_end)

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
                days_png = render_daily_activity(
                    dates=[d[5:] for d in all_dates],  # MM-DD
                    values=s["_month_days"],
                    title_text=f"Monthly Activity by Day — {s['player']}",
                    ylabel="Total Kills",
                    y_max=y_max_days
                )

                outdir = os.path.join("reports", "monthly", server, s["player_key"])
                os.makedirs(outdir, exist_ok=True)
                with open(os.path.join(outdir, f"{month_start.strftime('%Y-%m')}.txt"), "w", encoding="utf-8") as f:
                    f.write(msg)
                with open(os.path.join(outdir, f"{s['player_key']}_monthly_median.png"), "wb") as f:
                    f.write(median_png.getvalue())
                with open(os.path.join(outdir, f"{s['player_key']}_monthly_sum.png"), "wb") as f:
                    f.write(sum_png.getvalue())
                with open(os.path.join(outdir, f"{s['player_key']}_monthly_days.png"), "wb") as f:
                    f.write(days_png.getvalue())

                if args.discord_webhook:
                    files = [
                        (f"{s['player_key']}_monthly_median.png", median_png.getvalue(), "image/png"),
                        (f"{s['player_key']}_monthly_sum.png",    sum_png.getvalue(),    "image/png"),
                        (f"{s['player_key']}_monthly_days.png",   days_png.getvalue(),   "image/png"),
                    ]
                    post_discord(args.discord_webhook, msg[:max_msg], files=files, max_len=2000)

    if not did_anything:
        print("Kein Bericht fällig")


if __name__ == "__main__":
    main()
