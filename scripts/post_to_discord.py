#!/usr/bin/env python3
import argparse, requests, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webhook", required=True)
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    with open(args.file, "r", encoding="utf-8") as f:
        content = f.read()
    r = requests.post(args.webhook, json={"content": f"```\n{content}\n```"}, timeout=20)
    r.raise_for_status()
    print("Gesendet")

if __name__ == "__main__":
    main()
