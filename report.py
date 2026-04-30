#!/usr/bin/env python3
"""report.py — generate the crawl report from analytics.shelve.

Run this at any point during or after crawling:
    python report.py

The output is printed to the terminal AND saved to analytics_report.txt.
"""
from analytics import generate_report
generate_report()
