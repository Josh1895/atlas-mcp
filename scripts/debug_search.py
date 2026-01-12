#!/usr/bin/env python3
"""Debug script to check raw search results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ddgs import DDGS

# Test raw DDG search
query = "python time.monotonic vs time.time best practices"
print(f"Query: {query}")
print("=" * 70)

results = list(DDGS().text(query, max_results=10))
print(f"\nTotal raw results: {len(results)}")

total_content = 0
for i, r in enumerate(results):
    url = r.get("href", "no url")
    body = r.get("body", "")
    total_content += len(body)
    print(f"\n[{i+1}] {url[:70]}")
    print(f"    Body ({len(body)} chars): {body[:150]}...")

print(f"\n{'=' * 70}")
print(f"Total content from {len(results)} results: {total_content} chars")

# Check which would pass trusted domains filter (expanded list)
trusted = [
    "stackoverflow.com", "stackexchange.com", "github.com", "gist.github.com",
    "docs.python.org", "python.org", "developer.mozilla.org",
    "readthedocs.io", "pypi.org", "medium.com", "dev.to", "realpython.com",
    "geeksforgeeks.org", "tutorialspoint.com", "w3schools.com", "freecodecamp.org",
    "towardsdatascience.com", "hackernoon.com", "superfastpython.com",
    "pythontutorial.net", "pytutorial.com", "zetcode.com", "programiz.com",
    "digitalocean.com", "linuxize.com", "bomberbot.com", "runebook.dev", "readmedium.com",
    "nedbatchelder.com", "mail.python.org",  # Python mailing lists
]

filtered = [r for r in results if any(d in r.get("href", "") for d in trusted)]
print(f"\nAfter trusted domain filter: {len(filtered)} results")
filtered_content = sum(len(r.get("body", "")) for r in filtered)
print(f"Filtered content: {filtered_content} chars")
