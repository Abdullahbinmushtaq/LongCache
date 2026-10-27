# CLI Reference

LongCache provides a command-line utility for cache administration, real-time metrics inspection, and maintenance.

```bash
longcache --help
```

---

## 1. `longcache stats`

Displays cache efficiency metrics: hit rate, total queries, estimated tokens saved, and financial cost savings.

```bash
longcache stats
```

### Options

- `--namespace TEXT`: Target namespace (defaults to `default`).
- `--db-path TEXT`: Path to SQLite database file (defaults to `.longcache/cache.db`).

### Sample Output

```text
============================================================
LongCache Statistics [Namespace: default]
============================================================
Total Requests:        14,250
Cache Hits:             8,970 (62.9%)
Cache Misses:           5,280 (37.1%)
Stored Vector Entries:  5,280
Estimated Tokens Saved: 4,485,000 tokens
Estimated Cost Saved:  ~$13.45 USD (based on GPT-4o pricing)
Avg Lookup Latency:     3.8 ms
============================================================
```

---

## 2. `longcache clear`

Evicts cache entries.

```bash
# Clear a specific namespace
longcache clear --namespace customer_support

# Clear all entries across all namespaces
longcache clear --all
```

---

## 3. `longcache inspect`

Looks up a query in the cache and displays the closest matching cached vectors along with their cosine similarity distance.

```bash
longcache inspect --query "how can I return an item?"
```

### Sample Output

```text
Query: "how can I return an item?"
Found 1 matching entry:
------------------------------------------------------------
ID: 42
Similarity: 0.9482 [MATCH >= 0.92]
Original Query: "What is your refund policy?"
Cached At: 2026-10-25 14:10:02 UTC
Response Preview: "Our refund policy allows returns within 30 days..."
------------------------------------------------------------
```

---

## 4. `longcache export`

Exports cached entries to JSON or CSV format for dataset analysis or cache migration.

```bash
# Export to JSON
longcache export --format json --out export.json

# Export to CSV
longcache export --format csv --out export.csv
```
