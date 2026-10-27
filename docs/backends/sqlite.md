# SQLite

The default path is `.longcache/v2/cache.db`. SQLite stores vectors, strict JSON response payloads, versioned identity metadata, tags, and a store manifest. WAL mode supports the declared single-process threading model.

A legacy file without format-2 metadata is refused before migration. Use a fresh path and preserve the old store for rollback. Back up live WAL databases with a consistent SQLite backup procedure.

Metadata-only counts/export/maintenance do not need model weights. Normal vector operations validate the persisted embedding manifest before index rebuild or writes.
