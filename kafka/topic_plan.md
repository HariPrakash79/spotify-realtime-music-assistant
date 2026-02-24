# Kafka Topic Plan

This topic plan keeps streaming contracts aligned with staged SQL tables.

## 1) Core Topics

`listen_events_raw`
- Purpose: raw playback events before cleanup.
- Key: `user_id` (string).
- Value (JSON): `source, user_id, track_id, artist_id, event_ts, event_type, track_name, artist_name`.
- Partitions: 6 (start small, scale later).
- Retention: 7 days.

`listen_events_clean`
- Purpose: validated, deduplicated playback events.
- Key: `user_id` (string).
- Value: same schema as `listen_events_raw` with normalized timestamp format.
- Partitions: 6.
- Retention: 14 days.

`track_catalog_upserts`
- Purpose: track metadata updates from FMA/Spotify API.
- Key: `track_id` (string).
- Value (JSON): `source, track_id, artist_id, track_name, artist_name, genre, duration_sec`.
- Partitions: 3.
- Retention: compacted topic (`cleanup.policy=compact`).

`user_features`
- Purpose: rolling user aggregates used by ranking/chat.
- Key: `user_id` (string).
- Value (JSON): `last_event_ts, top_genres, avg_session_gap_sec, updated_at`.
- Partitions: 3.
- Retention: compacted topic.

## 2) Producer Flow

1. `S3 stage listen_events` -> producer -> `listen_events_raw`
2. Stream SQL cleanup (`event_ts` parse, null checks, dedupe) -> `listen_events_clean`
3. Aggregate windows by `user_id` -> `user_features`
4. Catalog sync from staged `track_catalog` -> `track_catalog_upserts`

## 3) SQL Processing (ksqlDB/Flink SQL)

Recommended first transformations:
- Parse timestamp to event-time column.
- Drop rows where `user_id` or `event_ts` is null.
- Deduplicate by `(user_id, track_id, event_ts)` using time-bounded window.
- Build 30-min session windows per user.
- Build 7-day rolling top genres from catalog join.

## 4) Sink to Postgres

`listen_events_clean` -> `music.listen_events`
- Insert-only sink.

`track_catalog_upserts` -> `music.track_catalog`
- Upsert by `(source, track_id)`.

`user_features` -> `music.user_features`
- Upsert by `user_id`.

## 5) Replay Strategy

- Keep at least 7 days in `listen_events_raw`.
- Use staged parquet in S3 as long-term replay source.
- Rebuild downstream topics/tables by replaying stage + raw as needed.
