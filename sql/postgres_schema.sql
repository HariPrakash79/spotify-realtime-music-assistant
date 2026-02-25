-- Postgres serving schema for the music assistant project.
-- Use this in RDS/Postgres after staging parquet is validated.

CREATE SCHEMA IF NOT EXISTS music;

CREATE TABLE IF NOT EXISTS music.track_catalog (
    source          TEXT NOT NULL,
    track_id        TEXT NOT NULL,
    artist_id       TEXT,
    track_name      TEXT,
    artist_name     TEXT,
    genre           TEXT,
    duration_sec    REAL,
    inserted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source, track_id)
);

CREATE INDEX IF NOT EXISTS idx_track_catalog_artist_id
    ON music.track_catalog (artist_id);

CREATE INDEX IF NOT EXISTS idx_track_catalog_genre
    ON music.track_catalog (genre);

CREATE TABLE IF NOT EXISTS music.listen_events (
    event_id        BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    track_id        TEXT,
    artist_id       TEXT,
    event_ts        TIMESTAMPTZ NOT NULL,
    event_type      TEXT NOT NULL DEFAULT 'play',
    track_name      TEXT,
    artist_name     TEXT,
    ingestion_ts    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_listen_events_user_ts
    ON music.listen_events (user_id, event_ts DESC);

CREATE INDEX IF NOT EXISTS idx_listen_events_track_ts
    ON music.listen_events (track_id, event_ts DESC);

CREATE INDEX IF NOT EXISTS idx_listen_events_source_ts
    ON music.listen_events (source, event_ts DESC);

CREATE TABLE IF NOT EXISTS music.user_features (
    user_id             TEXT PRIMARY KEY,
    last_event_ts       TIMESTAMPTZ,
    top_genres          TEXT[],
    avg_session_gap_sec REAL,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE VIEW music.v_user_recent_activity AS
SELECT
    user_id,
    COUNT(*) AS plays_7d,
    MAX(event_ts) AS last_play_ts
FROM music.listen_events
WHERE event_ts >= NOW() - INTERVAL '7 days'
GROUP BY user_id;
