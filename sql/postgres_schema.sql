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
    event_hash      TEXT,
    ingestion_ts    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE music.listen_events
    ADD COLUMN IF NOT EXISTS event_hash TEXT;

UPDATE music.listen_events
SET event_hash =
    md5(
        source || chr(31) ||
        user_id || chr(31) ||
        COALESCE(track_id, '') || chr(31) ||
        COALESCE(artist_id, '') || chr(31) ||
        (to_char(event_ts AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US') || '+00:00') || chr(31) ||
        event_type || chr(31) ||
        COALESCE(track_name, '') || chr(31) ||
        COALESCE(artist_name, '')
    )
WHERE event_hash IS NULL;

WITH ranked AS (
    SELECT
        event_id,
        ROW_NUMBER() OVER (PARTITION BY event_hash ORDER BY event_id) AS rn
    FROM music.listen_events
    WHERE event_hash IS NOT NULL
)
DELETE FROM music.listen_events t
USING ranked r
WHERE t.event_id = r.event_id
  AND r.rn > 1;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_listen_events_event_hash'
          AND connamespace = 'music'::regnamespace
    ) THEN
        ALTER TABLE music.listen_events
            ADD CONSTRAINT uq_listen_events_event_hash UNIQUE (event_hash);
    END IF;
END $$;

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
WITH anchor AS (
    SELECT MAX(event_ts) AS max_event_ts
    FROM music.listen_events
)
SELECT
    user_id,
    COUNT(*) AS plays_7d,
    MAX(event_ts) AS last_play_ts
FROM music.listen_events le
CROSS JOIN anchor a
WHERE a.max_event_ts IS NOT NULL
  AND le.event_ts >= a.max_event_ts - INTERVAL '7 days'
GROUP BY user_id;

CREATE OR REPLACE VIEW music.v_user_top_tracks_30d AS
WITH anchor AS (
    SELECT MAX(event_ts) AS max_event_ts
    FROM music.listen_events
),
agg AS (
    SELECT
        user_id,
        COALESCE(track_id, '__unknown__') AS track_id_key,
        MAX(track_id) AS track_id,
        COALESCE(MAX(NULLIF(track_name, '')), MAX(track_id), '__unknown_track__') AS track_name,
        MAX(artist_id) AS artist_id,
        COALESCE(MAX(NULLIF(artist_name, '')), '__unknown_artist__') AS artist_name,
        COUNT(*)::BIGINT AS plays_30d,
        MAX(event_ts) AS last_play_ts
    FROM music.listen_events le
    CROSS JOIN anchor a
    WHERE a.max_event_ts IS NOT NULL
      AND le.event_ts >= a.max_event_ts - INTERVAL '30 days'
      AND le.event_type = 'play'
    GROUP BY user_id, COALESCE(track_id, '__unknown__')
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY plays_30d DESC, last_play_ts DESC, track_id_key
        ) AS rank_in_user
    FROM agg
)
SELECT
    user_id,
    rank_in_user,
    track_id,
    track_name,
    artist_id,
    artist_name,
    plays_30d,
    last_play_ts
FROM ranked;

CREATE OR REPLACE VIEW music.v_user_top_artists_30d AS
WITH anchor AS (
    SELECT MAX(event_ts) AS max_event_ts
    FROM music.listen_events
),
agg AS (
    SELECT
        user_id,
        COALESCE(artist_id, '__unknown__') AS artist_id_key,
        MAX(artist_id) AS artist_id,
        COALESCE(MAX(NULLIF(artist_name, '')), '__unknown_artist__') AS artist_name,
        COUNT(*)::BIGINT AS plays_30d,
        COUNT(DISTINCT COALESCE(track_id, track_name, '__unknown_track__'))::BIGINT AS unique_tracks_30d,
        MAX(event_ts) AS last_play_ts
    FROM music.listen_events le
    CROSS JOIN anchor a
    WHERE a.max_event_ts IS NOT NULL
      AND le.event_ts >= a.max_event_ts - INTERVAL '30 days'
      AND le.event_type = 'play'
    GROUP BY user_id, COALESCE(artist_id, '__unknown__')
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY plays_30d DESC, unique_tracks_30d DESC, last_play_ts DESC, artist_id_key
        ) AS rank_in_user
    FROM agg
)
SELECT
    user_id,
    rank_in_user,
    artist_id,
    artist_name,
    plays_30d,
    unique_tracks_30d,
    last_play_ts
FROM ranked;

CREATE OR REPLACE VIEW music.v_track_popularity_7d AS
WITH anchor AS (
    SELECT MAX(event_ts) AS max_event_ts
    FROM music.listen_events
)
SELECT
    COALESCE(track_id, '__unknown__') AS track_id_key,
    MAX(track_id) AS track_id,
    COALESCE(MAX(NULLIF(track_name, '')), MAX(track_id), '__unknown_track__') AS track_name,
    MAX(artist_id) AS artist_id,
    COALESCE(MAX(NULLIF(artist_name, '')), '__unknown_artist__') AS artist_name,
    COUNT(*)::BIGINT AS plays_7d,
    COUNT(DISTINCT user_id)::BIGINT AS unique_listeners_7d,
    MAX(event_ts) AS last_play_ts
FROM music.listen_events le
CROSS JOIN anchor a
WHERE a.max_event_ts IS NOT NULL
  AND le.event_ts >= a.max_event_ts - INTERVAL '7 days'
  AND le.event_type = 'play'
GROUP BY COALESCE(track_id, '__unknown__');

CREATE OR REPLACE VIEW music.v_global_trending_tracks_7d AS
WITH ranked AS (
    SELECT
        track_id,
        track_name,
        artist_id,
        artist_name,
        plays_7d,
        unique_listeners_7d,
        last_play_ts,
        ROW_NUMBER() OVER (
            ORDER BY plays_7d DESC, unique_listeners_7d DESC, last_play_ts DESC, track_id_key
        ) AS global_rank_7d
    FROM music.v_track_popularity_7d
    WHERE track_id IS NOT NULL
)
SELECT
    global_rank_7d,
    track_id,
    track_name,
    artist_id,
    artist_name,
    plays_7d,
    unique_listeners_7d,
    last_play_ts
FROM ranked;

CREATE OR REPLACE VIEW music.v_user_recommendations_30d AS
WITH anchor AS (
    SELECT MAX(event_ts) AS max_event_ts
    FROM music.listen_events
),
active_users AS (
    SELECT DISTINCT
        le.user_id
    FROM music.listen_events le
    CROSS JOIN anchor a
    WHERE a.max_event_ts IS NOT NULL
      AND le.event_ts >= a.max_event_ts - INTERVAL '30 days'
      AND le.event_type = 'play'
),
user_played AS (
    SELECT DISTINCT
        user_id,
        COALESCE(track_id, '__unknown__') AS track_id_key
    FROM music.listen_events le
    CROSS JOIN anchor a
    WHERE a.max_event_ts IS NOT NULL
      AND le.event_ts >= a.max_event_ts - INTERVAL '30 days'
      AND le.event_type = 'play'
),
user_top_artists AS (
    SELECT
        user_id,
        artist_id,
        rank_in_user
    FROM music.v_user_top_artists_30d
    WHERE rank_in_user <= 5
      AND artist_id IS NOT NULL
),
artist_track_pop AS (
    SELECT
        COALESCE(artist_id, '__unknown__') AS artist_id_key,
        COALESCE(track_id, '__unknown__') AS track_id_key,
        MAX(track_id) AS track_id,
        COALESCE(MAX(NULLIF(track_name, '')), MAX(track_id), '__unknown_track__') AS track_name,
        MAX(artist_id) AS artist_id,
        COALESCE(MAX(NULLIF(artist_name, '')), '__unknown_artist__') AS artist_name,
        COUNT(*)::BIGINT AS plays_30d
    FROM music.listen_events le
    CROSS JOIN anchor a
    WHERE a.max_event_ts IS NOT NULL
      AND le.event_ts >= a.max_event_ts - INTERVAL '30 days'
      AND le.event_type = 'play'
    GROUP BY
        COALESCE(artist_id, '__unknown__'),
        COALESCE(track_id, '__unknown__')
),
candidates AS (
    SELECT
        uta.user_id,
        atp.track_id,
        atp.track_name,
        atp.artist_id,
        atp.artist_name,
        uta.rank_in_user AS artist_affinity_rank,
        atp.plays_30d AS track_popularity_30d,
        (atp.plays_30d::NUMERIC * (1.0 / uta.rank_in_user)) AS recommendation_score
    FROM user_top_artists uta
    JOIN artist_track_pop atp
      ON COALESCE(uta.artist_id, '__unknown__') = atp.artist_id_key
    LEFT JOIN user_played up
      ON up.user_id = uta.user_id
     AND up.track_id_key = atp.track_id_key
    WHERE up.user_id IS NULL
      AND atp.track_id IS NOT NULL
),
dedup AS (
    SELECT
        user_id,
        track_id,
        COALESCE(track_id, '__unknown__') AS track_id_key,
        MAX(track_name) AS track_name,
        MAX(artist_id) AS artist_id,
        MAX(artist_name) AS artist_name,
        MIN(artist_affinity_rank) AS artist_affinity_rank,
        MAX(track_popularity_30d) AS track_popularity_30d,
        MAX(recommendation_score) AS recommendation_score
    FROM candidates
    GROUP BY user_id, track_id
),
ranked_personalized AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY recommendation_score DESC, track_popularity_30d DESC, track_id
        ) AS recommendation_rank
    FROM dedup
),
personalized_counts AS (
    SELECT
        user_id,
        COUNT(*)::BIGINT AS personalized_count
    FROM ranked_personalized
    GROUP BY user_id
),
fallback_candidates AS (
    SELECT
        u.user_id,
        gt.track_id,
        gt.track_name,
        gt.artist_id,
        gt.artist_name,
        NULL::BIGINT AS artist_affinity_rank,
        gt.plays_7d::BIGINT AS track_popularity_30d,
        ((gt.plays_7d::NUMERIC * 0.001) + (1.0 / (1000 + gt.global_rank_7d))) AS recommendation_score,
        gt.global_rank_7d
    FROM active_users u
    JOIN music.v_global_trending_tracks_7d gt
      ON TRUE
    LEFT JOIN user_played up
      ON up.user_id = u.user_id
     AND up.track_id_key = COALESCE(gt.track_id, '__unknown__')
    LEFT JOIN dedup d
      ON d.user_id = u.user_id
     AND d.track_id_key = COALESCE(gt.track_id, '__unknown__')
    WHERE up.user_id IS NULL
      AND d.user_id IS NULL
      AND gt.track_id IS NOT NULL
),
ranked_fallback AS (
    SELECT
        fc.user_id,
        (COALESCE(pc.personalized_count, 0) + ROW_NUMBER() OVER (
            PARTITION BY fc.user_id
            ORDER BY fc.global_rank_7d, fc.track_id
        )) AS recommendation_rank,
        fc.track_id,
        fc.track_name,
        fc.artist_id,
        fc.artist_name,
        fc.artist_affinity_rank,
        fc.track_popularity_30d,
        fc.recommendation_score
    FROM fallback_candidates fc
    LEFT JOIN personalized_counts pc
      ON pc.user_id = fc.user_id
)
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_id,
    artist_name,
    artist_affinity_rank,
    track_popularity_30d,
    recommendation_score
FROM ranked_personalized
UNION ALL
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_id,
    artist_name,
    artist_affinity_rank,
    track_popularity_30d,
    recommendation_score
FROM ranked_fallback;
