-- Common recommendation queries for music assistant Postgres.
-- Run these in VS Code SQL tools or any Postgres client.

-- 1) Top active users.
SELECT
    user_id,
    COUNT(*) AS plays
FROM music.listen_events
GROUP BY user_id
ORDER BY plays DESC
LIMIT 20;

-- 2) Personalized recommendations for one user.
-- Replace USER_ID_HERE with a real user id.
SELECT
    user_id,
    recommendation_rank,
    track_name,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_30d
WHERE user_id = 'USER_ID_HERE'
  AND recommendation_rank <= 20
ORDER BY recommendation_rank;

-- 3) Global trending tracks fallback.
SELECT
    global_rank_7d,
    track_name,
    artist_name,
    plays_7d,
    unique_listeners_7d
FROM music.v_global_trending_tracks_7d
WHERE global_rank_7d <= 20
ORDER BY global_rank_7d;
