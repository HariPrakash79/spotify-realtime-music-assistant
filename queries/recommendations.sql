-- Common recommendation queries for music assistant Postgres.
-- Run these in VS Code SQL tools or any Postgres client.

-- 1) Top active users in recommendation-ready slice.
SELECT
    user_id,
    plays
FROM music.v_model_users_1000_ready
ORDER BY plays DESC, user_id
LIMIT 20;

-- 2) Personalized recommendations (Hybrid, latest model) for one user.
-- Replace USER_ID_HERE with a real user id.
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_hybrid_ready
WHERE user_id = 'USER_ID_HERE'
  AND recommendation_rank <= 20
ORDER BY recommendation_rank;

-- 3) Personalized recommendations (MF, latest model) for one user.
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_mf_ready
WHERE user_id = 'USER_ID_HERE'
  AND recommendation_rank <= 20
ORDER BY recommendation_rank;

-- 4) Rule-based dense fallback recommendations for one user.
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_30d_dense_1000_ready
WHERE user_id = 'USER_ID_HERE'
  AND recommendation_rank <= 20
ORDER BY recommendation_rank;

-- 5) Global trending tracks (ready slice).
SELECT
    global_rank_7d,
    track_id,
    track_name,
    artist_id,
    artist_name,
    plays_7d,
    unique_listeners_7d
FROM music.v_global_trending_tracks_7d_ready
WHERE global_rank_7d <= 20
ORDER BY global_rank_7d;

-- 6) Recommendation-ready model metrics.
SELECT
    events,
    users,
    tracks,
    events_per_user
FROM music.v_model_metrics_1000_ready;

-- 7) Display-name map used by API/chat demos.
SELECT
    display_name,
    user_id
FROM music.user_profiles
ORDER BY display_name
LIMIT 100;
