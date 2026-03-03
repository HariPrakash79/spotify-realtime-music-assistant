#!/usr/bin/env python3
"""
Simple interactive music assistant on top of the recommendation API.

Behavior:
- If user_id is present -> personalized recommendations.
- If requested song is not found -> ask vibe and return closest vibe-based songs.
- Without context -> return trending tracks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from recommendation_client import get_recs, get_trending, get_vibe, search_tracks, submit_feedback


VIBE_OPTIONS = ["chill", "focus", "happy", "sad", "party", "energetic", "romantic"]


def find_user_id(text: str) -> Optional[str]:
    t = text.strip().lower()
    m = re.search(r"\buser_\d+\b", t)
    if m:
        return m.group(0)

    # Accept raw numeric ids like 101617 when user says "my user id is 101617"
    m_num = re.search(r"\b(\d{3,})\b", t)
    if m_num and ("user" in t or "id" in t or "recommend" in t):
        return m_num.group(1)
    return None


def find_vibe(text: str) -> Optional[str]:
    t = text.strip().lower()
    for v in VIBE_OPTIONS:
        if v in t:
            return v
    return None


def extract_song_query(text: str) -> Optional[str]:
    t = text.strip()
    quoted = re.findall(r'"([^"]+)"', t)
    if quoted:
        return quoted[0].strip()

    lower = t.lower()
    markers = ["song ", "track ", "play ", "find "]
    for marker in markers:
        idx = lower.find(marker)
        if idx >= 0:
            q = t[idx + len(marker) :].strip()
            if q:
                return q
    return None


def print_tracks(items: List[Dict[str, Any]], max_rows: int = 10) -> None:
    if not items:
        print("No tracks found.")
        return

    for i, row in enumerate(items[:max_rows], start=1):
        track = row.get("track_name") or row.get("track_id") or "__unknown_track__"
        artist = row.get("artist_name") or "__unknown_artist__"
        score = row.get("recommendation_score")
        if score is not None:
            print(f"{i}. {track} - {artist} (score={score})")
        else:
            print(f"{i}. {track} - {artist}")


def maybe_collect_feedback(default_user_id: Optional[str] = None) -> None:
    ans = input(
        "If any suggested track has wrong vibe, type: feedback <track_id> <correct_vibe> "
        "(or press Enter to skip): "
    ).strip()
    if not ans:
        return
    if not ans.lower().startswith("feedback "):
        print("Skipping feedback (format not recognized).")
        return

    parts = ans.split()
    if len(parts) < 3:
        print("Feedback format: feedback <track_id> <correct_vibe>")
        return
    track_id = parts[1].strip()
    vibe = parts[2].strip().lower()
    if len(parts) >= 4:
        user_id = parts[3].strip()
    else:
        user_id = default_user_id or input("Enter your user_id for feedback tracking: ").strip()
    if not user_id:
        print("Feedback skipped: user_id is required.")
        return
    try:
        res = submit_feedback(user_id=user_id, track_id=track_id, vibe=vibe)
        print("Feedback accepted.")
        ov = res.get("override_update", {})
        if ov:
            print(f"Consensus update: applied={ov.get('applied')} top_share={ov.get('top_share')} users={ov.get('total_unique_users')}")
    except Exception as exc:
        print(f"Feedback failed: {exc}")


def ask_vibe_and_recommend(limit: int = 10) -> None:
    vibe = input(
        "I couldn't find that song clearly. What vibe do you want? "
        "(chill/focus/happy/sad/party/energetic/romantic): "
    ).strip()
    if not vibe:
        vibe = "chill"
    res = get_vibe(vibe=vibe, limit=limit)
    print(f"\nClosest vibe recommendations ({res.get('mode')}):")
    print_tracks(res.get("items", []), max_rows=limit)
    maybe_collect_feedback()


def handle_message(message: str, limit: int = 10) -> None:
    text = message.strip()
    if not text:
        return

    lower = text.lower()
    user_id = find_user_id(text)
    vibe = find_vibe(text)

    # Song request path
    if any(token in lower for token in ["song", "track", "play", "find", "similar"]):
        query = extract_song_query(text)
        if query:
            result = search_tracks(query=query, limit=5)
            items = result.get("items", [])
            if items:
                print(f"\nI found these for '{query}':")
                print_tracks(items, max_rows=5)
                if "similar" in lower or "like" in lower:
                    if not vibe:
                        vibe = input(
                            "What vibe should I target for similar songs? "
                            "(chill/focus/happy/sad/party/energetic/romantic): "
                        ).strip()
                    vibe_res = get_vibe(vibe=vibe or "chill", limit=limit)
                    print(f"\nClosest vibe recommendations ({vibe_res.get('mode')}):")
                    print_tracks(vibe_res.get("items", []), max_rows=limit)
                    maybe_collect_feedback(default_user_id=user_id)
                return

            ask_vibe_and_recommend(limit=limit)
            return

    # Personalized path
    if user_id:
        recs = get_recs(user_id=user_id, limit=limit, fallback_to_trending=True)
        mode = recs.get("mode")
        print(f"\nRecommendations for {user_id} ({mode}):")
        print_tracks(recs.get("items", []), max_rows=limit)
        if mode == "fallback_trending":
            print("Tip: this user has sparse history in dense slice; using trending fallback.")
        maybe_collect_feedback(default_user_id=user_id)
        return

    # Vibe-only path
    if vibe:
        vibe_res = get_vibe(vibe=vibe, limit=limit)
        print(f"\nVibe recommendations ({vibe_res.get('mode')}):")
        print_tracks(vibe_res.get("items", []), max_rows=limit)
        maybe_collect_feedback(default_user_id=user_id)
        return

    # Default path
    trending = get_trending(limit=limit)
    print("\nTrending recommendations:")
    print_tracks(trending.get("items", []), max_rows=limit)
    maybe_collect_feedback(default_user_id=user_id)


def main() -> None:
    print("Music Assistant ready. Type 'exit' to quit.")
    print("Examples:")
    print("- Recommend for user 101617")
    print('- Find song "Morning Child"')
    print("- Give me chill songs")
    print("- Suggest songs similar to Amber Sky")
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if msg.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            return
        try:
            handle_message(msg, limit=10)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
