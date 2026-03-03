#!/usr/bin/env python3
"""
Client helper for the local recommendation API.

Examples:
  python scripts/recommendation_client.py --query metrics
  python scripts/recommendation_client.py --query trending --limit 10
  python scripts/recommendation_client.py --query recs --user-id 101617 --limit 20
  python scripts/recommendation_client.py --query favorites --user-id 101617 --limit 20
  python scripts/recommendation_client.py --query search --text "Morning Child" --limit 5
  python scripts/recommendation_client.py --query vibe --text chill --limit 10
  python scripts/recommendation_client.py --query feedback --user-id 101617 --track-id 45659 --vibe energetic --predicted-vibe chill
  python scripts/recommendation_client.py --query interaction --user-id 101617 --track-id 45659 --action like --source-endpoint /recs
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import requests


def api_base_url() -> str:
    return os.environ.get("RECOMMENDATION_API_BASE_URL", "http://localhost:8000").rstrip("/")


def _request(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{api_base_url()}{path}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{api_base_url()}{path}"
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_model_metrics() -> Dict[str, Any]:
    return _request("/metrics/model")


def get_trending(limit: int = 20, user_id: str | None = None, session_id: str | None = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": limit}
    if user_id:
        params["user_id"] = user_id
    if session_id:
        params["session_id"] = session_id
    return _request("/trending", params=params)


def get_recs(
    user_id: str,
    limit: int = 20,
    fallback_to_trending: bool = True,
    session_id: str | None = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "limit": limit,
        "fallback_to_trending": str(fallback_to_trending).lower(),
    }
    if session_id:
        params["session_id"] = session_id
    return _request(
        f"/recs/{user_id}",
        params=params,
    )


def get_user_favorites(
    user_id: str,
    limit: int = 20,
    fallback_to_recs: bool = True,
    session_id: str | None = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "limit": limit,
        "fallback_to_recs": str(fallback_to_recs).lower(),
    }
    if session_id:
        params["session_id"] = session_id
    return _request(
        f"/favorites/{user_id}",
        params=params,
    )


def search_tracks(query: str, limit: int = 10) -> Dict[str, Any]:
    return _request("/search/tracks", params={"query": query, "limit": limit})


def get_vibe(
    vibe: str,
    limit: int = 10,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"vibe": vibe, "limit": limit}
    if user_id:
        params["user_id"] = user_id
    if session_id:
        params["session_id"] = session_id
    return _request("/vibe", params=params)


def submit_feedback(
    user_id: str,
    track_id: str,
    vibe: str,
    predicted_vibe: str | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "track_id": track_id,
        "user_selected_vibe": vibe,
    }
    if predicted_vibe:
        payload["predicted_vibe"] = predicted_vibe
    return _post("/feedback/vibe", payload=payload)


def submit_interaction_feedback(
    user_id: str,
    action: str,
    track_id: str | None = None,
    source_endpoint: str | None = None,
    context_vibe: str | None = None,
    recommendation_rank: int | None = None,
    session_id: str | None = None,
    signal_strength: float | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "action": action,
    }
    if track_id:
        payload["track_id"] = track_id
    if source_endpoint:
        payload["source_endpoint"] = source_endpoint
    if context_vibe:
        payload["context_vibe"] = context_vibe
    if recommendation_rank is not None:
        payload["recommendation_rank"] = recommendation_rank
    if session_id:
        payload["session_id"] = session_id
    if signal_strength is not None:
        payload["signal_strength"] = signal_strength
    return _post("/feedback/interaction", payload=payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the recommendation API.")
    parser.add_argument(
        "--query",
        choices=["metrics", "trending", "recs", "favorites", "search", "vibe", "feedback", "interaction"],
        required=True,
        help="Which API endpoint to call.",
    )
    parser.add_argument("--user-id", default=None, help="User id for --query recs.")
    parser.add_argument("--track-id", default=None, help="Track id for --query feedback.")
    parser.add_argument("--action", default=None, help="Interaction action for --query interaction.")
    parser.add_argument("--source-endpoint", default=None, help="Source endpoint for --query interaction.")
    parser.add_argument("--session-id", default=None, help="Session id for --query interaction.")
    parser.add_argument("--text", default=None, help="Text input for --query search or --query vibe.")
    parser.add_argument("--vibe", default=None, help="Vibe label for --query feedback.")
    parser.add_argument("--predicted-vibe", default=None, help="Optional predicted vibe for --query feedback.")
    parser.add_argument("--rank", type=int, default=None, help="Recommendation rank for --query interaction.")
    parser.add_argument("--signal-strength", type=float, default=None, help="Optional signal strength for --query interaction.")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to request.")
    parser.add_argument(
        "--fallback-to-trending",
        action="store_true",
        help="For --query recs, allow trending fallback when personalized recs are missing.",
    )
    args = parser.parse_args()

    if args.query == "metrics":
        data = get_model_metrics()
    elif args.query == "trending":
        data = get_trending(limit=args.limit, user_id=args.user_id, session_id=args.session_id)
    elif args.query == "recs":
        if not args.user_id:
            raise ValueError("--user-id is required when --query recs is used.")
        data = get_recs(
            user_id=args.user_id,
            limit=args.limit,
            fallback_to_trending=args.fallback_to_trending,
            session_id=args.session_id,
        )
    elif args.query == "favorites":
        if not args.user_id:
            raise ValueError("--user-id is required when --query favorites is used.")
        data = get_user_favorites(
            user_id=args.user_id,
            limit=args.limit,
            fallback_to_recs=True,
            session_id=args.session_id,
        )
    elif args.query == "search":
        if not args.text:
            raise ValueError("--text is required when --query search is used.")
        data = search_tracks(query=args.text, limit=args.limit)
    elif args.query == "vibe":
        if not args.text:
            raise ValueError("--text is required when --query vibe is used.")
        data = get_vibe(vibe=args.text, limit=args.limit, user_id=args.user_id, session_id=args.session_id)
    elif args.query == "feedback":
        if not args.user_id:
            raise ValueError("--user-id is required when --query feedback is used.")
        if not args.track_id:
            raise ValueError("--track-id is required when --query feedback is used.")
        if not args.vibe:
            raise ValueError("--vibe is required when --query feedback is used.")
        data = submit_feedback(
            user_id=args.user_id,
            track_id=args.track_id,
            vibe=args.vibe,
            predicted_vibe=args.predicted_vibe,
        )
    else:
        if not args.user_id:
            raise ValueError("--user-id is required when --query interaction is used.")
        if not args.action:
            raise ValueError("--action is required when --query interaction is used.")
        data = submit_interaction_feedback(
            user_id=args.user_id,
            action=args.action,
            track_id=args.track_id,
            source_endpoint=args.source_endpoint,
            context_vibe=args.vibe,
            recommendation_rank=args.rank,
            session_id=args.session_id,
            signal_strength=args.signal_strength,
        )

    print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
