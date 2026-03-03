#!/usr/bin/env python3
"""
Client helper for the local recommendation API.

Examples:
  python scripts/recommendation_client.py --query metrics
  python scripts/recommendation_client.py --query trending --limit 10
  python scripts/recommendation_client.py --query recs --user-id 101617 --limit 20
  python scripts/recommendation_client.py --query search --text "Morning Child" --limit 5
  python scripts/recommendation_client.py --query vibe --text chill --limit 10
  python scripts/recommendation_client.py --query feedback --user-id 101617 --track-id 45659 --vibe energetic --predicted-vibe chill
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


def get_trending(limit: int = 20) -> Dict[str, Any]:
    return _request("/trending", params={"limit": limit})


def get_recs(user_id: str, limit: int = 20, fallback_to_trending: bool = True) -> Dict[str, Any]:
    return _request(
        f"/recs/{user_id}",
        params={"limit": limit, "fallback_to_trending": str(fallback_to_trending).lower()},
    )


def search_tracks(query: str, limit: int = 10) -> Dict[str, Any]:
    return _request("/search/tracks", params={"query": query, "limit": limit})


def get_vibe(vibe: str, limit: int = 10) -> Dict[str, Any]:
    return _request("/vibe", params={"vibe": vibe, "limit": limit})


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the recommendation API.")
    parser.add_argument(
        "--query",
        choices=["metrics", "trending", "recs", "search", "vibe", "feedback"],
        required=True,
        help="Which API endpoint to call.",
    )
    parser.add_argument("--user-id", default=None, help="User id for --query recs.")
    parser.add_argument("--track-id", default=None, help="Track id for --query feedback.")
    parser.add_argument("--text", default=None, help="Text input for --query search or --query vibe.")
    parser.add_argument("--vibe", default=None, help="Vibe label for --query feedback.")
    parser.add_argument("--predicted-vibe", default=None, help="Optional predicted vibe for --query feedback.")
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
        data = get_trending(limit=args.limit)
    elif args.query == "recs":
        if not args.user_id:
            raise ValueError("--user-id is required when --query recs is used.")
        data = get_recs(
            user_id=args.user_id,
            limit=args.limit,
            fallback_to_trending=args.fallback_to_trending,
        )
    elif args.query == "search":
        if not args.text:
            raise ValueError("--text is required when --query search is used.")
        data = search_tracks(query=args.text, limit=args.limit)
    elif args.query == "vibe":
        if not args.text:
            raise ValueError("--text is required when --query vibe is used.")
        data = get_vibe(vibe=args.text, limit=args.limit)
    else:
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

    print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
