#!/usr/bin/env python3
"""
LangChain + Bedrock music assistant grounded on the local recommendation API.

Usage:
  python scripts/langchain_bedrock_assistant.py

Prerequisites:
  - API server running: python scripts/recommendation_api.py
  - Bedrock model access enabled in your AWS account/region.

Optional env:
  AWS_PROFILE=spotify
  AWS_REGION=us-east-1
  BEDROCK_REGION=us-east-1
  BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Set

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import recommendation_client as rc


SYSTEM_PROMPT = """
You are a music recommendation assistant for a real data platform.
Always use tools for recommendations/search/vibe/metrics and do not invent tracks.

Rules:
1) Personalized ask -> use get_recs.
1a) Favorites ask (e.g., "<name>'s favorites") -> use get_user_favorites.
2) Song/title ask -> use search_tracks first.
3) If user asks by mood/vibe or song not found -> use get_vibe.
4) Global/hot ask -> use get_trending.
5) Keep answers concise and practical.
6) If data is missing/fallback, explain why and give the next best option.
7) Never output <thinking> tags, XML tags, or hidden reasoning. Respond in plain text only.
""".strip()


VIBE_OPTIONS = ["chill", "focus", "happy", "sad", "party", "energetic", "romantic"]
FALLBACK_CANCEL_TOKENS = {"cancel", "skip", "never mind", "nevermind", "stop"}
USER_PRONOUN_TOKENS = [
    "for him",
    "for her",
    "for them",
    "for his",
    "for their",
]
GENRE_QUERY_HINTS = {"genre", "genres", "type", "types", "category", "categories"}


def _to_json(data: Any) -> str:
    return json.dumps(data, default=str)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _clean_model_text(text: str) -> str:
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?response>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?thinking>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_user_id(text: str) -> str | None:
    lower = text.lower()
    m = re.search(r"\buser[_\s-]?(\d+)\b", lower)
    if m:
        return f"user_{int(m.group(1)):06d}"

    m2 = re.search(r"\b(\d{3,})\b", lower)
    if m2 and ("user" in lower or "recommend" in lower or "recs" in lower):
        raw = m2.group(1)
        # Keep numeric user IDs as-is for sources where user_id is numeric.
        return raw
    return None


def _clean_candidate_user_ref(raw: str) -> str | None:
    candidate = raw.strip().strip("\"'.,!?;:")
    # Trim conversational fillers often appended in follow-ups: "for abi then".
    while True:
        trimmed = re.sub(
            r"\b(?:please|pls|then|now|ok|okay|also|too)\b$",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if trimmed == candidate:
            break
        candidate = trimmed
    if not candidate:
        return None

    lower = candidate.lower()
    if lower.startswith("for "):
        return None
    if " for " in lower:
        return None
    blocked = {
        "me",
        "my",
        "him",
        "her",
        "them",
        "his",
        "their",
        "us",
        "everyone",
        "all",
        "song",
        "songs",
        "track",
        "tracks",
        "music",
        "playlist",
    }
    if lower in blocked:
        return None
    if len(candidate) > 80:
        return None
    if any(v in lower for v in VIBE_OPTIONS) and len(lower.split()) <= 2:
        return None
    return candidate


def _extract_user_reference(text: str) -> str | None:
    user_id = _extract_user_id(text)
    if user_id:
        return user_id

    patterns = [
        r"\bfor\s+([A-Za-z][A-Za-z .'\-]{1,80})$",
        r"\b(?:recommend|recs?|suggest|give)\b.*?\bfor\s+([A-Za-z][A-Za-z .'\-]{1,80})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_candidate_user_ref(match.group(1))
        if candidate:
            return candidate
    return None


def _extract_standalone_user_candidate(text: str) -> str | None:
    # For follow-up turns like: "Aarav" after "party songs for him".
    cleaned = text.strip().strip("\"'")
    if not cleaned:
        return None
    if len(cleaned) > 80:
        return None
    lower = cleaned.lower()
    if lower.endswith("?"):
        return None
    if re.search(r"\d", cleaned):
        return None
    if any(tok in lower for tok in ["song", "songs", "track", "tracks", "recommend", "vibe", "trending"]):
        return None
    if any(tok in lower for tok in GENRE_QUERY_HINTS):
        return None
    if lower.startswith(("what ", "which ", "how ", "who ", "where ", "when ", "why ")):
        return None
    if len(lower.split()) > 3:
        return None
    return _clean_candidate_user_ref(cleaned)


def _extract_vibe(text: str) -> str | None:
    lower = text.lower()
    for vibe in VIBE_OPTIONS:
        if vibe in lower:
            return vibe
    if "energy" in lower or "energetic" in lower:
        return "energetic"
    if "party" in lower:
        return "party"
    return None


def _extract_favorites_user_ref(text: str) -> str | None:
    patterns = [
        r"\b([A-Za-z][A-Za-z .'\-]{1,80})'?s\s+favo(?:u)?rites?\b",
        r"\bfavo(?:u)?rites?\s+for\s+([A-Za-z][A-Za-z .'\-]{1,80})\b",
        r"\b([A-Za-z][A-Za-z .'\-]{1,80})\s+favo(?:u)?rites?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        cleaned = _clean_candidate_user_ref(match.group(1))
        if cleaned:
            return cleaned
    return None


def _route_tool(text: str) -> tuple[str, Dict[str, Any]] | None:
    lower = text.lower()
    user_ref = _extract_user_reference(text)
    favorites_user_ref = _extract_favorites_user_ref(text)
    vibe = _extract_vibe(text)

    if favorites_user_ref is not None:
        return ("get_user_favorites", {"user_id": favorites_user_ref, "limit": 20, "fallback_to_recs": True})

    if user_ref and any(t in lower for t in ["recommend", "recs", "suggest"]):
        return ("get_recs", {"user_id": user_ref, "limit": 20, "fallback_to_trending": True})

    if any(t in lower for t in ["trending", "popular", "hot", "top songs", "top tracks"]):
        return ("get_trending", {"limit": 10})

    if vibe is not None:
        return ("get_vibe", {"vibe": vibe, "limit": 10})

    # Route "give songs for <name>" style prompts to personalized recommendations.
    if user_ref and any(t in lower for t in ["song", "songs", "track", "tracks", "music", "give"]):
        return ("get_recs", {"user_id": user_ref, "limit": 20, "fallback_to_trending": True})

    m = re.search(r"(?:find|search|song|track)\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        query = m.group(1).strip().strip('"')
        if query:
            return ("search_tracks", {"query": query, "limit": 5})

    return None


def _extract_requested_count(text: str, default: int = 10) -> int:
    lower = text.lower()
    m = re.search(r"\b(\d{1,3})\b", lower)
    if m and any(tok in lower for tok in ["song", "songs", "track", "tracks", "more", "recommend", "give"]):
        n = int(m.group(1))
        return max(1, min(n, 100))
    if "few" in lower:
        return 5
    return default


def _is_more_request(text: str) -> bool:
    lower = text.lower()
    return any(tok in lower for tok in [" more", "more ", "another", "next"])


def _is_implicit_count_followup(text: str) -> bool:
    """
    Detect short continuation asks like:
    - give 10
    - need 5
    - show 20
    """
    lower = text.lower().strip()
    if not re.search(r"\b\d{1,3}\b", lower):
        return False
    if not any(tok in lower for tok in ["give", "need", "show"]):
        return False
    # If user states a fresh explicit intent, this is not just a continuation.
    if any(tok in lower for tok in ["trending", "popular", "hot", "recommend", "recs", "suggest", "find", "search"]):
        return False
    return True


def _is_genre_capability_question(text: str) -> bool:
    lower = text.lower()
    if not any(hint in lower for hint in GENRE_QUERY_HINTS):
        return False
    if any(token in lower for token in ["what", "which", "can", "available", "give", "support"]):
        return True
    if " for " in f" {lower} ":
        return True
    if any(token in lower for token in ["specific", "specifically", "personalize", "personalized"]):
        return True
    return False


def _extract_user_refs_for_capability_question(text: str, fallback_user_ref: str | None = None) -> List[str]:
    refs: List[str] = []
    explicit = _extract_user_reference(text)
    if explicit:
        explicit_l = explicit.lower()
        if not any(sep in explicit_l for sep in [" and ", ",", " & "]):
            refs.append(explicit)

    match = re.search(r"\bfor\s+(.+)$", text, flags=re.IGNORECASE)
    if match:
        tail = match.group(1).strip().strip(".!?")
        parts = re.split(r"\s*(?:,| and | & )\s*", tail, flags=re.IGNORECASE)
        for part in parts:
            cleaned = _clean_candidate_user_ref(part)
            if cleaned:
                refs.append(cleaned)

    if fallback_user_ref and _references_last_user(text):
        refs.append(fallback_user_ref)

    deduped: List[str] = []
    seen: Set[str] = set()
    for ref in refs:
        key = ref.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _build_user_vibe_coverage_summary(user_refs: List[str]) -> str:
    if not user_refs:
        return _genre_capability_response()

    vibe_key_cache: Dict[str, Set[str]] = {}

    def _vibe_keys(vibe: str) -> Set[str]:
        cached = vibe_key_cache.get(vibe)
        if cached is not None:
            return cached
        vibe_data = _invoke_tool_json("get_vibe", {"vibe": vibe, "limit": 100})
        items = vibe_data.get("items")
        keys = {_track_key(row) for row in items} if isinstance(items, list) else set()
        vibe_key_cache[vibe] = keys
        return keys

    lines: List[str] = ["Supported vibe labels: " + ", ".join(VIBE_OPTIONS), ""]
    for ref in user_refs:
        rec_data = _invoke_tool_json(
            "get_recs",
            {"user_id": ref, "limit": 200, "fallback_to_trending": False},
        )
        context_name = str(rec_data.get("user_display_name") or rec_data.get("user_id") or ref).strip()
        mode = str(rec_data.get("mode") or "")
        rec_items = rec_data.get("items")
        if not mode.startswith("personalized") or not isinstance(rec_items, list):
            lines.append(f"{context_name}: personalized slice not available yet.")
            continue

        rec_keys = {_track_key(row) for row in rec_items}
        counts: List[tuple[str, int]] = []
        for vibe in VIBE_OPTIONS:
            overlap = len(rec_keys.intersection(_vibe_keys(vibe)))
            counts.append((vibe, overlap))
        positives = [f"{vibe}({count})" for vibe, count in counts if count > 0]
        if positives:
            lines.append(f"{context_name}: " + ", ".join(positives))
        else:
            lines.append(f"{context_name}: no strong personalized vibe overlap yet.")
    lines.append("")
    lines.append("Ask like: 'romantic songs for Aarav Edwards' or 'party songs for Abigail Johnson'.")
    return "\n".join(lines)


def _is_user_switch_followup_for_last_vibe(text: str) -> bool:
    """
    Detect follow-ups like:
    - now give for abi
    - show for aarav
    where user intent is likely "same vibe, different user".
    """
    lower = text.lower().strip()
    if any(tok in lower for tok in ["trending", "popular", "hot", "search", "find"]):
        return False
    if any(tok in lower for tok in ["recommend", "recs", "suggest"]):
        return False
    return (" for " in f" {lower} ") and any(tok in lower for tok in ["now", "give", "show", "play"])


def _genre_capability_response() -> str:
    vibes = ", ".join(VIBE_OPTIONS)
    return (
        "This demo is grounded on vibe labels (not full genre metadata). "
        f"Supported vibe requests: {vibes}.\n\n"
        "Examples:\n"
        "- party songs for Aarav Edwards\n"
        "- sad songs for Abigail Johnson\n"
        "- 5 more"
    )


def _references_last_user(text: str) -> bool:
    normalized = re.sub(r"[^a-z ]+", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    lower = f" {normalized} "
    return any(f" {token} " in lower for token in USER_PRONOUN_TOKENS)


def _track_key(row: Dict[str, Any]) -> str:
    track_id = str(row.get("track_id") or "").strip().lower()
    if track_id:
        return track_id
    track = str(row.get("track_name") or "").strip().lower()
    artist = str(row.get("artist_name") or "").strip().lower()
    return f"{track}::{artist}"


def _paginate_vibe_items(
    vibe_data: Dict[str, Any],
    seen_keys: Set[str],
    page_size: int,
) -> List[Dict[str, Any]]:
    items = vibe_data.get("items")
    if not isinstance(items, list):
        return []
    unseen: List[Dict[str, Any]] = []
    for row in items:
        key = _track_key(row)
        if key in seen_keys:
            continue
        unseen.append(row)
    page = unseen[: max(page_size, 1)]
    for row in page:
        seen_keys.add(_track_key(row))
    return page


def _invoke_tool_json(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    mapping = _tool_map()
    tool_obj = mapping[tool_name]
    raw = tool_obj.invoke(tool_args)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
    if isinstance(raw, dict):
        return raw
    return {"raw": str(raw)}


def _format_items(items: List[Dict[str, Any]], limit: int = 10) -> str:
    lines: List[str] = []
    for idx, row in enumerate(items[:limit], start=1):
        track = row.get("track_name") or row.get("track_id") or "__unknown_track__"
        artist = row.get("artist_name") or "__unknown_artist__"
        lines.append(f"{idx}. {track} - {artist}")
    return "\n".join(lines)


def _format_grounded_response(tool_name: str, data: Dict[str, Any]) -> str:
    if "error" in data:
        return f"Tool error: {data['error']}"

    items = data.get("items") if isinstance(data, dict) else None
    if isinstance(items, list):
        if not items:
            return data.get("message", "No matching tracks available in current dataset.")
        formatted_items = _format_items(items, limit=10)
        msg = data.get("message")
        if msg:
            return f"{msg}\n\n{formatted_items}"
        return formatted_items

    # metrics path
    if all(k in data for k in ("events", "users", "tracks")):
        return (
            f"Model slice metrics:\n"
            f"- events: {data.get('events')}\n"
            f"- users: {data.get('users')}\n"
            f"- tracks: {data.get('tracks')}\n"
            f"- events_per_user: {data.get('events_per_user')}"
        )

    return json.dumps(data, default=str)


def _is_missing_song_result(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    items = data.get("items")
    if not isinstance(items, list):
        return False
    return len(items) == 0


def _cancel_pending_requested(text: str) -> bool:
    lower = text.strip().lower()
    return any(token in lower for token in FALLBACK_CANCEL_TOKENS)


def _get_personalized_vibe_candidates(user_ref: str, vibe: str) -> tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    rec_data = _invoke_tool_json(
        "get_recs",
        {"user_id": user_ref, "limit": 200, "fallback_to_trending": False},
    )
    context_name = str(
        rec_data.get("user_display_name")
        or rec_data.get("user_id")
        or user_ref
    ).strip()
    rec_items = rec_data.get("items")
    if not isinstance(rec_items, list):
        return context_name, rec_data, []

    mode = str(rec_data.get("mode") or "")
    if not mode.startswith("personalized"):
        return context_name, rec_data, []

    vibe_data = _invoke_tool_json("get_vibe", {"vibe": vibe, "limit": 100})
    vibe_items = vibe_data.get("items")
    if not isinstance(vibe_items, list) or not vibe_items:
        return context_name, rec_data, []

    vibe_keys = {_track_key(row) for row in vibe_items}
    filtered = [row for row in rec_items if _track_key(row) in vibe_keys]
    return context_name, rec_data, filtered


def _get_vibe_fallback_page(
    vibe: str,
    page_size: int,
    vibe_seen_keys: Dict[str, Set[str]],
) -> tuple[List[Dict[str, Any]], str]:
    vibe_data = _invoke_tool_json("get_vibe", {"vibe": vibe, "limit": 100})
    seen = vibe_seen_keys.setdefault(vibe, set())
    page = _paginate_vibe_items(vibe_data, seen, page_size=page_size)
    message = str(vibe_data.get("message") or "")
    return page, message


@tool
def get_model_metrics() -> str:
    """Get model-slice metrics (events, users, tracks, events_per_user)."""
    return _to_json(rc.get_model_metrics())


@tool
def get_trending(limit: int = 10) -> str:
    """Get globally trending tracks. Use limit between 1 and 200."""
    return _to_json(rc.get_trending(limit=limit))


@tool
def get_recs(user_id: str, limit: int = 20, fallback_to_trending: bool = True) -> str:
    """Get personalized recommendations for a user reference (user_id or display name)."""
    return _to_json(
        rc.get_recs(
            user_id=user_id,
            limit=limit,
            fallback_to_trending=fallback_to_trending,
        )
    )


@tool
def get_user_favorites(user_id: str, limit: int = 20, fallback_to_recs: bool = True) -> str:
    """Get top listened tracks (favorites) for a user reference (user_id or display name)."""
    return _to_json(
        rc.get_user_favorites(
            user_id=user_id,
            limit=limit,
            fallback_to_recs=fallback_to_recs,
        )
    )


@tool
def search_tracks(query: str, limit: int = 5) -> str:
    """Search tracks by query text."""
    return _to_json(rc.search_tracks(query=query, limit=limit))


@tool
def get_vibe(vibe: str, limit: int = 10) -> str:
    """Get tracks by vibe label (chill/focus/happy/sad/party/energetic/romantic)."""
    return _to_json(rc.get_vibe(vibe=vibe, limit=limit))


@tool
def submit_feedback(
    user_id: str,
    track_id: str,
    vibe: str,
    predicted_vibe: str = "",
) -> str:
    """Submit user feedback to correct a track vibe."""
    pv = predicted_vibe.strip() or None
    return _to_json(
        rc.submit_feedback(
            user_id=user_id,
            track_id=track_id,
            vibe=vibe,
            predicted_vibe=pv,
        )
    )


TOOLS = [
    get_model_metrics,
    get_trending,
    get_recs,
    get_user_favorites,
    search_tracks,
    get_vibe,
    submit_feedback,
]


def _tool_map() -> Dict[str, Any]:
    return {t.name: t for t in TOOLS}


def assistant_turn(llm_with_tools: Any, messages: List[Any]) -> str:
    mapping = _tool_map()
    while True:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = ai_msg.tool_calls or []
        if not tool_calls:
            return _clean_model_text(_extract_text(ai_msg.content))

        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args", {})
            call_id = tc.get("id")
            tool_obj = mapping.get(name)
            if tool_obj is None:
                result = _to_json({"error": f"unknown tool: {name}"})
            else:
                try:
                    result = tool_obj.invoke(args)
                except Exception as exc:
                    result = _to_json(
                        {
                            "error": str(exc),
                            "tool": name,
                            "args": args,
                        }
                    )
            messages.append(ToolMessage(content=result, tool_call_id=call_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain Bedrock music assistant.")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "us-east-1"),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=700)
    args = parser.parse_args()

    llm = ChatBedrockConverse(
        model=args.model_id,
        region_name=args.region,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    pending_missing_song_query: str | None = None
    last_vibe_label: str | None = None
    vibe_seen_keys: Dict[str, Set[str]] = {}
    last_rec_user_ref: str | None = None
    rec_seen_keys: Dict[str, Set[str]] = {}
    user_vibe_seen_keys: Dict[str, Set[str]] = {}
    last_user_vibe_ref: str | None = None
    last_user_vibe_label: str | None = None
    pending_user_vibe_request: Dict[str, Any] | None = None
    last_user_list_tool: str | None = None

    print("LangChain Bedrock Music Assistant ready. Type 'exit' to quit.")
    print(f"model={args.model_id} region={args.region}")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            return
        if not user_text:
            continue

        if _is_genre_capability_question(user_text):
            capability_refs = _extract_user_refs_for_capability_question(user_text, fallback_user_ref=last_rec_user_ref)
            try:
                summary = _build_user_vibe_coverage_summary(capability_refs)
                print(f"\nAssistant: {summary}")
            except Exception:
                print(f"\nAssistant: {_genre_capability_response()}")
            continue

        if pending_user_vibe_request is not None:
            if _cancel_pending_requested(user_text):
                pending_user_vibe_request = None
                print("\nAssistant: No problem. I cancelled that request.")
                continue

            pending_user_ref = _extract_user_reference(user_text) or _extract_standalone_user_candidate(user_text)
            if pending_user_ref is None:
                print("\nAssistant: Tell me the user name (for example: 'Aarav Edwards') or type 'cancel'.")
                continue

            pending_vibe = str(pending_user_vibe_request.get("vibe") or "").strip().lower()
            pending_limit = int(pending_user_vibe_request.get("limit") or 10)
            pending_user_vibe_request = None
            try:
                context_name, rec_data, filtered_items = _get_personalized_vibe_candidates(
                    user_ref=pending_user_ref,
                    vibe=pending_vibe,
                )
                context_key = context_name.lower()
                state_key = f"{context_key}|{pending_vibe}"
                seen = user_vibe_seen_keys.setdefault(state_key, set())
                filtered_payload = {"items": filtered_items}
                page = _paginate_vibe_items(filtered_payload, seen, page_size=pending_limit)
                if page:
                    print(
                        f"\nAssistant: Here are {len(page)} personalized '{pending_vibe}' songs for {context_name}:\n\n"
                        f"{_format_items(page, limit=len(page))}"
                    )
                    last_user_vibe_ref = context_name
                    last_user_vibe_label = pending_vibe
                    last_vibe_label = pending_vibe
                    last_rec_user_ref = context_name
                else:
                    mode = str(rec_data.get("mode") or "")
                    if mode.startswith("personalized"):
                        last_user_vibe_ref = context_name
                        last_user_vibe_label = pending_vibe
                        last_vibe_label = pending_vibe
                        last_rec_user_ref = context_name
                        vibe_page, vibe_msg = _get_vibe_fallback_page(
                            pending_vibe,
                            pending_limit,
                            vibe_seen_keys,
                        )
                        if vibe_page:
                            print(
                                f"\nAssistant: I could not confidently map '{pending_vibe}' for {context_name} yet. "
                                f"Here are {len(vibe_page)} popular '{pending_vibe}' tracks:\n\n"
                                f"{_format_items(vibe_page, limit=len(vibe_page))}"
                            )
                        else:
                            fallback_seen = rec_seen_keys.setdefault(context_key, set())
                            fallback_page = _paginate_vibe_items(rec_data, fallback_seen, page_size=pending_limit)
                            if fallback_page:
                                print(
                                    f"\nAssistant: I couldn't find enough '{pending_vibe}' tracks in the catalog right now. "
                                    f"Here are {len(fallback_page)} personalized songs for {context_name} instead:\n\n"
                                    f"{_format_items(fallback_page, limit=len(fallback_page))}"
                                )
                            else:
                                final_msg = (
                                    vibe_msg
                                    or f"I couldn't find enough '{pending_vibe}' tracks in the current catalog."
                                )
                                print(f"\nAssistant: {final_msg}")
                    else:
                        vibe_page, vibe_msg = _get_vibe_fallback_page(
                            pending_vibe,
                            pending_limit,
                            vibe_seen_keys,
                        )
                        if vibe_page:
                            print(
                                f"\nAssistant: I couldn't find personalized results for {context_name} yet. "
                                f"Here are {len(vibe_page)} popular '{pending_vibe}' tracks:\n\n"
                                f"{_format_items(vibe_page, limit=len(vibe_page))}"
                            )
                        else:
                            final_msg = (
                                vibe_msg
                                or "No matching tracks available in current dataset."
                            )
                            print(f"\nAssistant: {final_msg}")
            except Exception as exc:
                print(f"\nAssistant error: {exc}")
            continue

        if pending_missing_song_query is not None:
            if _cancel_pending_requested(user_text):
                print("\nAssistant: No problem. I cancelled that song-based fallback.")
                pending_missing_song_query = None
                continue

            pending_vibe = _extract_vibe(user_text)
            if pending_vibe is None:
                print(
                    "\nAssistant: I still need a vibe to continue. "
                    "Try one of: chill, focus, happy, sad, party, energetic, romantic. "
                    "Or type 'cancel'."
                )
                continue

            try:
                vibe_data = _invoke_tool_json("get_vibe", {"vibe": pending_vibe, "limit": 100})
                vibe_seen_keys[pending_vibe] = set()
                page = _paginate_vibe_items(vibe_data, vibe_seen_keys[pending_vibe], page_size=10)
                if page:
                    grounded = _format_items(page, limit=10)
                else:
                    grounded = _format_grounded_response("get_vibe", vibe_data)
                print(
                    "\nAssistant: I couldn't find that exact song in the current dataset. "
                    f"Closest matches for '{pending_missing_song_query}' using vibe '{pending_vibe}':\n\n"
                    f"{grounded}"
                )
                last_vibe_label = pending_vibe
            except Exception as exc:
                print(f"\nAssistant error: {exc}")
            finally:
                pending_missing_song_query = None
            continue

        requested_vibe = _extract_vibe(user_text)
        explicit_user_ref = _extract_user_reference(user_text)
        refers_pronoun_user = _references_last_user(user_text)
        pronoun_user_ref = last_rec_user_ref if refers_pronoun_user else None
        target_user_ref = explicit_user_ref or pronoun_user_ref
        if (
            requested_vibe is None
            and explicit_user_ref is not None
            and last_user_vibe_label is not None
            and _is_user_switch_followup_for_last_vibe(user_text)
        ):
            requested_vibe = last_user_vibe_label
            target_user_ref = explicit_user_ref
        if (
            requested_vibe
            and target_user_ref is None
            and last_user_vibe_ref is not None
            and last_user_vibe_label == requested_vibe
        ):
            target_user_ref = last_user_vibe_ref

        if requested_vibe and refers_pronoun_user and last_rec_user_ref is None:
            pending_user_vibe_request = {
                "vibe": requested_vibe,
                "limit": _extract_requested_count(user_text, default=10),
            }
            print(
                "\nAssistant: Tell me which user you mean "
                f"(for example: '{requested_vibe} songs for Abigail Johnson')."
            )
            continue

        if requested_vibe and target_user_ref and not _is_more_request(user_text):
            page_size = _extract_requested_count(user_text, default=10)
            try:
                context_name, rec_data, filtered_items = _get_personalized_vibe_candidates(
                    user_ref=target_user_ref,
                    vibe=requested_vibe,
                )
                context_key = context_name.lower()
                state_key = f"{context_key}|{requested_vibe}"
                seen = user_vibe_seen_keys.setdefault(state_key, set())
                filtered_payload = {"items": filtered_items}
                page = _paginate_vibe_items(filtered_payload, seen, page_size=page_size)

                if page:
                    print(
                        f"\nAssistant: Here are {len(page)} personalized '{requested_vibe}' songs for {context_name}:\n\n"
                        f"{_format_items(page, limit=len(page))}"
                    )
                    last_user_vibe_ref = context_name
                    last_user_vibe_label = requested_vibe
                    last_vibe_label = requested_vibe
                    last_rec_user_ref = context_name
                else:
                    # If vibe intersection is empty, keep personalization instead of falling back to global vibe.
                    mode = str(rec_data.get("mode") or "")
                    if mode.startswith("personalized"):
                        last_user_vibe_ref = context_name
                        last_user_vibe_label = requested_vibe
                        last_vibe_label = requested_vibe
                        last_rec_user_ref = context_name
                        vibe_page, vibe_msg = _get_vibe_fallback_page(
                            requested_vibe,
                            page_size,
                            vibe_seen_keys,
                        )
                        if vibe_page:
                            print(
                                f"\nAssistant: I could not confidently map '{requested_vibe}' for {context_name} yet. "
                                f"Here are {len(vibe_page)} popular '{requested_vibe}' tracks:\n\n"
                                f"{_format_items(vibe_page, limit=len(vibe_page))}"
                            )
                        else:
                            fallback_seen = rec_seen_keys.setdefault(context_key, set())
                            fallback_page = _paginate_vibe_items(rec_data, fallback_seen, page_size=page_size)
                            if fallback_page:
                                print(
                                    f"\nAssistant: I couldn't find enough '{requested_vibe}' tracks in the catalog right now. "
                                    f"Here are {len(fallback_page)} personalized songs for {context_name} instead:\n\n"
                                    f"{_format_items(fallback_page, limit=len(fallback_page))}"
                                )
                            else:
                                final_msg = (
                                    vibe_msg
                                    or f"I couldn't find enough '{requested_vibe}' tracks in the current catalog."
                                )
                                print(f"\nAssistant: {final_msg}")
                    else:
                        vibe_page, vibe_msg = _get_vibe_fallback_page(
                            requested_vibe,
                            page_size,
                            vibe_seen_keys,
                        )
                        if vibe_page:
                            print(
                                f"\nAssistant: I couldn't find personalized results for {context_name} yet. "
                                f"Here are {len(vibe_page)} popular '{requested_vibe}' tracks:\n\n"
                                f"{_format_items(vibe_page, limit=len(vibe_page))}"
                            )
                        else:
                            final_msg = (
                                vibe_msg
                                or "No matching tracks available in current dataset."
                            )
                            print(f"\nAssistant: {final_msg}")
            except Exception as exc:
                print(f"\nAssistant error: {exc}")
            continue

        is_more_like = _is_more_request(user_text) or (
            _is_implicit_count_followup(user_text)
            and (last_user_vibe_ref is not None or last_rec_user_ref is not None or last_vibe_label is not None)
        )
        if is_more_like:
            requested_more_vibe = _extract_vibe(user_text)
            requested_more_user = _extract_user_reference(user_text)
            if requested_more_user is None and _references_last_user(user_text):
                requested_more_user = last_rec_user_ref

            if requested_more_user is None and last_user_vibe_ref is not None:
                if requested_more_vibe is None or requested_more_vibe == last_user_vibe_label:
                    requested_more_user = last_user_vibe_ref
                    if requested_more_vibe is None:
                        requested_more_vibe = last_user_vibe_label

            if requested_more_user and requested_more_vibe:
                page_size = _extract_requested_count(user_text, default=5)
                try:
                    context_name, rec_data, filtered_items = _get_personalized_vibe_candidates(
                        user_ref=requested_more_user,
                        vibe=requested_more_vibe,
                    )
                    context_key = context_name.lower()
                    state_key = f"{context_key}|{requested_more_vibe}"
                    seen = user_vibe_seen_keys.setdefault(state_key, set())
                    filtered_payload = {"items": filtered_items}
                    page = _paginate_vibe_items(filtered_payload, seen, page_size=page_size)
                    if page:
                        print(
                            f"\nAssistant: Here are {len(page)} more personalized '{requested_more_vibe}' songs for {context_name}:\n\n"
                            f"{_format_items(page, limit=len(page))}"
                        )
                        last_user_vibe_ref = context_name
                        last_user_vibe_label = requested_more_vibe
                        last_vibe_label = requested_more_vibe
                        last_rec_user_ref = context_name
                    else:
                        mode = str(rec_data.get("mode") or "")
                        if mode.startswith("personalized"):
                            vibe_page, vibe_msg = _get_vibe_fallback_page(
                                requested_more_vibe,
                                page_size,
                                vibe_seen_keys,
                            )
                            if vibe_page:
                                print(
                                    f"\nAssistant: I ran out of unseen personalized '{requested_more_vibe}' songs for {context_name}. "
                                    f"Here are {len(vibe_page)} additional popular '{requested_more_vibe}' tracks:\n\n"
                                    f"{_format_items(vibe_page, limit=len(vibe_page))}"
                                )
                            else:
                                final_msg = (
                                    vibe_msg
                                    or f"I ran out of unseen '{requested_more_vibe}' tracks in the current catalog."
                                )
                                print(f"\nAssistant: {final_msg}")
                        else:
                            vibe_page, vibe_msg = _get_vibe_fallback_page(
                                requested_more_vibe,
                                page_size,
                                vibe_seen_keys,
                            )
                            if vibe_page:
                                print(
                                    f"\nAssistant: I couldn't find additional personalized results for {context_name}. "
                                    f"Here are {len(vibe_page)} popular '{requested_more_vibe}' tracks:\n\n"
                                    f"{_format_items(vibe_page, limit=len(vibe_page))}"
                                )
                            else:
                                final_msg = (
                                    vibe_msg
                                    or "No matching tracks available in current dataset."
                                )
                                print(f"\nAssistant: {final_msg}")
                except Exception as exc:
                    print(f"\nAssistant error: {exc}")
                continue

            more_vibe = _extract_vibe(user_text) or last_vibe_label
            if more_vibe:
                page_size = _extract_requested_count(user_text, default=5)
                try:
                    vibe_data = _invoke_tool_json("get_vibe", {"vibe": more_vibe, "limit": 100})
                    seen = vibe_seen_keys.setdefault(more_vibe, set())
                    page = _paginate_vibe_items(vibe_data, seen, page_size=page_size)
                    if page:
                        print(
                            f"\nAssistant: Here are {len(page)} more '{more_vibe}' songs:\n\n"
                            f"{_format_items(page, limit=len(page))}"
                        )
                    else:
                        msg = vibe_data.get("message") or (
                            f"I ran out of unseen '{more_vibe}' tracks in the current catalog."
                        )
                        print(f"\nAssistant: {msg}")
                    last_vibe_label = more_vibe
                except Exception as exc:
                    print(f"\nAssistant error: {exc}")
                continue

            # Support "5 more" after personalized recommendations.
            more_user_ref = _extract_user_reference(user_text) or last_rec_user_ref
            if more_user_ref:
                page_size = _extract_requested_count(user_text, default=5)
                try:
                    list_tool = "get_user_favorites" if last_user_list_tool == "get_user_favorites" else "get_recs"
                    rec_data = _invoke_tool_json(
                        list_tool,
                        (
                            {
                                "user_id": more_user_ref,
                                "limit": 100,
                                "fallback_to_recs": True,
                            }
                            if list_tool == "get_user_favorites"
                            else {
                                "user_id": more_user_ref,
                                "limit": 100,
                                "fallback_to_trending": True,
                            }
                        ),
                    )
                    context_name = str(
                        rec_data.get("user_display_name")
                        or rec_data.get("user_id")
                        or more_user_ref
                    ).strip()
                    context_key = context_name.lower()
                    seen = rec_seen_keys.setdefault(context_key, set())
                    page = _paginate_vibe_items(rec_data, seen, page_size=page_size)
                    if page:
                        label = "favorites" if list_tool == "get_user_favorites" else "recommendations"
                        print(
                            f"\nAssistant: Here are {len(page)} more {label} for {context_name}:\n\n"
                            f"{_format_items(page, limit=len(page))}"
                        )
                    else:
                        msg = rec_data.get("message") or (
                            f"I ran out of unseen tracks for {context_name}."
                        )
                        print(f"\nAssistant: {msg}")
                    last_rec_user_ref = context_name
                    last_user_list_tool = list_tool
                except Exception as exc:
                    print(f"\nAssistant error: {exc}")
                continue

        planned = _route_tool(user_text)
        if planned is not None:
            tool_name, tool_args = planned
            try:
                if tool_name in {"get_recs", "get_user_favorites"}:
                    tool_args = dict(tool_args)
                    tool_args["limit"] = max(int(tool_args.get("limit", 20)), 100)
                data = _invoke_tool_json(tool_name, tool_args)
                if tool_name == "search_tracks" and _is_missing_song_result(data):
                    pending_missing_song_query = str(tool_args.get("query", "")).strip() or "that song"
                    print(
                        "\nAssistant: I couldn't find that exact song in the current dataset. "
                        "Tell me the vibe you want (chill/focus/happy/sad/party/energetic/romantic) "
                        "and I'll return the closest tracks."
                    )
                elif tool_name == "get_vibe":
                    vibe = str(tool_args.get("vibe") or "").strip().lower()
                    page_size = _extract_requested_count(user_text, default=10)
                    vibe_seen_keys[vibe] = set()
                    page = _paginate_vibe_items(data, vibe_seen_keys[vibe], page_size=page_size)
                    if page:
                        msg = data.get("message") or f"Here are the closest '{vibe}' tracks from your catalog."
                        print(f"\nAssistant: {msg}\n\n{_format_items(page, limit=len(page))}")
                    else:
                        print(f"\nAssistant: {_format_grounded_response(tool_name, data)}")
                    last_vibe_label = vibe or last_vibe_label
                elif tool_name == "get_recs":
                    page_size = _extract_requested_count(user_text, default=10)
                    requested_user_ref = str(tool_args.get("user_id") or "").strip()
                    context_name = str(
                        data.get("user_display_name")
                        or data.get("user_id")
                        or requested_user_ref
                    ).strip()
                    context_key = context_name.lower()
                    rec_seen_keys[context_key] = set()
                    page = _paginate_vibe_items(data, rec_seen_keys[context_key], page_size=page_size)
                    if page:
                        mode = str(data.get("mode") or "")
                        if mode.startswith("personalized"):
                            intro = f"Here are {len(page)} personalized recommendations for {context_name}:"
                        else:
                            intro = data.get("message") or f"Here are recommendations for {context_name}:"
                        print(f"\nAssistant: {intro}\n\n{_format_items(page, limit=len(page))}")
                    else:
                        print(f"\nAssistant: {_format_grounded_response(tool_name, data)}")
                    last_rec_user_ref = context_name
                    last_user_vibe_ref = context_name
                    last_user_list_tool = "get_recs"
                elif tool_name == "get_user_favorites":
                    page_size = _extract_requested_count(user_text, default=10)
                    requested_user_ref = str(tool_args.get("user_id") or "").strip()
                    context_name = str(
                        data.get("user_display_name")
                        or data.get("user_id")
                        or requested_user_ref
                    ).strip()
                    context_key = context_name.lower()
                    rec_seen_keys[context_key] = set()
                    page = _paginate_vibe_items(data, rec_seen_keys[context_key], page_size=page_size)
                    if page:
                        mode = str(data.get("mode") or "")
                        if mode == "user_favorites":
                            intro = f"Here are {len(page)} favorite tracks for {context_name}:"
                        else:
                            intro = data.get("message") or f"Here are tracks for {context_name}:"
                        print(f"\nAssistant: {intro}\n\n{_format_items(page, limit=len(page))}")
                    else:
                        print(f"\nAssistant: {_format_grounded_response(tool_name, data)}")
                    last_rec_user_ref = context_name
                    last_user_list_tool = "get_user_favorites"
                else:
                    print(f"\nAssistant: {_format_grounded_response(tool_name, data)}")
            except Exception as exc:
                print(f"\nAssistant error: {exc}")
            continue

        messages.append(HumanMessage(content=user_text))
        try:
            answer = assistant_turn(llm_with_tools=llm_with_tools, messages=messages)
            print(f"\nAssistant: {answer}")
        except Exception as exc:
            print(f"\nAssistant error: {exc}")


if __name__ == "__main__":
    main()
