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
from typing import Any, Dict, List

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import recommendation_client as rc


SYSTEM_PROMPT = """
You are a music recommendation assistant for a real data platform.
Always use tools for recommendations/search/vibe/metrics and do not invent tracks.

Rules:
1) Personalized ask -> use get_recs.
2) Song/title ask -> use search_tracks first.
3) If user asks by mood/vibe or song not found -> use get_vibe.
4) Global/hot ask -> use get_trending.
5) Keep answers concise and practical.
6) If data is missing/fallback, explain why and give the next best option.
7) Never output <thinking> tags, XML tags, or hidden reasoning. Respond in plain text only.
""".strip()


VIBE_OPTIONS = ["chill", "focus", "happy", "sad", "party", "energetic", "romantic"]


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


def _route_tool(text: str) -> tuple[str, Dict[str, Any]] | None:
    lower = text.lower()
    user_id = _extract_user_id(text)
    vibe = _extract_vibe(text)

    if user_id and any(t in lower for t in ["recommend", "recs", "suggest"]):
        return ("get_recs", {"user_id": user_id, "limit": 20, "fallback_to_trending": True})

    if any(t in lower for t in ["trending", "popular", "hot", "top songs", "top tracks"]):
        return ("get_trending", {"limit": 10})

    if vibe is not None:
        return ("get_vibe", {"vibe": vibe, "limit": 10})

    m = re.search(r"(?:find|search|song|track)\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        query = m.group(1).strip().strip('"')
        if query:
            return ("search_tracks", {"query": query, "limit": 5})

    return None


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
    """Get personalized recommendations for a user_id."""
    return _to_json(
        rc.get_recs(
            user_id=user_id,
            limit=limit,
            fallback_to_trending=fallback_to_trending,
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

        planned = _route_tool(user_text)
        if planned is not None:
            tool_name, tool_args = planned
            try:
                data = _invoke_tool_json(tool_name, tool_args)
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
