#!/usr/bin/env python3
"""
LLM-powered music assistant grounded on the local recommendation API.

Usage:
  python scripts/llm_music_assistant.py

Required env:
  OPENAI_API_KEY

Optional env:
  OPENAI_MODEL (default: gpt-4o-mini)
  RECOMMENDATION_API_BASE_URL (default: http://localhost:8000)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from recommendation_client import (
    get_model_metrics,
    get_recs,
    get_trending,
    get_vibe,
    search_tracks,
    submit_feedback,
)


SYSTEM_PROMPT = """You are a music recommendation assistant for a real data platform.
You MUST use tools for recommendations/search/vibe/metrics and never invent tracks.
Rules:
1) If the user asks for personalized recommendations, call get_recs.
2) If user asks for a song/title, call search_tracks first.
3) If a song is not found or user asks by mood/vibe, call get_vibe.
4) If user asks for hot/global songs, call get_trending.
5) Keep answers concise and practical.
6) If tools return empty or fallback, explain what happened and suggest the next best option.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": "Get dataset metrics used by the recommendation model.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trending",
            "description": "Get globally trending tracks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 10}
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recs",
            "description": "Get personalized recommendations for a user_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                    "fallback_to_trending": {"type": "boolean", "default": True},
                },
                "required": ["user_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tracks",
            "description": "Search tracks by text query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vibe",
            "description": "Get tracks by vibe label (chill/focus/happy/sad/party/energetic/romantic).",
            "parameters": {
                "type": "object",
                "properties": {
                    "vibe": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                },
                "required": ["vibe"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_feedback",
            "description": "Submit user feedback to correct a track vibe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "track_id": {"type": "string"},
                    "vibe": {"type": "string"},
                    "predicted_vibe": {"type": "string"},
                },
                "required": ["user_id", "track_id", "vibe"],
                "additionalProperties": False,
            },
        },
    },
]


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if name == "get_model_metrics":
            return get_model_metrics()
        if name == "get_trending":
            return get_trending(limit=int(args.get("limit", 10)))
        if name == "get_recs":
            return get_recs(
                user_id=str(args["user_id"]),
                limit=int(args.get("limit", 20)),
                fallback_to_trending=bool(args.get("fallback_to_trending", True)),
            )
        if name == "search_tracks":
            return search_tracks(query=str(args["query"]), limit=int(args.get("limit", 5)))
        if name == "get_vibe":
            return get_vibe(vibe=str(args["vibe"]), limit=int(args.get("limit", 10)))
        if name == "submit_feedback":
            return submit_feedback(
                user_id=str(args["user_id"]),
                track_id=str(args["track_id"]),
                vibe=str(args["vibe"]),
                predicted_vibe=(str(args["predicted_vibe"]) if args.get("predicted_vibe") else None),
            )
        return {"error": f"Unknown tool '{name}'"}
    except Exception as exc:
        return {"error": str(exc), "tool": name, "args": args}


def parse_tool_args(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def llm_turn(client: OpenAI, model: str, messages: List[Dict[str, Any]]) -> str:
    while True:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = completion.choices[0].message

        tool_calls = msg.tool_calls or []
        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                args = parse_tool_args(tc.function.arguments)
                result = run_tool(tc.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps(result, default=str),
                    }
                )
            continue

        answer = msg.content or ""
        messages.append({"role": "assistant", "content": answer})
        return answer.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-powered music assistant.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required.")

    client = OpenAI(api_key=api_key)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("LLM Music Assistant ready. Type 'exit' to quit.")
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

        messages.append({"role": "user", "content": user_text})
        try:
            response = llm_turn(client, args.model, messages)
            print(f"\nAssistant: {response}")
        except Exception as exc:
            print(f"\nAssistant error: {exc}")


if __name__ == "__main__":
    main()
