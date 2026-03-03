#!/usr/bin/env python3
"""
Bedrock-powered music assistant grounded on the local recommendation API.

Usage:
  python scripts/bedrock_music_assistant.py

Required setup:
  - AWS credentials configured (profile or env vars)
  - Bedrock model access enabled in selected region

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

import boto3

from recommendation_client import (
    get_model_metrics,
    get_recs,
    get_trending,
    get_vibe,
    search_tracks,
    submit_feedback,
)


SYSTEM_PROMPT = """
You are a music recommendation assistant for a real data platform.
You must use tools for recommendations/search/vibe/metrics and never invent tracks.

Rules:
1) If user asks personalized recommendations, call get_recs.
2) If user asks for a song/title, call search_tracks first.
3) If requested song is missing or user asks by mood, call get_vibe.
4) If user asks for global/hot songs, call get_trending.
5) Keep responses concise and practical.
6) If data is missing, explain and suggest the next best option.
7) Never output <thinking> tags, XML tags, or hidden reasoning. Respond in plain text only.
""".strip()


TOOLS: List[Dict[str, Any]] = [
    {
        "toolSpec": {
            "name": "get_model_metrics",
            "description": "Get dataset metrics used by the recommendation model.",
            "inputSchema": {"json": {"type": "object", "properties": {}, "additionalProperties": False}},
        }
    },
    {
        "toolSpec": {
            "name": "get_trending",
            "description": "Get globally trending tracks.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 10}
                    },
                    "additionalProperties": False,
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_recs",
            "description": "Get personalized recommendations for a user_id.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                        "fallback_to_trending": {"type": "boolean", "default": True},
                    },
                    "required": ["user_id"],
                    "additionalProperties": False,
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "search_tracks",
            "description": "Search tracks by text query.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_vibe",
            "description": "Get tracks by vibe label (chill/focus/happy/sad/party/energetic/romantic).",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "vibe": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                    },
                    "required": ["vibe"],
                    "additionalProperties": False,
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "submit_feedback",
            "description": "Submit user feedback to correct a track vibe.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "track_id": {"type": "string"},
                        "vibe": {"type": "string"},
                        "predicted_vibe": {"type": "string"},
                    },
                    "required": ["user_id", "track_id", "vibe"],
                    "additionalProperties": False,
                }
            },
        }
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


def clean_model_text(text: str) -> str:
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?response>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?thinking>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def create_bedrock_client(region: str, profile: str | None) -> Any:
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
    else:
        session = boto3.Session(region_name=region)
    client = session.client("bedrock-runtime", region_name=region)
    if not hasattr(client, "converse"):
        raise RuntimeError("Installed boto3 does not support Bedrock converse API. Upgrade boto3/botocore.")
    return client


def run_conversation_turn(
    client: Any,
    model_id: str,
    messages: List[Dict[str, Any]],
) -> str:
    while True:
        response = client.converse(
            modelId=model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig={"tools": TOOLS},
            inferenceConfig={"temperature": 0.2, "maxTokens": 700},
        )

        output_message = response["output"]["message"]
        content_blocks = output_message.get("content", [])
        tool_uses = [blk["toolUse"] for blk in content_blocks if "toolUse" in blk]
        text_parts = [blk["text"] for blk in content_blocks if "text" in blk]

        if tool_uses:
            messages.append(output_message)
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use["name"]
                tool_input = tool_use.get("input", {})
                result = run_tool(tool_name, tool_input)
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"json": result}],
                        }
                    }
                )

            messages.append({"role": "user", "content": tool_results})
            continue

        messages.append(output_message)
        text = "\n".join(text_parts).strip()
        if text:
            return clean_model_text(text)
        return json.dumps(response, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bedrock LLM music assistant.")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "us-east-1"),
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE"),
    )
    args = parser.parse_args()

    client = create_bedrock_client(region=args.region, profile=args.profile)
    messages: List[Dict[str, Any]] = []

    print("Bedrock Music Assistant ready. Type 'exit' to quit.")
    print(f"model={args.model_id} region={args.region} profile={args.profile or '__default__'}")

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

        messages.append({"role": "user", "content": [{"text": user_text}]})
        try:
            answer = run_conversation_turn(client=client, model_id=args.model_id, messages=messages)
            print(f"\nAssistant: {answer}")
        except Exception as exc:
            print(f"\nAssistant error: {exc}")


if __name__ == "__main__":
    main()
