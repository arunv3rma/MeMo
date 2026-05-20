#!/usr/bin/env python3
"""Simple script to query the cartridges server with or without cartridges."""

from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def build_payload(question: str, model: str, with_cartridge: bool, cartridge_id: str | None) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question in JSON format "
                "with an 'answer' field that is a string. Do not wrap JSON in markdown."
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.2,
    }

    if with_cartridge:
        if not cartridge_id:
            raise SystemExit("--cartridge-id is required when --with-cartridge is set")
        payload["cartridges"] = [{"id": cartridge_id, "source": "local"}]

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the cartridges server")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:10210/custom/cartridge/chat/completions",
        help="Server endpoint URL",
    )
    parser.add_argument(
        "--question",
        default="What is the capital of Paris?",
        help="Question to send",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-32B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--with-cartridge",
        action="store_true",
        help="Include a cartridge in the request",
    )
    parser.add_argument(
        "--cartridge-id",
        default=None,
        help="Cartridge id to load (required when --with-cartridge)",
    )

    args = parser.parse_args()

    payload = build_payload(args.question, args.model, args.with_cartridge, args.cartridge_id)

    try:
        response = requests.post(args.endpoint, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise SystemExit(f"Request failed: {exc}") from exc

    print("Status:", response.status_code)
    print("Raw response:")
    print(response.text)

    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            data = response.json()
        except json.JSONDecodeError:
            return

        # Try to print model output if present
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is not None:
                print("\nParsed content:")
                print(content)


if __name__ == "__main__":
    main()
