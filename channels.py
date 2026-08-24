"""Loads the list of channels to collect from.

Personal channel IDs live in channels.json (gitignored, not shared) - copy
channels.example.json to get started, then replace the IDs with your own
(see README for how to find a channel's numeric ID). Message IDs are only
unique within a chat, so each channel is synced and tracked independently.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "channels.json"
EXAMPLE_PATH = Path(__file__).parent / "channels.example.json"


def load_channels() -> dict[int, str]:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"{CONFIG_PATH.name} not found. Copy {EXAMPLE_PATH.name} to {CONFIG_PATH.name} "
            "and fill in your own channel IDs (see README)."
        )
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {int(chat_id): name for chat_id, name in raw.items()}


CHANNELS = load_channels()
