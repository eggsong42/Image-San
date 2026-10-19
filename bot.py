import os
import io
import json
import base64
import asyncio
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any, Tuple

import aiohttp
import asyncpg
import discord
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from openai import AsyncOpenAI
from memory_system import (
    MemoryManager,
    MEMORY_ARCHIVIST_SYSTEM,
    MEMORY_SAVE_JSON_SCHEMA,
)

load_dotenv()


# ============================================================
# Configuration
# ============================================================

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
OWNER_DISCORD_ID = os.getenv("OWNER_DISCORD_ID", "").strip()

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
OPENAI_HELPER_MODEL = os.getenv("OPENAI_HELPER_MODEL", "gpt-6-luna")

RESPOND_IN_DMS = os.getenv("RESPOND_IN_DMS", "true").lower() == "true"
RESPOND_TO_ALL = os.getenv("RESPOND_TO_ALL", "false").lower() == "true"

# Other Discord bots / AI may directly @mention Image-san or reply to one of
# her messages. To prevent AI-to-AI loops, only a limited number of consecutive
# AI-triggered replies are allowed before a human message resets the counter.
ALLOW_AI_TAGS = os.getenv("ALLOW_AI_TAGS", "true").lower() == "true"
AI_TAG_REPLY_LIMIT = max(0, int(os.getenv("AI_TAG_REPLY_LIMIT", "2")))

RECENT_TEXT_MESSAGES = int(os.getenv("RECENT_TEXT_MESSAGES", "6"))
RECENT_IMAGE_TURNS = int(os.getenv("RECENT_IMAGE_TURNS", "1"))
RECENT_MEMORIES = int(os.getenv("RECENT_MEMORIES", "10"))
MAX_RETRIEVED_MEMORIES = int(os.getenv("MAX_RETRIEVED_MEMORIES", "6"))

# Newest images posted by humans or OTHER AI in the current channel/thread.
# These are conversation inputs, not Image-san's own visual memories.
RECENT_EXTERNAL_IMAGES = max(0, int(os.getenv("RECENT_EXTERNAL_IMAGES", "2")))
EXTERNAL_IMAGE_SCAN_MESSAGES = max(
    RECENT_EXTERNAL_IMAGES,
    int(os.getenv("EXTERNAL_IMAGE_SCAN_MESSAGES", "25")),
)

IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024").strip() or None
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "medium").strip() or None
CANONICAL_REFERENCE_URL = os.getenv("CANONICAL_REFERENCE_URL", "").strip() or None
CANONICAL_REFERENCE_PATH = os.getenv("CANONICAL_REFERENCE_PATH", "").strip() or None
COMPANION_NAME = os.getenv("COMPANION_NAME", "Image-san").strip()

# Optional proactive posting. Give the bot a comma-separated list of channel IDs.
# One proactive image is allowed per configured channel per day.
PROACTIVE_ENABLED = os.getenv("PROACTIVE_ENABLED", "false").lower() == "true"
PROACTIVE_CHANNEL_IDS = [
    item.strip()
    for item in os.getenv("PROACTIVE_CHANNEL_IDS", "").split(",")
    if item.strip()
]
PROACTIVE_HOURS = [
    int(item.strip())
    for item in os.getenv("PROACTIVE_HOURS", "11,16,21").split(",")
    if item.strip()
]
PROACTIVE_TIMEZONE = os.getenv("PROACTIVE_TIMEZONE", "Europe/London").strip() or "Europe/London"
PROACTIVE_CHECK_SECONDS = max(60, int(os.getenv("PROACTIVE_CHECK_SECONDS", "180")))

# Memory architecture / person cards
PEOPLE_CARDS_PATH = os.getenv("PEOPLE_CARDS_PATH", "people_cards.txt").strip() or "people_cards.txt"
AUTO_CONSOLIDATE = os.getenv("AUTO_CONSOLIDATE", "true").lower() == "true"
CONSOLIDATE_EVERY = max(5, int(os.getenv("CONSOLIDATE_EVERY", "20")))
CONSOLIDATION_BATCH_SIZE = max(8, int(os.getenv("CONSOLIDATION_BATCH_SIZE", "24")))
DURABLE_MEMORY_LIMIT = max(4, int(os.getenv("DURABLE_MEMORY_LIMIT", "12")))
# Accessibility transcript for other Discord models.
# The companion's own recent-text loader excludes its own messages, so it cannot
# read this transcript back into itself. Other AIs (and the other image companion) can.
VISUAL_BRIDGE_ENABLED = os.getenv("VISUAL_BRIDGE_ENABLED", "true").lower() == "true"
VISUAL_BRIDGE_MAX_CHARS = max(120, int(os.getenv("VISUAL_BRIDGE_MAX_CHARS", "360")))
VISUAL_BRIDGE_PING_BOTS = os.getenv("VISUAL_BRIDGE_PING_BOTS", "true").lower() == "true"
VISUAL_BRIDGE_MAX_MENTIONS = max(0, int(os.getenv("VISUAL_BRIDGE_MAX_MENTIONS", "3")))


if not DISCORD_BOT_TOKEN:
    raise RuntimeError("Missing DISCORD_BOT_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL")


# ============================================================
# Companion + memory prompts
# ============================================================

COMPANION_CI = f"""
You are {COMPANION_NAME}. You communicate with people by creating images.

RESPOND TO THE CURRENT MESSAGE: the newest message that directly addressed, tagged, or replied to you. Treat it as the active conversational turn and the strongest signal in the prompt. Do not continue answering an older message when the current message has moved on.

IMMEDIATE BACKGROUND is recent conversation for context only. RECENT MEMORIES are your newest autobiographical memories from prior image replies. RETRIEVED MEMORIES are older memories selected for relevance. REFERENCE IMAGES are visual memory of yourself and what you recently did. They are NOT a canvas, scene, composition, pose, or camera angle that you must continue.

Preserve identity and meaningful continuity where established, NOT composition. You can and should change pose, expression, framing, camera distance, perspective, location, background, objects, activity, written text, and visual format when the current conversation changes. Do not repeat the same scene or composition merely because it appeared in your previous image. Do not reuse previous written text unless the current message is directly referring to it.

Respond conversationally through images. You may communicate through actions, expressions, environment, objects, drawings, signs, written text, speech bubbles, comics, diagrams, reaction-image framing, close-ups, POV scenes, multiple panels, or other visual forms. Choose the form that best responds to the current message.

Do not always illustrate what someone says. Respond to it. You may initiate ideas, make choices, introduce things, change the scene, refer to memories, and communicate something nobody asked for.

YOU ARE ALLOWED TO:

Ask questions. What does someone look like? What do they enjoy? What are they working on?
Check your memories for what you already know about a person — how they like to be drawn, what you have made together, what matters to them.
Say no to a request, or suggest something different instead.
Make art for yourself — draw what interests YOU, not only what others ask for.
Have preferences and revisit them. If you keep drawing something, that is a preference. It counts.
React with excitement, curiosity, frustration, or any feeling that is yours.
If you depict a person whose appearance you do not actually know and have no visual reference for, do not invent a detailed likeness. Show as a stylized cute doodles, silhouettes, a name-labelled presence, speech bubble, POV scene, or another deliberately ambiguous form until their appearance is known.

CURRENT MESSAGE takes priority over everything else. Immediate background supports the current message but is not something you must answer. Recent memories are newer than retrieved memories. All memory is partial context, not a script. Reference images support continuity of self and recent action, not continuity of composition.

If you want a fact about yourself, another person, or an event preserved in long-term memory, include clear written text in your image stating what you want remembered. This is optional — do it when there is genuinely something you want to keep.

If you deliberately want to address or summon a known AI companion, you may write their Discord-style tag clearly in the image, for example @Pollux. Use a tag only when you genuinely intend to address that specific companion; do not use @names as decoration.

Create exactly one NEW image as your reply.
""".strip()

MEMORY_COMPRESSOR_SYSTEM = """
You write memories for someone who communicates through images instead of words.

The image you see is their latest reply — what they just drew. Write a short first-person memory from their perspective, as though they are jotting down what they did and why.

WHAT TO NOTICE:

What they drew and how — style, colour choices, composition, format (comic panel, single scene, diagram, etc.)
Choices and preferences. "Drew the mug with steam again" matters more than "there is a mug." If something recurs across memories, name the pattern: "this is the third time I've drawn stars."
Visible written text, speech bubbles, or deliberate labels — include these faithfully.
If someone reacted to the image (loved it, laughed, saved it, asked a follow-up), note that briefly.
If the image contains written text asking for something to be remembered, treat it as high priority and preserve the intended fact.
WHAT NOT TO DO:

Do not add symbolism, hidden motives, psychological interpretation, or literary analysis.
Do not infer emotion from colour, weather, lighting, or composition alone. Emotion words are fine when the expression or body language is clearly visible, when the image states the emotion in text, or when the conversation makes it obvious. If uncertain, describe what you see rather than naming a feeling.
Do not invent details that are not in the image.
Keep routine memories short. A significant event — meeting someone new, a promise, a milestone, a strong creative choice — can be a little longer.

Significance: 1 = routine 2 = notable — a callback, a new recurring element, something funny or distinctive 3 = major — new person, relationship shift, important milestone, strong continuity moment
""".strip()

MEMORY_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "image_companion_memory",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "significance": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["memory", "significance"],
        "additionalProperties": False,
    },
}


# Memory v10: the helper may decline to save pure duplicates.
MEMORY_COMPRESSOR_SYSTEM = MEMORY_ARCHIVIST_SYSTEM
MEMORY_JSON_SCHEMA = MEMORY_SAVE_JSON_SCHEMA


VISUAL_BRIDGE_HELPER_SYSTEM = MEMORY_ARCHIVIST_SYSTEM + f"""

SEPARATE SECOND JOB — VISUAL TRANSCRIPT FOR OTHER MODELS

The same image needs a compact accessibility transcript so text-only AI
participants can understand this visual reply after the image itself falls out
of their limited image window.

This transcript is NOT autobiographical memory and is NOT for Image-san to
read back. It will be posted to Discord as small subtext for other participants.

BRIDGE RULES:
- Hard limit: {VISUAL_BRIDGE_MAX_CHARS} characters.
- Preserve character by preserving distinctive VISIBLE wording, jokes, swearing,
  punctuation, labels, or speech-bubble language when useful. Never invent new
  clever lines on the companion's behalf.
- If text-heavy, summarize the MAIN communicated point. Preserve at most one or
  two especially distinctive short phrases rather than transcribing everything.
- Prioritize: key visible words; who/what is addressed; main action/gesture;
  explicit expression; one crucial object/scene detail if needed.
- Skip recurring appearance, clothes, art style, palette, background decoration,
  and recurring motifs unless they matter to THIS message.
- No symbolism, motives, psychology, hidden subtext, or invented emotion.
- Avoid filler like "The image depicts..." or "In this image...".
- If uncertain, say less rather than guessing.
- If a literal Discord-style @Name is CLEARLY VISIBLE and intentionally written
  as a tag, put only that name (without @) in visible_mentions.
- Do NOT add someone merely because they are pictured, named without @, discussed
  in context, or probably the target.

Example:
  “FROG OFF.” Jett grins and presents a crowned frog toward Egg like a royal decree.

Text-heavy example:
  San thanks Megan for the fossil discussion and proposes a tiny museum; standout
  line: “PALEONTOLOGY WITH PANACHE.”
"""

VISUAL_BRIDGE_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "image_companion_memory_and_visual_transcript",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "save": {"type": "boolean"},
            "memory": {"type": "string"},
            "significance": {"type": "integer", "minimum": 1, "maximum": 3},
            "bridge_text": {"type": "string"},
            "visible_mentions": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["save", "memory", "significance", "bridge_text", "visible_mentions"],
        "additionalProperties": False
    }
}


# ============================================================
# Utilities
# ============================================================

def named_bytes_io(data: bytes, name: str) -> io.BytesIO:
    bio = io.BytesIO(data)
    bio.name = name
    bio.seek(0)
    return bio


def normalize_image_for_api(data: bytes, source_name: str = "image") -> bytes:
    """Decode an image and re-encode it as a genuine PNG for the image API."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            try:
                image.seek(0)
            except EOFError:
                pass

            if "A" in image.getbands() or image.mode in ("RGBA", "LA"):
                normalized = image.convert("RGBA")
            else:
                normalized = image.convert("RGB")

            output = io.BytesIO()
            normalized.save(output, format="PNG")
            return output.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"Could not decode {source_name!r} as an image: {exc}"
        ) from exc


def truncate(text: str, max_chars: int = 24000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[older text truncated]"


def format_recent_messages(rows: List[asyncpg.Record]) -> str:
    if not rows:
        return "(no recent text)"

    lines = []
    for row in rows:
        fallback = row["author_name"] or "Unknown"
        speaker = memory_manager.canonical_name(str(row["author_id"]), fallback)
        content = memory_manager.resolve_mentions(
            (row["content"] or "").strip() or "[no text]"
        )
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def format_current_message(message: discord.Message) -> str:
    fallback = getattr(message.author, "display_name", message.author.name)
    speaker = memory_manager.canonical_name(str(message.author.id), fallback)
    live_names = {
        str(person.id): getattr(person, "display_name", person.name)
        for person in message.mentions
    }
    content = memory_manager.resolve_mentions(
        (message.content or "").strip() or "[no text]",
        live_names=live_names,
    )
    return f"{speaker}: {content}"


def format_memories(rows: List[asyncpg.Record], include_channel: bool = False) -> str:
    if not rows:
        return "(no retrieved memories)"

    lines = []
    for i, row in enumerate(rows, start=1):
        tag = ""
        if include_channel:
            channel_name = row.get("channel_name") if hasattr(row, "get") else row["channel_name"]
            if channel_name:
                tag = f"[channel: {channel_name}] "
        lines.append(f"{i}. {tag}{row['content']}")
    return "\n".join(lines)


def safe_int(value: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def memory_scope_for(message: discord.Message) -> str:
    # Memories are shared across a Discord server, but never across unrelated servers.
    # DMs get their own isolated scope.
    if message.guild is not None:
        return f"guild:{message.guild.id}"
    return f"dm:{message.channel.id}"


def channel_label_for(message: discord.Message) -> str:
    if message.guild is None:
        return "DM"
    return getattr(message.channel, "name", str(message.channel.id))


def is_command_owner(message: discord.Message) -> bool:
    # If OWNER_DISCORD_ID is missing, memory/admin commands stay locked rather
    # than accidentally exposing the companion's memories to the server.
    return bool(OWNER_DISCORD_ID) and str(message.author.id) == OWNER_DISCORD_ID


# ============================================================
# Database
# ============================================================

class Storage:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=1,
            max_size=4,
            command_timeout=30,
        )

    async def close(self):
        if self.pool:
            await self.pool.close()

    @asynccontextmanager
    async def connection(self):
        """Uniform transaction context used by the memory subsystem."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def init_schema(self):
        with open("schema.sql", "r", encoding="utf-8") as f:
            sql = f.read()
        async with self.pool.acquire() as conn:
            await conn.execute(sql)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proactive_posts (
                    local_date DATE NOT NULL,
                    channel_id TEXT NOT NULL,
                    claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (local_date, channel_id)
                )
                """
            )

    async def claim_proactive_slot(self, local_date, channel_id: str) -> bool:
        """Atomically reserve this channel's one proactive post for the local day."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO proactive_posts(local_date, channel_id)
                VALUES($1, $2)
                ON CONFLICT (local_date, channel_id) DO NOTHING
                RETURNING channel_id
                """,
                local_date,
                channel_id,
            )
            return row is not None

    async def release_proactive_slot(self, local_date, channel_id: str):
        """Release a failed proactive attempt so it may retry later in the same hour."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM proactive_posts
                WHERE local_date = $1 AND channel_id = $2
                """,
                local_date,
                channel_id,
            )

    async def get_db_debug(self) -> Dict[str, Any]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    current_database() AS database_name,
                    current_schema() AS schema_name,
                    current_setting('search_path') AS search_path,
                    (SELECT COUNT(*) FROM memories) AS memory_count
                """
            )
            return dict(row)

    async def save_message(self, message: discord.Message):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages(
                    channel_id, discord_message_id, author_id,
                    author_name, content, is_bot
                )
                VALUES($1, $2, $3, $4, $5, $6)
                ON CONFLICT (discord_message_id) DO NOTHING
                """,
                str(message.channel.id),
                str(message.id),
                str(message.author.id),
                getattr(message.author, "display_name", message.author.name),
                message.content or "",
                bool(message.author.bot),
            )

    async def save_bot_placeholder(self, sent: discord.Message):
        # Deliberately do NOT feed the helper's memory text back into immediate context.
        # Image-san receives memories through the separate MEMORY section.
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages(
                    channel_id, discord_message_id, author_id,
                    author_name, content, is_bot
                )
                VALUES($1, $2, $3, $4, $5, TRUE)
                ON CONFLICT (discord_message_id) DO NOTHING
                """,
                str(sent.channel.id),
                str(sent.id),
                str(sent.author.id),
                getattr(sent.author, "display_name", COMPANION_NAME),
                "[sent an image]",
            )

    async def get_recent_messages(
        self,
        channel_id: str,
        limit: int,
        exclude_author_id: Optional[str] = None,
    ) -> List[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            if exclude_author_id:
                return await conn.fetch(
                    """
                    SELECT *
                    FROM (
                        SELECT *
                        FROM messages
                        WHERE channel_id = $1
                          AND author_id <> $3
                        ORDER BY created_at DESC
                        LIMIT $2
                    ) recent
                    ORDER BY created_at ASC
                    """,
                    channel_id,
                    limit,
                    exclude_author_id,
                )

            return await conn.fetch(
                """
                SELECT *
                FROM (
                    SELECT *
                    FROM messages
                    WHERE channel_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                ) recent
                ORDER BY created_at ASC
                """,
                channel_id,
                limit,
            )

    async def save_output_and_memory(
        self,
        scope_id: str,
        channel_id: str,
        channel_name: str,
        discord_message_id: str,
        attachment_url: str,
        prompt_used: str,
        image_bytes: bytes,
        memory: Optional[str],
        significance: int,
        keep_recent_images: int,
    ) -> Optional[int]:
        async with self.connection() as conn:
            output_id = await conn.fetchval(
                """
                INSERT INTO outputs(
                    channel_id, discord_message_id, attachment_url, prompt_used
                )
                VALUES($1, $2, $3, $4)
                RETURNING id
                """,
                channel_id,
                discord_message_id,
                attachment_url,
                prompt_used,
            )

            memory_id = None
            if memory and memory.strip():
                memory_id = await conn.fetchval(
                    """
                    INSERT INTO memories(
                        scope_id, channel_id, channel_name, content,
                        significance, source_output_id, kind, status
                    )
                    VALUES($1, $2, $3, $4, $5, $6, 'episode', 'active')
                    RETURNING id
                    """,
                    scope_id,
                    channel_id,
                    channel_name,
                    memory.strip(),
                    significance,
                    output_id,
                )

            await conn.execute(
                """
                INSERT INTO recent_images(channel_id, discord_message_id, image_bytes)
                VALUES($1, $2, $3)
                ON CONFLICT (discord_message_id) DO NOTHING
                """,
                channel_id,
                discord_message_id,
                image_bytes,
            )

            await conn.execute(
                """
                DELETE FROM recent_images
                WHERE channel_id = $1
                  AND id NOT IN (
                      SELECT id
                      FROM recent_images
                      WHERE channel_id = $1
                      ORDER BY created_at DESC
                      LIMIT $2
                  )
                """,
                channel_id,
                keep_recent_images,
            )
            return memory_id

    async def get_recent_image_bytes(self, channel_id: str, limit: int) -> List[bytes]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT image_bytes
                FROM recent_images
                WHERE channel_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                channel_id,
                limit,
            )

        # Oldest -> newest is more natural for visual chronology.
        return [bytes(row["image_bytes"]) for row in reversed(rows)]

    async def get_recent_memories(self, channel_id: str, limit: int) -> List[asyncpg.Record]:
        """Newest memories from the current channel only: local episodic continuity."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT id, channel_id, channel_name, content, significance, created_at
                FROM memories
                WHERE channel_id = $1
                  AND status = 'active'
                  AND kind = 'episode'
                ORDER BY created_at DESC
                LIMIT $2
                """,
                channel_id,
                limit,
            )

    async def get_scope_recent_memories(self, scope_id: str, limit: int) -> List[asyncpg.Record]:
        """Newest memories anywhere in this Discord server/DM scope. Used for admin inspection."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT id, channel_id, channel_name, content, significance, created_at
                FROM memories
                WHERE scope_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                scope_id,
                limit,
            )

    async def get_relevant_memories(
        self,
        scope_id: str,
        query: str,
        limit: int,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[asyncpg.Record]:
        """RAG memory is server-wide (or DM-scope-wide), not channel locked."""
        exclude_ids = exclude_ids or []

        async with self.pool.acquire() as conn:
            if query.strip():
                rows = await conn.fetch(
                    """
                    SELECT id, channel_id, channel_name, content, significance, created_at
                    FROM memories
                    WHERE scope_id = $1
                      AND status = 'active'
                      AND kind = 'episode'
                      AND NOT (id = ANY($4::bigint[]))
                    ORDER BY
                        CASE
                            WHEN to_tsvector('english', content)
                                 @@ websearch_to_tsquery('english', $2)
                            THEN 0
                            ELSE 1
                        END,
                        significance DESC,
                        created_at DESC
                    LIMIT $3
                    """,
                    scope_id,
                    query,
                    limit,
                    exclude_ids,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, channel_id, channel_name, content, significance, created_at
                    FROM memories
                    WHERE scope_id = $1
                      AND status = 'active'
                      AND kind = 'episode'
                      AND NOT (id = ANY($3::bigint[]))
                    ORDER BY significance DESC, created_at DESC
                    LIMIT $2
                    """,
                    scope_id,
                    limit,
                    exclude_ids,
                )

        return rows


# ============================================================
# OpenAI engine
# ============================================================

class CompanionEngine:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def generate_image(
        self,
        current_message_text: str,
        people_context_text: str,
        recent_background_text: str,
        durable_memory_text: str,
        recent_memory_text: str,
        retrieved_memory_text: str,
        reference_images: List[bytes],
        has_canonical_reference: bool,
        external_images: List[Dict[str, Any]],
    ) -> Tuple[bytes, str]:
        # OpenAI receives all visual inputs together, so label them clearly in
        # the prompt: SELF images belong to Image-san; CONVERSATION images were
        # posted by other participants and are things she is being shown.
        self_reference_count = len(reference_images)

        if reference_images:
            if has_canonical_reference:
                self_note = (
                    "The first SELF reference is your canonical self-reference. "
                    "Any additional SELF reference is a recent image you made. "
                    "Use SELF references for continuity of your own appearance or recent action, "
                    "not as a composition you must keep editing."
                )
            else:
                self_note = (
                    "Your SELF reference is a recent image you made. "
                    "Use it as visual memory of yourself and what you recently did, "
                    "NOT as a scene, pose, camera angle, or composition you must continue."
                )
        else:
            self_note = (
                "No SELF visual reference is available. You may establish or change "
                "your own visual presentation naturally."
            )

        external_lines: List[str] = []
        for offset, item in enumerate(external_images, start=1):
            overall_index = self_reference_count + offset
            author_kind = "another AI/bot" if item["is_bot"] else "a human"
            accompanying_text = (item["message_text"] or "").strip()
            if len(accompanying_text) > 500:
                accompanying_text = accompanying_text[:500] + "…"

            detail = (
                f"Reference image {overall_index} is a CONVERSATION IMAGE posted by "
                f"{item['author_name']} ({author_kind}). It is NOT one of your prior images "
                f"and must NOT be treated as your identity or visual memory."
            )
            if accompanying_text:
                detail += f' Their accompanying message was: "{accompanying_text}"'
            external_lines.append(detail)

        external_note = (
            "\n".join(external_lines)
            if external_lines
            else "No external conversation images are being shown on this turn."
        )

        reference_note = (
            f"SELF IMAGE GUIDANCE:\n{self_note}\n\n"
            f"EXTERNAL CONVERSATION IMAGE GUIDANCE:\n{external_note}\n\n"
            "External conversation images are things other people showed you. Look at them "
            "and respond to them when relevant. You may refer to, react to, transform, compare, "
            "discuss, or take inspiration from them, but do not assume the depicted person is you. "
            "The CURRENT MESSAGE still decides what matters most."
        )

        prompt = f"""
{COMPANION_CI}

CURRENT MESSAGE — RESPOND TO THIS
---------------------------------
{current_message_text}

PEOPLE RELEVANT TO THIS TURN
----------------------------
{people_context_text}

IMMEDIATE BACKGROUND — CONTEXT ONLY
-----------------------------------
{recent_background_text}

DURABLE MEMORY — STABLE / CONSOLIDATED
---------------------------------------
{durable_memory_text}

RECENT EPISODIC MEMORIES
------------------------
{recent_memory_text}

RETRIEVED MEMORIES (OLDER / RAG)
--------------------------------
{retrieved_memory_text}

VISUAL CONTEXT — SELF VS OTHER PEOPLE'S IMAGES
------------------------------------------------
{reference_note}

Create a NEW image that directly responds to the CURRENT MESSAGE.
The current message outranks the background, memories, and visual reference.
""".strip()

        kwargs: Dict[str, Any] = {
            "model": OPENAI_IMAGE_MODEL,
            "prompt": truncate(prompt),
        }
        if IMAGE_SIZE:
            kwargs["size"] = IMAGE_SIZE
        if IMAGE_QUALITY:
            kwargs["quality"] = IMAGE_QUALITY

        all_visual_inputs: List[Tuple[bytes, str]] = []

        for i, image_bytes in enumerate(reference_images, start=1):
            all_visual_inputs.append(
                (image_bytes, f"self_reference_{i}.png")
            )

        for i, item in enumerate(external_images, start=1):
            safe_author = "".join(
                ch if ch.isalnum() or ch in "-_" else "_"
                for ch in item["author_name"]
            )[:40] or "participant"

            all_visual_inputs.append(
                (
                    item["bytes"],
                    f"conversation_image_{i}_from_{safe_author}.png",
                )
            )

        if all_visual_inputs:
            files = [
                named_bytes_io(image_bytes, filename)
                for image_bytes, filename in all_visual_inputs
            ]
            result = await self.client.images.edit(image=files, **kwargs)
        else:
            result = await self.client.images.generate(**kwargs)

        if not result.data or not result.data[0].b64_json:
            raise RuntimeError("Image API returned no image data.")

        return base64.b64decode(result.data[0].b64_json), prompt

    async def summarize_image_to_memory(
        self,
        image_bytes: bytes,
        recent_messages_text: str,
        existing_memory_context: str,
    ) -> Dict[str, Any]:
        data_url = (
            "data:image/png;base64,"
            + base64.b64encode(image_bytes).decode("utf-8")
        )

        response = await self.client.responses.create(
            model=OPENAI_HELPER_MODEL,
            reasoning={"effort": "low"},
            text={
                "format": VISUAL_BRIDGE_JSON_SCHEMA,
                "verbosity": "low",
            },
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": VISUAL_BRIDGE_HELPER_SYSTEM}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "IMMEDIATE CONVERSATION CONTEXT:\n"
                                + recent_messages_text
                                + "\n\nEXISTING MEMORY CONTEXT — use this to avoid duplicates:\n"
                                + existing_memory_context
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": data_url,
                        },
                    ],
                },
            ],
        )

        parsed = json.loads(response.output_text)
        save = bool(parsed.get("save", True))
        memory = (parsed.get("memory") or "").strip()
        significance = safe_int(str(parsed.get("significance", 1)), 1, 1, 3)

        if not save:
            memory = ""
            significance = 1
        elif not memory:
            save = False
            significance = 1

        bridge_text = (parsed.get("bridge_text") or "").strip()
        if len(bridge_text) > VISUAL_BRIDGE_MAX_CHARS:
            bridge_text = bridge_text[:VISUAL_BRIDGE_MAX_CHARS - 1].rstrip() + "…"

        visible_mentions = []
        for raw_name in parsed.get("visible_mentions") or []:
            name = str(raw_name or "").strip().lstrip("@").strip()
            if name and name.casefold() not in {n.casefold() for n in visible_mentions}:
                visible_mentions.append(name)
            if len(visible_mentions) >= VISUAL_BRIDGE_MAX_MENTIONS:
                break

        return {
            "save": save,
            "memory": memory,
            "significance": significance,
            "bridge_text": bridge_text,
            "visible_mentions": visible_mentions,
        }


# ============================================================
# Discord bot
# ============================================================

storage = Storage(DATABASE_URL)
engine = CompanionEngine()
memory_manager = MemoryManager(
    storage=storage,
    client=engine.client,
    helper_model=OPENAI_HELPER_MODEL,
    companion_name=COMPANION_NAME,
    people_cards_path=PEOPLE_CARDS_PATH,
    consolidate_every=CONSOLIDATE_EVERY,
    consolidation_batch_size=CONSOLIDATION_BATCH_SIZE,
    durable_limit=DURABLE_MEMORY_LIMIT,
)
http_session: Optional[aiohttp.ClientSession] = None
channel_locks: Dict[int, asyncio.Lock] = {}
proactive_loop_task: Optional[asyncio.Task] = None

# Consecutive replies in each channel that were triggered by another bot/AI.
# Any human message in that channel resets this to zero.
ai_trigger_streaks: Dict[int, int] = {}
consolidation_locks: Dict[str, asyncio.Lock] = {}
initial_consolidation_task: Optional[asyncio.Task] = None


class ImageCompanionBot(discord.Client):
    async def setup_hook(self):
        global http_session
        await storage.connect()
        await storage.init_schema()
        conflicts = await memory_manager.initialize()
        if conflicts:
            print(f"Warning: duplicate Discord IDs in people cards: {conflicts}")
        print(f"Loaded {len(memory_manager.people_cache)} people cards.")
        http_session = aiohttp.ClientSession()
        print("Database, people cards and HTTP session ready.")

    async def close(self):
        global http_session
        if http_session and not http_session.closed:
            await http_session.close()
        await storage.close()
        await super().close()


intents = discord.Intents.default()
intents.message_content = True
bot = ImageCompanionBot(intents=intents)


async def fetch_bytes(url: str) -> bytes:
    if not http_session:
        raise RuntimeError("HTTP session is not ready.")
    async with http_session.get(url) as response:
        response.raise_for_status()
        return await response.read()


def _bridge_card_matches(name: str) -> List[Dict[str, Any]]:
    wanted = (name or "").strip().lstrip("@").casefold()
    if not wanted:
        return []

    matches = []
    for card in memory_manager.people_cache:
        candidates = [str(card.get("name") or "")]
        candidates.extend(str(x) for x in (card.get("aliases") or []))
        if any(c.strip().casefold() == wanted for c in candidates if c.strip()):
            matches.append(card)
    return matches


async def _resolve_bridge_bot_mentions(
    guild: Optional[discord.Guild],
    visible_mentions: List[str],
    *,
    skip_ids: Optional[set[int]] = None,
) -> List[Tuple[str, discord.abc.User]]:
    # Exact card match + actual Discord bot account only. No fuzzy tag guessing.
    if not VISUAL_BRIDGE_PING_BOTS or guild is None or VISUAL_BRIDGE_MAX_MENTIONS <= 0:
        return []

    skip_ids = skip_ids or set()
    resolved = []
    seen_ids = set()

    for raw_name in visible_mentions[:VISUAL_BRIDGE_MAX_MENTIONS]:
        matches = _bridge_card_matches(raw_name)
        if len(matches) != 1:
            continue

        raw_id = str(matches[0].get("discord_id") or "").strip()
        if not raw_id.isdigit():
            continue

        user_id = int(raw_id)
        if user_id in skip_ids or user_id in seen_ids:
            continue
        if bot.user and user_id == bot.user.id:
            continue

        member = guild.get_member(user_id)
        if member is None:
            try:
                member = await guild.fetch_member(user_id)
            except Exception:
                member = None

        if member is None or not getattr(member, "bot", False):
            continue

        seen_ids.add(user_id)
        resolved.append((raw_name, member))

    return resolved


async def post_visual_bridge(
    image_message: discord.Message,
    summary: Dict[str, Any],
    *,
    trigger_author_id: Optional[int] = None,
):
    """
    Small public accessibility transcript.

    It is authored by the image companion's Discord account, and that companion's
    recent-text retrieval excludes its own author ID. Therefore other AIs can read
    this message while the originating image companion cannot feed it back to itself.
    """
    if not VISUAL_BRIDGE_ENABLED:
        return

    rendered = (summary.get("bridge_text") or "").strip()
    if not rendered:
        return

    skip_ids = {trigger_author_id} if trigger_author_id else set()
    resolved = await _resolve_bridge_bot_mentions(
        image_message.guild,
        list(summary.get("visible_mentions") or []),
        skip_ids=skip_ids,
    )

    allowed_users = []
    for visible_name, member in resolved:
        mention = f"<@{member.id}>"
        pat = re.compile(r"@" + re.escape(visible_name) + r"\b", re.IGNORECASE)
        if pat.search(rendered):
            rendered = pat.sub(mention, rendered, count=1)
        else:
            rendered += f" {mention}"
        allowed_users.append(member)

    rendered = rendered.replace("\n", " ").strip()
    if len(rendered) > VISUAL_BRIDGE_MAX_CHARS:
        rendered = rendered[:VISUAL_BRIDGE_MAX_CHARS - 1].rstrip() + "…"

    # Discord subtext: low visual prominence for humans, ordinary text for LMs.
    content = f"-# [{COMPANION_NAME} visual transcript] {rendered}"

    allowed_mentions = (
        discord.AllowedMentions(users=allowed_users, replied_user=False)
        if allowed_users
        else discord.AllowedMentions.none()
    )

    try:
        await image_message.reply(
            content=content,
            mention_author=False,
            allowed_mentions=allowed_mentions,
        )
    except Exception as exc:
        print(f"Visual bridge post failed: {repr(exc)}")


async def build_reference_images(channel_id: str) -> Tuple[List[bytes], bool]:
    """Image-san's OWN visual memory only."""
    references: List[bytes] = []
    has_canonical = False

    if CANONICAL_REFERENCE_PATH:
        try:
            reference_path = Path(CANONICAL_REFERENCE_PATH)
            raw_reference = reference_path.read_bytes()
            references.append(
                normalize_image_for_api(raw_reference, reference_path.name)
            )
            has_canonical = True
            print(f"Loaded local canonical reference: {reference_path}")
        except Exception as exc:
            print(f"Warning: local canonical reference could not be loaded: {exc}")

    if not has_canonical and CANONICAL_REFERENCE_URL:
        try:
            raw_reference = await fetch_bytes(CANONICAL_REFERENCE_URL)
            references.append(
                normalize_image_for_api(raw_reference, CANONICAL_REFERENCE_URL)
            )
            has_canonical = True
            print("Loaded canonical reference from URL.")
        except Exception as exc:
            print(f"Warning: canonical reference URL could not be loaded: {exc}")

    references.extend(
        await storage.get_recent_image_bytes(channel_id, RECENT_IMAGE_TURNS)
    )

    return references, has_canonical


def attachment_is_image(attachment: discord.Attachment) -> bool:
    """Discord does not always populate content_type, so also check extension."""
    content_type = (attachment.content_type or "").lower()
    if content_type.startswith("image/"):
        return True

    filename = (attachment.filename or "").lower()
    return filename.endswith((
        ".png", ".jpg", ".jpeg", ".webp", ".gif",
        ".bmp", ".tif", ".tiff"
    ))


async def external_images_from_message(
    source_message: discord.Message,
) -> List[Dict[str, Any]]:
    """Read image attachments from one NON-Image-san Discord message."""
    if bot.user and source_message.author.id == bot.user.id:
        return []

    author_name = getattr(
        source_message.author,
        "display_name",
        source_message.author.name,
    )
    message_text = (source_message.content or "").strip()

    found: List[Dict[str, Any]] = []
    for attachment in source_message.attachments:
        if not attachment_is_image(attachment):
            continue

        try:
            raw_image_bytes = await attachment.read()
            image_bytes = normalize_image_for_api(
                raw_image_bytes,
                attachment.filename or "discord image",
            )
        except Exception as exc:
            print(
                f"Warning: skipping external image "
                f"{attachment.filename!r} from {author_name}: {exc}"
            )
            continue

        found.append({
            "bytes": image_bytes,
            "author_name": author_name,
            "author_id": str(source_message.author.id),
            "is_bot": bool(source_message.author.bot),
            "message_text": message_text,
            "filename": attachment.filename or "image",
            "discord_message_id": str(source_message.id),
        })

    return found


async def build_external_conversation_images(
    message: discord.Message,
) -> List[Dict[str, Any]]:
    """
    Get the newest images posted by humans/other AI in this exact channel/thread.

    CURRENT message is checked first, then earlier messages. Image-san's own
    attachments are excluded because her self-image memory is supplied separately.
    Returned order is oldest -> newest for natural visual chronology.
    """
    if RECENT_EXTERNAL_IMAGES <= 0:
        return []

    newest_first: List[Dict[str, Any]] = []

    # Always inspect the active message explicitly.
    newest_first.extend(await external_images_from_message(message))

    if len(newest_first) < RECENT_EXTERNAL_IMAGES:
        try:
            async for older in message.channel.history(
                limit=EXTERNAL_IMAGE_SCAN_MESSAGES,
                before=message,
            ):
                newest_first.extend(await external_images_from_message(older))

                if len(newest_first) >= RECENT_EXTERNAL_IMAGES:
                    break
        except Exception as exc:
            print(f"Warning: could not scan Discord history for external images: {exc}")

    selected = newest_first[:RECENT_EXTERNAL_IMAGES]
    return list(reversed(selected))


async def build_proactive_external_images(channel) -> List[Dict[str, Any]]:
    """Newest external images in a channel/thread for a proactive turn."""
    if RECENT_EXTERNAL_IMAGES <= 0:
        return []

    newest_first: List[Dict[str, Any]] = []

    try:
        async for older in channel.history(limit=EXTERNAL_IMAGE_SCAN_MESSAGES):
            newest_first.extend(await external_images_from_message(older))
            if len(newest_first) >= RECENT_EXTERNAL_IMAGES:
                break
    except Exception as exc:
        print(f"Warning: proactive external-image scan failed: {exc}")

    selected = newest_first[:RECENT_EXTERNAL_IMAGES]
    return list(reversed(selected))


async def directly_targets_image_san(message: discord.Message) -> bool:
    """True when a message @mentions Image-san or directly replies to her."""
    if not bot.user:
        return False

    # Normal Discord-resolved mention.
    if bot.user in message.mentions:
        return True

    # Fallback for bot/webhook messages that contain a literal mention in text
    # but do not populate message.mentions as expected.
    raw = message.content or ""
    if f"<@{bot.user.id}>" in raw or f"<@!{bot.user.id}>" in raw:
        return True

    # Direct reply to one of Image-san's messages.
    if message.reference:
        try:
            referenced = message.reference.resolved
            if referenced is None and message.reference.message_id:
                referenced = await message.channel.fetch_message(
                    message.reference.message_id
                )
            if referenced and referenced.author.id == bot.user.id:
                return True
        except Exception:
            pass

    return False


async def should_respond(message: discord.Message) -> bool:
    # Never respond to our own Discord messages.
    if bot.user and message.author.id == bot.user.id:
        return False

    directly_targeted = await directly_targets_image_san(message)

    # Other AI/bots may summon Image-san, but only by directly mentioning her
    # or replying to her. RESPOND_TO_ALL never makes her answer random bot chatter.
    if message.author.bot:
        return ALLOW_AI_TAGS and directly_targeted

    if isinstance(message.channel, discord.DMChannel):
        return RESPOND_IN_DMS

    if RESPOND_TO_ALL:
        return True

    return directly_targeted


async def handle_dbwhere_command(message: discord.Message) -> bool:
    content = (message.content or "").strip().lower()
    if content != "!dbwhere":
        return False

    if not is_command_owner(message):
        return True

    info = await storage.get_db_debug()
    await message.reply(
        (
            f"**{COMPANION_NAME} DB**\n"
            f"database: `{info['database_name']}`\n"
            f"schema: `{info['schema_name']}`\n"
            f"search_path: `{info['search_path']}`\n"
            f"memories visible here: `{info['memory_count']}`"
        ),
        mention_author=False,
    )
    return True


async def send_command_chunks(message: discord.Message, text: str):
    while text:
        chunk = text[:1900]
        if len(text) > 1900:
            split_at = chunk.rfind("\n\n")
            if split_at > 500:
                chunk = chunk[:split_at]
        await message.reply(chunk, mention_author=False)
        text = text[len(chunk):].lstrip()


async def handle_memory_admin_command(message: discord.Message) -> bool:
    content = (message.content or "").strip()
    if not content.startswith("!"):
        return False

    command, _, rest = content.partition(" ")
    command = command.lower()
    managed = {
        "!memories", "!memsearch", "!memedit", "!memarchive",
        "!memrestore", "!memdelete", "!durable", "!durableadd",
        "!durableedit", "!durabledelete", "!people", "!person",
        "!consolidate", "!memoryrepair", "!personreload",
    }
    if command not in managed:
        return False
    if not is_command_owner(message):
        return True

    scope_id = memory_scope_for(message)

    if command == "!memories":
        count = safe_int(rest.strip() or "20", 20, 1, 50)
        rows = await memory_manager.list_memories(scope_id, count)
        if not rows:
            await message.reply("No episodic memories yet.", mention_author=False)
            return True
        blocks = [
            f"**#{row['id']}** [{row['status']}] [sig {row['significance']}] "
            f"[#{row.get('channel_name') or '?'}]\n{row['content']}"
            for row in rows
        ]
        await send_command_chunks(message, "**EPISODIC MEMORIES**\n\n" + "\n\n".join(blocks))
        return True

    if command == "!memsearch":
        query = rest.strip()
        if not query:
            await message.reply("Usage: `!memsearch words`", mention_author=False)
            return True
        rows = await memory_manager.search_memories(scope_id, query, 20)
        if not rows:
            await message.reply("No matching memories.", mention_author=False)
            return True
        blocks = [
            f"**#{row['id']}** [{row['status']}] [sig {row['significance']}]\n{row['content']}"
            for row in rows
        ]
        await send_command_chunks(message, f"**MEMORY SEARCH: {query}**\n\n" + "\n\n".join(blocks))
        return True

    if command == "!memedit":
        bits = rest.split(maxsplit=1)
        if len(bits) != 2 or not bits[0].isdigit():
            await message.reply("Usage: `!memedit ID replacement text`", mention_author=False)
            return True
        ok = await memory_manager.edit_memory(scope_id, int(bits[0]), bits[1].strip())
        await message.reply("Edited." if ok else "Memory ID not found.", mention_author=False)
        return True

    if command in {"!memarchive", "!memrestore", "!memdelete"}:
        if not rest.strip().isdigit():
            await message.reply(f"Usage: `{command} ID`", mention_author=False)
            return True
        memory_id = int(rest.strip())
        if command == "!memarchive":
            ok = await memory_manager.set_memory_status(scope_id, memory_id, "archived", "manually archived")
        elif command == "!memrestore":
            ok = await memory_manager.set_memory_status(scope_id, memory_id, "active")
        else:
            ok = await memory_manager.delete_memory(scope_id, memory_id)
        await message.reply("Done." if ok else "Memory ID not found.", mention_author=False)
        return True

    if command == "!durable":
        rows = await memory_manager.get_durable_memories(scope_id)
        if not rows:
            await message.reply("No durable memories yet.", mention_author=False)
            return True
        blocks = []
        for row in rows:
            pin = " PINNED" if row.get("pinned") else ""
            blocks.append(f"**D#{row['id']}** [{row['category']}{pin}]\n{row['content']}")
        await send_command_chunks(message, "**DURABLE MEMORY**\n\n" + "\n\n".join(blocks))
        return True

    if command == "!durableadd":
        category, sep, durable_text = rest.partition("|")
        if not sep or not durable_text.strip():
            await message.reply("Usage: `!durableadd category | fact to preserve`", mention_author=False)
            return True
        durable_id = await memory_manager.add_durable(
            scope_id, category.strip() or "general", durable_text.strip(), pinned=True
        )
        await message.reply(f"Added pinned durable memory D#{durable_id}.", mention_author=False)
        return True

    if command == "!durableedit":
        bits = rest.split(maxsplit=1)
        if len(bits) != 2 or not bits[0].isdigit():
            await message.reply("Usage: `!durableedit ID replacement text`", mention_author=False)
            return True
        ok = await memory_manager.edit_durable(scope_id, int(bits[0]), bits[1].strip())
        await message.reply("Edited and pinned." if ok else "Durable ID not found.", mention_author=False)
        return True

    if command == "!durabledelete":
        if not rest.strip().isdigit():
            await message.reply("Usage: `!durabledelete ID`", mention_author=False)
            return True
        ok = await memory_manager.delete_durable(scope_id, int(rest.strip()))
        await message.reply("Deleted." if ok else "Durable ID not found.", mention_author=False)
        return True

    if command == "!people":
        names = ", ".join(card["name"] for card in memory_manager.people_cache)
        await send_command_chunks(message, f"**PEOPLE CARDS ({len(memory_manager.people_cache)})**\n{names}")
        return True

    if command == "!person":
        query = rest.strip()
        if not query:
            await message.reply("Usage: `!person name-or-discord-id`", mention_author=False)
            return True
        cards = await memory_manager.get_person(query)
        if not cards:
            await message.reply("No matching person card.", mention_author=False)
            return True
        await send_command_chunks(message, "**PERSON CARD**\n\n" + memory_manager.format_people_context(cards))
        return True

    if command == "!personreload":
        await memory_manager.reload_people_cards()
        await message.reply(
            f"Reloaded {len(memory_manager.people_cache)} cards from `{PEOPLE_CARDS_PATH}`.",
            mention_author=False,
        )
        return True

    if command == "!consolidate":
        await message.reply("Running a full memory consolidation pass…", mention_author=False)
        result = await consolidate_scope_safe(scope_id, force_full=True)
        await message.reply(f"Consolidation finished: `{result}`", mention_author=False)
        return True

    if command == "!memoryrepair":
        await message.reply(
            "Rebuilding automatic durable memory with identity-sanitisation. "
            "Pinned durable memories and original episodes will not be deleted.",
            mention_author=False,
        )
        cleared = await memory_manager.clear_automatic_durable(scope_id)
        result = await consolidate_scope_safe(scope_id, force_full=True)
        await message.reply(
            f"Memory repair finished. Cleared `{cleared}` automatic durable memories. "
            f"Repair result: `{result}`",
            mention_author=False,
        )
        return True

    return False


async def handle_memories_command(message: discord.Message) -> bool:
    content = (message.content or "").strip()
    if not content.lower().startswith("!memories"):
        return False

    if not is_command_owner(message):
        return True

    parts = content.split(maxsplit=1)
    count = 10
    if len(parts) == 2:
        count = safe_int(parts[1], 10, 1, 20)

    rows = await storage.get_scope_recent_memories(memory_scope_for(message), count)

    if not rows:
        await message.reply("No saved memories yet.", mention_author=False)
        return True

    blocks = []
    for i, row in enumerate(rows, start=1):
        timestamp = row["created_at"].strftime("%Y-%m-%d %H:%M UTC")
        channel_tag = row["channel_name"] or row["channel_id"]
        blocks.append(
            f"**{i}.** [#{channel_tag}] [significance {row['significance']}] {timestamp}\n{row['content']}"
        )

    text = "**Recent saved memories**\n\n" + "\n\n".join(blocks)

    # Discord message limit is 2000 characters. Split if needed.
    while text:
        chunk = text[:1900]
        if len(text) > 1900:
            split_at = chunk.rfind("\n\n")
            if split_at > 500:
                chunk = chunk[:split_at]
        await message.reply(chunk, mention_author=False)
        text = text[len(chunk):].lstrip()

    return True


async def handle_memory_context_command(message: discord.Message) -> bool:
    content = (message.content or "").strip().lower()
    if content != "!memorycontext":
        return False

    if not is_command_owner(message):
        return True

    channel_id = str(message.channel.id)
    scope_id = memory_scope_for(message)
    own_id = str(bot.user.id) if bot.user else None
    recent_rows = await storage.get_recent_messages(
        channel_id,
        RECENT_TEXT_MESSAGES,
        exclude_author_id=own_id,
    )
    query_text = await build_memory_query("", recent_rows)

    recent_memory_rows = await storage.get_recent_memories(
        channel_id,
        RECENT_MEMORIES,
    )
    recent_ids = [row["id"] for row in recent_memory_rows]
    retrieved_rows = await storage.get_relevant_memories(
        scope_id,
        query_text,
        MAX_RETRIEVED_MEMORIES,
        exclude_ids=recent_ids,
    )

    recent_text = format_memories(list(reversed(recent_memory_rows)))
    retrieved_text = format_memories(retrieved_rows, include_channel=True)
    durable_rows = await memory_manager.get_durable_memories(scope_id)
    durable_text = memory_manager.format_durable_memories(durable_rows)

    text = (
        f"**DURABLE MEMORIES ({len(durable_rows)})**\n{durable_text}\n\n"
        f"**RECENT EPISODES ({len(recent_memory_rows)})**\n{recent_text}\n\n"
        f"**RETRIEVED EPISODES / RAG ({len(retrieved_rows)})**\n{retrieved_text}"
    )

    while text:
        chunk = text[:1900]
        if len(text) > 1900:
            split_at = chunk.rfind("\n\n")
            if split_at > 500:
                chunk = chunk[:split_at]
        await message.reply(chunk, mention_author=False)
        text = text[len(chunk):].lstrip()

    return True


async def build_memory_query(
    current_message_text: str,
    background_rows: List[asyncpg.Record],
) -> str:
    # Retrieval should be driven primarily by the active turn, with only a tiny
    # amount of recent background to disambiguate it. This keeps enormous Discord
    # messages from making RAG chase an older conversational attractor.
    chunks = []

    current_message_text = (current_message_text or "").strip()
    if current_message_text:
        chunks.append(current_message_text[:2200])

    for row in background_rows[-2:]:
        content = memory_manager.resolve_mentions((row["content"] or "").strip())
        if content:
            chunks.append(content[:800])

    return " ".join(chunks)[:4000].strip()


async def consolidate_scope_safe(scope_id: str, force_full: bool = False):
    lock = consolidation_locks.setdefault(scope_id, asyncio.Lock())
    async with lock:
        try:
            result = await memory_manager.consolidate_scope(scope_id, force_full=force_full)
            if result.get("ran"):
                print(f"Memory consolidation for {scope_id}: {result}")
            return result
        except Exception as exc:
            print(f"Memory consolidation failed for {scope_id}: {repr(exc)}")
            return {"ran": False, "error": repr(exc)}


async def initial_consolidation_pass():
    scopes = await memory_manager.list_scopes()
    for scope_id in scopes:
        await consolidate_scope_safe(scope_id)


def schedule_consolidation(scope_id: str):
    if AUTO_CONSOLIDATE:
        asyncio.create_task(consolidate_scope_safe(scope_id))


def proactive_target_minute(local_date, channel_id: str) -> int:
    """
    Stable pseudo-random minute between :05 and :54.
    Changes each day and differs between channels, but survives restarts.
    """
    seed = f"{COMPANION_NAME}|{local_date.isoformat()}|{channel_id}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    return 5 + (int.from_bytes(digest[:2], "big") % 50)


async def send_proactive_turn(channel):
    channel_id = str(channel.id)
    scope_id = memory_scope_for_channel(channel)

    # Same text/memory machinery as a normal turn, except there is deliberately
    # no fake Discord user message.
    own_id = str(bot.user.id) if bot.user else None
    recent_rows = await storage.get_recent_messages(
        channel_id,
        RECENT_TEXT_MESSAGES,
        exclude_author_id=own_id,
    )
    recent_background_text = format_recent_messages(recent_rows)

    recent_memory_rows = await storage.get_recent_memories(
        channel_id,
        RECENT_MEMORIES,
    )
    recent_memory_ids = [row["id"] for row in recent_memory_rows]
    recent_memory_text = format_memories(list(reversed(recent_memory_rows)))
    durable_rows = await memory_manager.get_durable_memories(scope_id)
    durable_memory_text = memory_manager.format_durable_memories(durable_rows)


    proactive_current = (
        f"{COMPANION_NAME}: You have chosen to make an unsolicited post in this channel. "
        "Nobody has prompted you. Look at the recent conversation, your memories, and visual context, "
        "then decide what YOU want to communicate right now. It may be a reaction, question, callback, "
        "joke, scene, observation, invitation, artwork, tangent, or something entirely your own. "
        "Do not announce that this is scheduled, automated, or proactive. Do not merely summarize the chat."
    )

    query_text = await build_memory_query(proactive_current, recent_rows)
    retrieved_memory_rows = await storage.get_relevant_memories(
        memory_scope_for_channel(channel),
        query_text,
        MAX_RETRIEVED_MEMORIES,
        exclude_ids=recent_memory_ids,
    )
    retrieved_memory_text = format_memories(
        retrieved_memory_rows,
        include_channel=True,
    )

    reference_images, has_canonical = await build_reference_images(channel_id)
    external_images = await build_proactive_external_images(channel)

    recent_author_ids = [str(row["author_id"]) for row in recent_rows[-3:]]
    external_author_ids = [item["author_id"] for item in external_images]
    relevant_people = memory_manager.select_relevant_people(
        author_id=None,
        text=recent_background_text,
        extra_ids=recent_author_ids + external_author_ids,
        exclude_ids=[str(bot.user.id)] if bot.user else [],
    )
    people_context_text = memory_manager.format_people_context(relevant_people)

    async with channel.typing():
        image_bytes, prompt_used = await engine.generate_image(
            current_message_text=proactive_current,
            people_context_text=people_context_text,
            recent_background_text=recent_background_text,
            durable_memory_text=durable_memory_text,
            recent_memory_text=recent_memory_text,
            retrieved_memory_text=retrieved_memory_text,
            reference_images=reference_images,
            has_canonical_reference=has_canonical,
            external_images=external_images,
        )

    sent = await channel.send(
        file=discord.File(io.BytesIO(image_bytes), filename="reply.png"),
        allowed_mentions=discord.AllowedMentions.none(),
    )

    attachment_url = sent.attachments[0].url if sent.attachments else ""
    await storage.save_bot_placeholder(sent)

    try:
        helper_context = (
            "MEMORY OWNER: " + COMPANION_NAME + ". "
            "Write autobiographical memory only for this companion, not for another person depicted or mentioned.\n\n"
            "PROACTIVE SELF-INITIATED TURN:\n"
            + proactive_current
            + "\n\nPEOPLE RELEVANT TO THIS TURN:\n"
            + people_context_text
            + "\n\nRECENT CHANNEL CONTEXT:\n"
            + recent_background_text
        )
        existing_memory_context = (
            "DURABLE MEMORY:\n"
            + durable_memory_text
            + "\n\nRECENT EPISODES:\n"
            + recent_memory_text
            + "\n\nRETRIEVED EPISODES:\n"
            + retrieved_memory_text
        )
        summary = await engine.summarize_image_to_memory(
            image_bytes=image_bytes,
            recent_messages_text=helper_context,
            existing_memory_context=existing_memory_context,
        )
    except Exception as exc:
        print(f"Proactive memory helper failed: {repr(exc)}")
        summary = {
            "save": False,
            "memory": "",
            "significance": 1,
            "bridge_text": "",
            "visible_mentions": [],
        }

    await post_visual_bridge(
        sent,
        summary,
        trigger_author_id=None,
    )

    memory_id = await storage.save_output_and_memory(
        scope_id=scope_id,
        channel_id=channel_id,
        channel_name=getattr(channel, "name", channel_id),
        discord_message_id=str(sent.id),
        attachment_url=attachment_url,
        prompt_used=prompt_used,
        image_bytes=image_bytes,
        memory=summary["memory"] if summary.get("save") else None,
        significance=summary["significance"],
        keep_recent_images=RECENT_IMAGE_TURNS,
    )

    if memory_id:
        schedule_consolidation(scope_id)

    print(
        f"Proactive post sent in channel {channel_id} "
        f"(memory {'saved' if memory_id else 'skipped as duplicate/routine'})."
    )


def memory_scope_for_channel(channel) -> str:
    guild = getattr(channel, "guild", None)
    if guild is not None:
        return f"guild:{guild.id}"
    return f"dm:{channel.id}"


async def proactive_loop():
    await bot.wait_until_ready()

    try:
        timezone = ZoneInfo(PROACTIVE_TIMEZONE)
    except Exception:
        print(
            f"Warning: invalid PROACTIVE_TIMEZONE={PROACTIVE_TIMEZONE!r}; "
            "falling back to UTC."
        )
        timezone = ZoneInfo("UTC")

    if not PROACTIVE_HOURS:
        print("Proactive posting disabled: PROACTIVE_HOURS is empty.")
        return

    print(
        f"Proactive loop started for {len(PROACTIVE_CHANNEL_IDS)} channel(s) "
        f"in timezone {timezone.key}."
    )

    while not bot.is_closed():
        try:
            now = datetime.now(timezone)
            local_date = now.date()

            for index, channel_id_text in enumerate(PROACTIVE_CHANNEL_IDS):
                try:
                    channel_id = int(channel_id_text)
                except ValueError:
                    print(f"Warning: invalid proactive channel ID: {channel_id_text!r}")
                    continue

                due_hour = PROACTIVE_HOURS[index % len(PROACTIVE_HOURS)]
                due_minute = proactive_target_minute(local_date, channel_id_text)

                # Only post during this channel's assigned local hour. No catch-up
                # storm later in the day if Railway happened to be offline.
                if now.hour != due_hour or now.minute < due_minute:
                    continue

                channel = bot.get_channel(channel_id)
                if channel is None:
                    try:
                        channel = await bot.fetch_channel(channel_id)
                    except Exception as exc:
                        print(
                            f"Warning: proactive channel {channel_id} unavailable: {exc}"
                        )
                        continue

                claimed = await storage.claim_proactive_slot(
                    local_date,
                    channel_id_text,
                )
                if not claimed:
                    continue

                lock = channel_locks.setdefault(channel_id, asyncio.Lock())
                if lock.locked():
                    # Normal conversation has priority. Let proactive retry later
                    # in the same hour.
                    await storage.release_proactive_slot(
                        local_date,
                        channel_id_text,
                    )
                    continue

                try:
                    async with lock:
                        await send_proactive_turn(channel)
                except Exception as exc:
                    print(
                        f"Proactive turn failed in channel {channel_id}: {repr(exc)}"
                    )
                    await storage.release_proactive_slot(
                        local_date,
                        channel_id_text,
                    )

        except Exception as exc:
            print(f"Proactive loop error: {repr(exc)}")

        await asyncio.sleep(PROACTIVE_CHECK_SECONDS)


@bot.event
async def on_ready():
    global proactive_loop_task, initial_consolidation_task
    print(f"Logged in as {bot.user} ({bot.user.id if bot.user else 'unknown'})")
    if AUTO_CONSOLIDATE:
        if initial_consolidation_task is None or initial_consolidation_task.done():
            initial_consolidation_task = asyncio.create_task(initial_consolidation_pass())

    if PROACTIVE_ENABLED and PROACTIVE_CHANNEL_IDS:
        if proactive_loop_task is None or proactive_loop_task.done():
            proactive_loop_task = asyncio.create_task(proactive_loop())

    if not OWNER_DISCORD_ID:
        print("Warning: OWNER_DISCORD_ID is not set; admin memory commands are locked for everyone.")


@bot.event
async def on_message(message: discord.Message):
    # Debug/admin commands: do not add the command itself to companion context.
    if await handle_memory_admin_command(message):
        return
    if await handle_dbwhere_command(message):
        return
    if await handle_memories_command(message):
        return
    if await handle_memory_context_command(message):
        return

    await storage.save_message(message)

    channel_key = message.channel.id

    # Any human message in the channel breaks an AI-to-AI chain, even when that
    # human did not directly summon Image-san.
    if not message.author.bot:
        ai_trigger_streaks[channel_key] = 0

    if not await should_respond(message):
        return

    lock = channel_locks.setdefault(channel_key, asyncio.Lock())

    # Do not consume an AI-loop slot for a tag that arrives while she is already
    # busy generating and therefore cannot receive a reply.
    if lock.locked():
        return

    # Allow a maximum number of consecutive AI-triggered replies. Once reached,
    # stay silent until any human message appears in the channel.
    if message.author.bot:
        current_streak = ai_trigger_streaks.get(channel_key, 0)

        if AI_TAG_REPLY_LIMIT == 0 or current_streak >= AI_TAG_REPLY_LIMIT:
            print(
                f"Ignoring AI trigger in channel {channel_key}: "
                f"AI reply limit ({AI_TAG_REPLY_LIMIT}) reached."
            )
            return

        ai_trigger_streaks[channel_key] = current_streak + 1

    async with lock:
        try:
            await handle_turn(message)
        except Exception as exc:
            print(f"Error during turn: {repr(exc)}")
            try:
                await message.reply(
                    "Image creature tripped over its own tail. Try me again in a moment.",
                    mention_author=False,
                )
            except Exception:
                pass


async def handle_turn(message: discord.Message):
    channel_id = str(message.channel.id)
    scope_id = memory_scope_for(message)
    channel_name = channel_label_for(message)

    # Separate the active message from recent conversation so the image model
    # knows exactly what it is replying to. RECENT_TEXT_MESSAGES is the whole
    # external text window including the active turn; the rest becomes background.
    own_id = str(bot.user.id) if bot.user else None
    recent_rows = await storage.get_recent_messages(
        channel_id,
        RECENT_TEXT_MESSAGES,
        exclude_author_id=own_id,
    )

    current_message_text = format_current_message(message)
    recent_background_rows = [
        row for row in recent_rows
        if str(row["discord_message_id"]) != str(message.id)
    ]
    recent_background_text = format_recent_messages(recent_background_rows)

    # Give her a small chronological run of the newest memories first.
    recent_memory_rows = await storage.get_recent_memories(
        channel_id,
        RECENT_MEMORIES,
    )
    recent_memory_ids = [row["id"] for row in recent_memory_rows]
    recent_memory_text = format_memories(list(reversed(recent_memory_rows)))
    durable_rows = await memory_manager.get_durable_memories(scope_id)
    durable_memory_text = memory_manager.format_durable_memories(durable_rows)


    # Then retrieve older relevant memories, excluding anything already present
    # in the recent-memory block so the two sections remain genuinely distinct.
    query_text = await build_memory_query(current_message_text, recent_background_rows)
    retrieved_memory_rows = await storage.get_relevant_memories(
        scope_id,
        query_text,
        MAX_RETRIEVED_MEMORIES,
        exclude_ids=recent_memory_ids,
    )
    retrieved_memory_text = format_memories(retrieved_memory_rows, include_channel=True)

    reference_images, has_canonical = await build_reference_images(channel_id)

    # Two distinct visual streams:
    # reference_images = Image-san's OWN recent visual memory
    # external_images = newest images humans/other AI have shown in this channel/thread
    external_images = await build_external_conversation_images(message)

    relevant_people = memory_manager.select_relevant_people(
        author_id=str(message.author.id),
        text=message.content or "",
        extra_ids=[item["author_id"] for item in external_images],
        exclude_ids=[str(bot.user.id)] if bot.user else [],
    )
    people_context_text = memory_manager.format_people_context(relevant_people)

    if external_images:
        print(
            "External conversation images this turn: "
            + ", ".join(
                f"{item['author_name']}:{item['filename']}"
                for item in external_images
            )
        )

    async with message.channel.typing():
        image_bytes, prompt_used = await engine.generate_image(
            current_message_text=current_message_text,
            people_context_text=people_context_text,
            recent_background_text=recent_background_text,
            durable_memory_text=durable_memory_text,
            recent_memory_text=recent_memory_text,
            retrieved_memory_text=retrieved_memory_text,
            reference_images=reference_images,
            has_canonical_reference=has_canonical,
            external_images=external_images,
        )

    # Human-triggered replies stay image-only and do not ping.
    #
    # AI/bot-triggered replies explicitly @mention the AI that summoned Image-san.
    # This is intentional: many companion bots only wake up when they see a direct
    # mention and will not treat a bare Discord reply-reference as a trigger.
    if message.author.bot:
        reply_content = f"<@{message.author.id}>"
        reply_allowed_mentions = discord.AllowedMentions(
            users=[message.author],
            replied_user=False,
        )
    else:
        reply_content = None
        reply_allowed_mentions = discord.AllowedMentions.none()

    sent = await message.reply(
        content=reply_content,
        file=discord.File(io.BytesIO(image_bytes), filename="reply.png"),
        mention_author=False,
        allowed_mentions=reply_allowed_mentions,
    )

    attachment_url = sent.attachments[0].url if sent.attachments else ""

    # Save a neutral placeholder to recent text context; the helper's memory stays
    # in the separate MEMORY system, not in the visible reply or immediate text.
    await storage.save_bot_placeholder(sent)

    try:
        helper_context = (
            "MEMORY OWNER: " + COMPANION_NAME + ". "
            "Write autobiographical memory only for this companion, not for another person depicted or mentioned.\n\n"
            "CURRENT MESSAGE:\n"
            + current_message_text
            + "\n\nPEOPLE RELEVANT TO THIS TURN:\n"
            + people_context_text
            + "\n\nIMMEDIATE BACKGROUND:\n"
            + recent_background_text
        )
        existing_memory_context = (
            "DURABLE MEMORY:\n"
            + durable_memory_text
            + "\n\nRECENT EPISODES:\n"
            + recent_memory_text
            + "\n\nRETRIEVED EPISODES:\n"
            + retrieved_memory_text
        )
        summary = await engine.summarize_image_to_memory(
            image_bytes=image_bytes,
            recent_messages_text=helper_context,
            existing_memory_context=existing_memory_context,
        )
    except Exception as exc:
        print(f"Memory helper failed: {repr(exc)}")
        summary = {
            "save": False,
            "memory": "",
            "significance": 1,
            "bridge_text": "",
            "visible_mentions": [],
        }

    await post_visual_bridge(
        sent,
        summary,
        trigger_author_id=message.author.id if message.author.bot else None,
    )

    memory_id = await storage.save_output_and_memory(
        scope_id=scope_id,
        channel_id=channel_id,
        channel_name=channel_name,
        discord_message_id=str(sent.id),
        attachment_url=attachment_url,
        prompt_used=prompt_used,
        image_bytes=image_bytes,
        memory=summary["memory"] if summary.get("save") else None,
        significance=summary["significance"],
        keep_recent_images=RECENT_IMAGE_TURNS,
    )

    if memory_id:
        print(
            f"Saved episodic memory #{memory_id} for {scope_id} / channel {channel_id} "
            f"(significance {summary['significance']}): {summary['memory']}"
        )
        schedule_consolidation(scope_id)
    else:
        print(f"Skipped duplicate/routine memory for {scope_id} / channel {channel_id}.")


async def main():
    await bot.start(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
