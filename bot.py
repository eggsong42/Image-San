import os
import io
import json
import base64
import asyncio
from typing import Optional, List, Dict, Any, Tuple

import aiohttp
import asyncpg
import discord
from dotenv import load_dotenv
from openai import AsyncOpenAI

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
COMPANION_NAME = os.getenv("COMPANION_NAME", "Image-san").strip()

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


# ============================================================
# Utilities
# ============================================================

def named_bytes_io(data: bytes, name: str) -> io.BytesIO:
    bio = io.BytesIO(data)
    bio.name = name
    bio.seek(0)
    return bio


def truncate(text: str, max_chars: int = 24000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[older text truncated]"


def format_recent_messages(rows: List[asyncpg.Record]) -> str:
    if not rows:
        return "(no recent text)"

    lines = []
    for row in rows:
        # Image-san's own messages are excluded before this formatter is called.
        # Other Discord bots/AI should keep their real names rather than being
        # mistaken for Image-san merely because Discord marks them as bots.
        speaker = row["author_name"] or "Unknown"
        content = (row["content"] or "").strip() or "[no text]"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def format_current_message(message: discord.Message) -> str:
    speaker = getattr(message.author, "display_name", message.author.name)
    content = (message.content or "").strip() or "[no text]"
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

    async def init_schema(self):
        with open("schema.sql", "r", encoding="utf-8") as f:
            sql = f.read()
        async with self.pool.acquire() as conn:
            await conn.execute(sql)

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
        memory: str,
        significance: int,
        keep_recent_images: int,
    ):
        async with self.pool.acquire() as conn:
            async with conn.transaction():
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

                await conn.execute(
                    """
                    INSERT INTO memories(
                        scope_id, channel_id, channel_name, content, significance, source_output_id
                    )
                    VALUES($1, $2, $3, $4, $5, $6)
                    """,
                    scope_id,
                    channel_id,
                    channel_name,
                    memory,
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

                # Keep only the newest N image blobs as visual working memory.
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
        recent_background_text: str,
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

IMMEDIATE BACKGROUND — CONTEXT ONLY
-----------------------------------
{recent_background_text}

RECENT MEMORIES
---------------
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
    ) -> Dict[str, Any]:
        data_url = (
            "data:image/png;base64,"
            + base64.b64encode(image_bytes).decode("utf-8")
        )

        response = await self.client.responses.create(
            model=OPENAI_HELPER_MODEL,
            reasoning={"effort": "low"},
            text={
                "format": MEMORY_JSON_SCHEMA,
                "verbosity": "low",
            },
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": MEMORY_COMPRESSOR_SYSTEM}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Immediate conversation context follows. Use it only "
                                "to identify factual continuity and obvious interaction facts:\n\n"
                                + recent_messages_text
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
        memory = (parsed.get("memory") or "").strip()
        significance = safe_int(str(parsed.get("significance", 1)), 1, 1, 3)

        if not memory:
            memory = "I replied with an image."

        return {
            "memory": memory,
            "significance": significance,
        }


# ============================================================
# Discord bot
# ============================================================

storage = Storage(DATABASE_URL)
engine = CompanionEngine()
http_session: Optional[aiohttp.ClientSession] = None
channel_locks: Dict[int, asyncio.Lock] = {}

# Consecutive replies in each channel that were triggered by another bot/AI.
# Any human message in that channel resets this to zero.
ai_trigger_streaks: Dict[int, int] = {}


class ImageCompanionBot(discord.Client):
    async def setup_hook(self):
        global http_session
        await storage.connect()
        await storage.init_schema()
        http_session = aiohttp.ClientSession()
        print("Database and HTTP session ready.")

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


async def build_reference_images(channel_id: str) -> Tuple[List[bytes], bool]:
    """Image-san's OWN visual memory only."""
    references: List[bytes] = []
    has_canonical = False

    if CANONICAL_REFERENCE_URL:
        try:
            references.append(await fetch_bytes(CANONICAL_REFERENCE_URL))
            has_canonical = True
        except Exception as exc:
            print(f"Warning: canonical reference could not be loaded: {exc}")

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
            image_bytes = await attachment.read()
        except Exception as exc:
            print(
                f"Warning: could not read external image "
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

    text = (
        f"**RECENT MEMORIES ({len(recent_memory_rows)})**\n{recent_text}\n\n"
        f"**RETRIEVED MEMORIES / RAG ({len(retrieved_rows)})**\n{retrieved_text}"
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
        content = (row["content"] or "").strip()
        if content:
            chunks.append(content[:800])

    return " ".join(chunks)[:4000].strip()


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} ({bot.user.id if bot.user else 'unknown'})")
    if not OWNER_DISCORD_ID:
        print("Warning: OWNER_DISCORD_ID is not set; admin memory commands are locked for everyone.")


@bot.event
async def on_message(message: discord.Message):
    # Debug/admin command: do not add the command itself to companion context.
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
            recent_background_text=recent_background_text,
            recent_memory_text=recent_memory_text,
            retrieved_memory_text=retrieved_memory_text,
            reference_images=reference_images,
            has_canonical_reference=has_canonical,
            external_images=external_images,
        )

    # The public companion response is the image only.
    sent = await message.reply(
        file=discord.File(io.BytesIO(image_bytes), filename="reply.png"),
        mention_author=False,
        allowed_mentions=discord.AllowedMentions.none(),
    )

    attachment_url = sent.attachments[0].url if sent.attachments else ""

    # Save a neutral placeholder to recent text context; the helper's memory stays
    # in the separate MEMORY system, not in the visible reply or immediate text.
    await storage.save_bot_placeholder(sent)

    try:
        helper_context = (
            "CURRENT MESSAGE:\n"
            + current_message_text
            + "\n\nIMMEDIATE BACKGROUND:\n"
            + recent_background_text
        )
        summary = await engine.summarize_image_to_memory(
            image_bytes=image_bytes,
            recent_messages_text=helper_context,
        )
    except Exception as exc:
        print(f"Memory helper failed: {repr(exc)}")
        summary = {
            "memory": "I replied with an image.",
            "significance": 1,
        }

    await storage.save_output_and_memory(
        scope_id=scope_id,
        channel_id=channel_id,
        channel_name=channel_name,
        discord_message_id=str(sent.id),
        attachment_url=attachment_url,
        prompt_used=prompt_used,
        image_bytes=image_bytes,
        memory=summary["memory"],
        significance=summary["significance"],
        keep_recent_images=RECENT_IMAGE_TURNS,
    )

    print(
        f"Saved memory for {scope_id} / channel {channel_id} "
        f"(significance {summary['significance']}): {summary['memory']}"
    )


async def main():
    await bot.start(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
