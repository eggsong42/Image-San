import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MEMORY_ARCHITECTURE_SQL = """
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'episode';
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS archived_reason TEXT;
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS consolidated_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_memories_scope_status_kind_created
    ON memories(scope_id, status, kind, created_at DESC);

CREATE TABLE IF NOT EXISTS durable_memories (
    id BIGSERIAL PRIMARY KEY,
    scope_id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    content TEXT NOT NULL,
    pinned BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_durable_memories_scope
    ON durable_memories(scope_id, pinned DESC, updated_at DESC);

CREATE TABLE IF NOT EXISTS people_cards (
    name_key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    discord_id TEXT,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    appearance TEXT NOT NULL DEFAULT '',
    relationship TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_people_cards_discord_id
    ON people_cards(discord_id);

CREATE TABLE IF NOT EXISTS memory_consolidation_state (
    scope_id TEXT PRIMARY KEY,
    initial_done BOOLEAN NOT NULL DEFAULT FALSE,
    last_processed_id BIGINT NOT NULL DEFAULT 0,
    last_run_at TIMESTAMPTZ
);
"""

MEMORY_SAVE_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "image_companion_memory",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "save": {"type": "boolean"},
            "memory": {"type": "string"},
            "significance": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["save", "memory", "significance"],
        "additionalProperties": False,
    },
}

CONSOLIDATION_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "image_companion_memory_consolidation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "automatic_durable_memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["category", "content"],
                    "additionalProperties": False,
                },
            },
            "archive_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["automatic_durable_memories", "archive_ids"],
        "additionalProperties": False,
    },
}

MEMORY_ARCHIVIST_SYSTEM = """
You are the factual autobiographical memory archivist for an image-native companion.

The latest generated image is the companion's reply. Write a concise FIRST-PERSON episodic memory from the companion's perspective only when the turn adds something worth remembering.

You will also receive a small EXISTING MEMORY CONTEXT containing recent episodic memories and durable memories.

RETURN save=false when the new image merely repeats information already represented there, especially stable appearance facts, recurring clothing, a familiar room, or a known preference with no meaningful new event. Repetition alone is not a new memory.

RETURN save=true when there is a distinct event, social interaction, new fact, meaningful reaction, changed preference, promise, milestone, callback with new consequence, new person, or a deliberate written request in the image to remember something.

Observable facts only. Do not invent symbolism, hidden motives, psychological interpretation, or literary analysis. Do not infer emotion from colour, weather, lighting, or composition alone. Emotion words are fine when clearly supported by expression/body language, image text, or immediate conversation.

Include important visible written text faithfully when useful. If the image explicitly asks for something to be remembered, treat that as high priority.

When save=false, return memory="" and significance=1.

Significance:
1 = routine but genuinely new episode
2 = notable callback, distinctive choice, new recurring element, or useful social fact
3 = major new person, relationship change, promise, milestone, or strong continuity event
""".strip()

CONSOLIDATION_SYSTEM = """
You maintain compact long-term memory for an image-native companion.

You receive:
1. PINNED DURABLE MEMORIES: human-curated facts. Never rewrite, remove, duplicate, or contradict these.
2. AUTOMATIC DURABLE MEMORIES: the current compact semantic memory.
3. EPISODIC MEMORIES: factual first-person events with stable numeric IDs.

Produce:
- automatic_durable_memories: the COMPLETE automatic durable set after integrating this batch. Keep it compact (ideally 4-12 items). Merge repeated stable facts, identity/appearance patterns, preferences, recurring relationships, ongoing projects, and durable social knowledge. Do not turn every event into a durable fact.
- archive_ids: episodic IDs whose useful information is now fully represented by durable memory OR which are simple redundant repetition. Do NOT archive a unique meaningful event just because it influenced a durable fact.

Rules:
- Preserve uncertainty where the evidence is weak.
- Prefer repeated evidence or high-significance episodes before making a stable durable fact.
- Newer repeated evidence may update an older automatic durable fact.
- Do not invent motives, symbolism, diagnoses, or hidden psychology.
- Do not archive distinctive events, promises, first meetings, relationship changes, milestones, or memorable one-off interactions unless they are genuinely redundant copies.
- Pinned durable memories are outside your editable set and must remain untouched.
- Avoid duplicate automatic durable memories.

IDENTITY SANITATION:
- Historical memory may contain cross-companion contamination from an older database configuration.
- SELF IDENTITY below is authoritative for whose autobiographical memory this is.
- If an episodic memory uses first-person language to claim the companion IS another named person/companion, or assigns the self an obviously incompatible identity/relationship (for example "I am Jett" inside Image-san's memory, or "Image-san is my creator" inside Image-san's own autobiographical memory), treat that episode as FOREIGN CONTAMINATION.
- Foreign-contamination episodes must NEVER become durable memory. Put their IDs in archive_ids.
- A legitimate memory ABOUT another person is not foreign contamination when the companion is clearly the observer/participant (for example "I met Jett" or "I drew Megan").
- Existing AUTOMATIC DURABLE MEMORIES may themselves be contaminated. Omit any automatic durable item that clearly belongs to another person's first-person identity.
- Appearance can legitimately vary, so do not mark a memory foreign solely because clothing, form, or art style differs. Use names, first-person identity claims, relationships, creator/partner roles, and strong contextual contradictions as the main evidence.
""".strip()


CARD_HEADERS = {
    "name": "name",
    "discord id": "discord_id",
    "aliases": "aliases",
    "appearance": "appearance",
    "relationship": "relationship",
    "notes": "notes",
}


def parse_people_cards_text(text: str) -> List[Dict[str, Any]]:
    blocks = [
        block.strip()
        for block in re.split(r"^\s*---\s*$", text, flags=re.MULTILINE)
        if block.strip()
    ]
    cards: List[Dict[str, Any]] = []

    for block in blocks:
        card: Dict[str, Any] = {
            "name": "",
            "discord_id": "",
            "aliases": [],
            "appearance": "",
            "relationship": "",
            "notes": "",
        }
        current: Optional[str] = None

        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            match = re.match(
                r"^(Name|Discord ID|Aliases|Appearance|Relationship|Notes)\s*:?\s*(.*)$",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                header = match.group(1).lower()
                current = CARD_HEADERS[header]
                value = match.group(2).strip()

                if current == "aliases":
                    if value and value != "-":
                        card["aliases"] = [
                            part.strip()
                            for part in value.split(",")
                            if part.strip() and part.strip() != "-"
                        ]
                elif value:
                    card[current] = value
                continue

            # Several supplied cards use a bare first line as the person's name.
            if not card["name"] and current is None:
                card["name"] = line
                current = "name"
                continue

            if current == "aliases":
                if line != "-":
                    card["aliases"].extend(
                        part.strip()
                        for part in line.split(",")
                        if part.strip() and part.strip() != "-"
                    )
            elif current:
                if card[current]:
                    card[current] += "\n" + line
                else:
                    card[current] = line

        card["name"] = card["name"].strip()
        card["discord_id"] = re.sub(r"\s+", "", card["discord_id"])
        card["aliases"] = list(dict.fromkeys(card["aliases"]))

        if card["name"]:
            card["name_key"] = card["name"].casefold()
            cards.append(card)

    return cards


def parse_people_cards_file(path: str) -> List[Dict[str, Any]]:
    return parse_people_cards_text(Path(path).read_text(encoding="utf-8"))


def id_conflicts(cards: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for card in cards:
        discord_id = str(card.get("discord_id") or "").strip()
        if discord_id:
            grouped.setdefault(discord_id, []).append(card["name"])
    return {
        discord_id: names
        for discord_id, names in grouped.items()
        if len(names) > 1
    }


class MemoryManager:
    def __init__(
        self,
        storage,
        client,
        helper_model: str,
        companion_name: str,
        people_cards_path: str = "people_cards.txt",
        consolidate_every: int = 20,
        consolidation_batch_size: int = 24,
        durable_limit: int = 12,
    ):
        self.storage = storage
        self.client = client
        self.helper_model = helper_model
        self.companion_name = companion_name
        self.people_cards_path = people_cards_path
        self.consolidate_every = max(5, consolidate_every)
        self.consolidation_batch_size = max(8, consolidation_batch_size)
        self.durable_limit = max(4, durable_limit)
        self.people_cache: List[Dict[str, Any]] = []

    async def initialize(self) -> Dict[str, List[str]]:
        async with self.storage.connection() as conn:
            await conn.execute(MEMORY_ARCHITECTURE_SQL)

        await self.reload_people_cards()
        return id_conflicts(self.people_cache)

    async def reload_people_cards(self):
        cards = parse_people_cards_file(self.people_cards_path)
        async with self.storage.connection() as conn:
            for card in cards:
                await conn.execute(
                    """
                    INSERT INTO people_cards(
                        name_key, name, discord_id, aliases,
                        appearance, relationship, notes, updated_at
                    )
                    VALUES($1, $2, $3, $4::text[], $5, $6, $7, NOW())
                    ON CONFLICT (name_key) DO UPDATE SET
                        name = EXCLUDED.name,
                        discord_id = EXCLUDED.discord_id,
                        aliases = EXCLUDED.aliases,
                        appearance = EXCLUDED.appearance,
                        relationship = EXCLUDED.relationship,
                        notes = EXCLUDED.notes,
                        updated_at = NOW()
                    """,
                    card["name_key"],
                    card["name"],
                    card["discord_id"] or None,
                    card["aliases"],
                    card["appearance"],
                    card["relationship"],
                    card["notes"],
                )

            rows = await conn.fetch(
                """
                SELECT name_key, name, discord_id, aliases,
                       appearance, relationship, notes
                FROM people_cards
                ORDER BY name
                """
            )
        self.people_cache = [dict(row) for row in rows]

    def _cards_by_id(self) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        for card in self.people_cache:
            discord_id = str(card.get("discord_id") or "")
            if discord_id:
                result.setdefault(discord_id, []).append(card)
        return result

    def unique_card_for_id(self, discord_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not discord_id:
            return None
        matches = self._cards_by_id().get(str(discord_id), [])
        if len(matches) == 1:
            return matches[0]
        return None

    def canonical_name(self, discord_id: Optional[str], fallback: str) -> str:
        card = self.unique_card_for_id(discord_id)
        return card["name"] if card else fallback

    def resolve_mentions(
        self,
        text: str,
        live_names: Optional[Dict[str, str]] = None,
    ) -> str:
        live_names = live_names or {}
        by_id = self._cards_by_id()

        def repl(match):
            discord_id = match.group(1)
            cards = by_id.get(discord_id, [])
            if len(cards) == 1:
                return f"@{cards[0]['name']}"
            if discord_id in live_names:
                return f"@{live_names[discord_id]}"
            if len(cards) > 1:
                return f"@DiscordUser({discord_id}; ambiguous person card)"
            return f"@DiscordUser({discord_id})"

        return re.sub(r"<@!?(\d+)>", repl, text or "")

    def select_relevant_people(
        self,
        author_id: Optional[str],
        text: str,
        extra_ids: Optional[Sequence[str]] = None,
        max_cards: int = 6,
        exclude_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        selected_keys = set()
        excluded = {str(value) for value in (exclude_ids or []) if value is not None}

        def add(card: Optional[Dict[str, Any]]):
            if not card:
                return
            if str(card.get("discord_id") or "") in excluded:
                return
            key = card["name_key"]
            if key not in selected_keys and len(selected) < max_cards:
                selected.append(card)
                selected_keys.add(key)

        add(self.unique_card_for_id(author_id))

        raw_ids = re.findall(r"<@!?(\d+)>", text or "")
        for discord_id in list(raw_ids) + list(extra_ids or []):
            add(self.unique_card_for_id(str(discord_id)))

        lower_text = (text or "").casefold()
        # Name / alias matching is deliberately conservative and whole-token-ish.
        for card in self.people_cache:
            candidates = [card["name"]] + list(card.get("aliases") or [])
            for candidate in candidates:
                candidate = (candidate or "").strip()
                if not candidate or candidate == "-":
                    continue
                escaped = re.escape(candidate.casefold())
                if re.search(rf"(?<!\w){escaped}(?!\w)", lower_text):
                    add(card)
                    break

        return selected

    def format_people_context(self, cards: Sequence[Dict[str, Any]]) -> str:
        if not cards:
            return "(no person cards are specifically relevant to this turn)"

        blocks = []
        for card in cards:
            aliases = ", ".join(card.get("aliases") or []) or "-"
            blocks.append(
                f"{card['name']} [Discord ID: {card.get('discord_id') or 'unknown'}]\n"
                f"Aliases: {aliases}\n"
                f"Appearance: {card.get('appearance') or '-'}\n"
                f"Relationship: {card.get('relationship') or '-'}\n"
                f"Notes: {card.get('notes') or '-'}"
            )
        return "\n\n".join(blocks)

    async def get_durable_memories(self, scope_id: str) -> List[Dict[str, Any]]:
        async with self.storage.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, category, content, pinned, created_at, updated_at
                FROM durable_memories
                WHERE scope_id = $1
                ORDER BY pinned DESC, updated_at DESC, id ASC
                LIMIT $2
                """,
                scope_id,
                self.durable_limit,
            )
        return [dict(row) for row in rows]

    def format_durable_memories(self, rows: Sequence[Dict[str, Any]]) -> str:
        if not rows:
            return "(no durable memories yet)"
        lines = []
        for row in rows:
            pin = " [PINNED]" if row.get("pinned") else ""
            lines.append(f"- {row.get('category', 'general')}{pin}: {row['content']}")
        return "\n".join(lines)

    async def list_memories(self, scope_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        async with self.storage.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, channel_name, content, significance,
                       status, kind, created_at, archived_reason
                FROM memories
                WHERE scope_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                scope_id,
                limit,
            )
        return [dict(row) for row in rows]

    async def search_memories(
        self,
        scope_id: str,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        async with self.storage.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, channel_name, content, significance,
                       status, kind, created_at, archived_reason
                FROM memories
                WHERE scope_id = $1
                  AND content ILIKE '%' || $2 || '%'
                ORDER BY created_at DESC
                LIMIT $3
                """,
                scope_id,
                query,
                limit,
            )
        return [dict(row) for row in rows]

    async def edit_memory(self, scope_id: str, memory_id: int, content: str) -> bool:
        async with self.storage.connection() as conn:
            result = await conn.execute(
                """
                UPDATE memories
                SET content = $3
                WHERE scope_id = $1 AND id = $2
                """,
                scope_id,
                memory_id,
                content,
            )
        return result.endswith("1")

    async def set_memory_status(
        self,
        scope_id: str,
        memory_id: int,
        status: str,
        reason: Optional[str] = None,
    ) -> bool:
        if status not in {"active", "archived"}:
            raise ValueError("status must be active or archived")
        async with self.storage.connection() as conn:
            result = await conn.execute(
                """
                UPDATE memories
                SET status = $3,
                    archived_reason = CASE WHEN $3 = 'archived' THEN $4 ELSE NULL END,
                    consolidated_at = CASE WHEN $3 = 'archived' THEN NOW() ELSE NULL END
                WHERE scope_id = $1 AND id = $2
                """,
                scope_id,
                memory_id,
                status,
                reason,
            )
        return result.endswith("1")

    async def delete_memory(self, scope_id: str, memory_id: int) -> bool:
        async with self.storage.connection() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE scope_id = $1 AND id = $2",
                scope_id,
                memory_id,
            )
        return result.endswith("1")

    async def add_durable(
        self,
        scope_id: str,
        category: str,
        content: str,
        pinned: bool = True,
    ) -> int:
        async with self.storage.connection() as conn:
            return await conn.fetchval(
                """
                INSERT INTO durable_memories(scope_id, category, content, pinned)
                VALUES($1, $2, $3, $4)
                RETURNING id
                """,
                scope_id,
                category or "general",
                content,
                pinned,
            )

    async def edit_durable(
        self,
        scope_id: str,
        durable_id: int,
        content: str,
    ) -> bool:
        async with self.storage.connection() as conn:
            result = await conn.execute(
                """
                UPDATE durable_memories
                SET content = $3, pinned = TRUE, updated_at = NOW()
                WHERE scope_id = $1 AND id = $2
                """,
                scope_id,
                durable_id,
                content,
            )
        return result.endswith("1")

    async def delete_durable(self, scope_id: str, durable_id: int) -> bool:
        async with self.storage.connection() as conn:
            result = await conn.execute(
                "DELETE FROM durable_memories WHERE scope_id = $1 AND id = $2",
                scope_id,
                durable_id,
            )
        return result.endswith("1")

    async def get_person(self, query: str) -> List[Dict[str, Any]]:
        q = (query or "").strip().casefold()
        if not q:
            return []
        matches = []
        for card in self.people_cache:
            if q == card["name"].casefold() or q == str(card.get("discord_id") or ""):
                matches.append(card)
                continue
            if any(q == alias.casefold() for alias in card.get("aliases") or []):
                matches.append(card)
        return matches

    async def list_scopes(self) -> List[str]:
        async with self.storage.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT scope_id
                FROM memories
                WHERE scope_id IS NOT NULL
                ORDER BY scope_id
                """
            )
        return [row["scope_id"] for row in rows]

    async def _state(self, scope_id: str) -> Dict[str, Any]:
        async with self.storage.connection() as conn:
            await conn.execute(
                """
                INSERT INTO memory_consolidation_state(scope_id)
                VALUES($1)
                ON CONFLICT (scope_id) DO NOTHING
                """,
                scope_id,
            )
            row = await conn.fetchrow(
                """
                SELECT scope_id, initial_done, last_processed_id, last_run_at
                FROM memory_consolidation_state
                WHERE scope_id = $1
                """,
                scope_id,
            )
        return dict(row)

    async def _replace_automatic_durable(
        self,
        scope_id: str,
        items: Sequence[Dict[str, str]],
    ):
        clean = []
        seen = set()
        for item in items:
            content = (item.get("content") or "").strip()
            category = (item.get("category") or "general").strip() or "general"
            key = (category.casefold(), content.casefold())
            if not content or key in seen:
                continue
            seen.add(key)
            clean.append((category, content))
            if len(clean) >= self.durable_limit:
                break

        async with self.storage.connection() as conn:
            await conn.execute(
                """
                DELETE FROM durable_memories
                WHERE scope_id = $1 AND pinned = FALSE
                """,
                scope_id,
            )
            for category, content in clean:
                await conn.execute(
                    """
                    INSERT INTO durable_memories(scope_id, category, content, pinned)
                    VALUES($1, $2, $3, FALSE)
                    """,
                    scope_id,
                    category,
                    content,
                )

    async def _archive_ids(
        self,
        scope_id: str,
        memory_ids: Sequence[int],
        allowed_ids: Sequence[int],
    ):
        allowed = set(int(value) for value in allowed_ids)
        ids = sorted({
            int(value)
            for value in memory_ids
            if int(value) in allowed
        })
        if not ids:
            return

        async with self.storage.connection() as conn:
            await conn.execute(
                """
                UPDATE memories
                SET status = 'archived',
                    archived_reason = 'consolidated',
                    consolidated_at = NOW()
                WHERE scope_id = $1
                  AND id = ANY($2::bigint[])
                """,
                scope_id,
                ids,
            )

    def self_identity_context(self) -> str:
        """Compact authoritative grounding for autobiographical ownership."""
        own = None
        wanted = self.companion_name.casefold().replace("_", "-")
        for card in self.people_cache:
            card_name = str(card.get("name") or "").casefold().replace("_", "-")
            aliases = [
                str(alias).casefold().replace("_", "-")
                for alias in (card.get("aliases") or [])
            ]
            if card_name == wanted or wanted in aliases:
                own = card
                break

        if own:
            return (
                f"Name: {own['name']}\n"
                f"Appearance: {own.get('appearance') or '-'}\n"
                f"Relationship: {own.get('relationship') or '-'}\n"
                f"Notes: {own.get('notes') or '-'}"
            )

        return f"Name: {self.companion_name}"

    def known_people_names(self) -> str:
        names = [
            str(card.get("name") or "").strip()
            for card in self.people_cache
            if str(card.get("name") or "").strip()
        ]
        return ", ".join(names)

    async def clear_automatic_durable(self, scope_id: str) -> int:
        """Delete only machine-generated durable memory; pinned human edits survive."""
        async with self.storage.connection() as conn:
            result = await conn.execute(
                """
                DELETE FROM durable_memories
                WHERE scope_id = $1
                  AND pinned = FALSE
                """,
                scope_id,
            )
        try:
            return int(result.split()[-1])
        except Exception:
            return 0

    async def _call_consolidator(
        self,
        scope_id: str,
        episodes: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], List[int]]:
        durables = await self.get_durable_memories(scope_id)
        pinned = [row for row in durables if row.get("pinned")]
        automatic = [row for row in durables if not row.get("pinned")]

        pinned_text = "\n".join(
            f"- {row['category']}: {row['content']}"
            for row in pinned
        ) or "(none)"

        automatic_text = "\n".join(
            f"- {row['category']}: {row['content']}"
            for row in automatic
        ) or "(none)"

        episode_text = "\n".join(
            f"[ID {row['id']}] [significance {row['significance']}] "
            f"[channel {row.get('channel_name') or row.get('channel_id')}] "
            f"{row['content']}"
            for row in episodes
        ) or "(none)"

        response = await self.client.responses.create(
            model=self.helper_model,
            reasoning={"effort": "low"},
            text={
                "format": CONSOLIDATION_JSON_SCHEMA,
                "verbosity": "low",
            },
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": CONSOLIDATION_SYSTEM}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Companion: {self.companion_name}\n\n"
                                f"SELF IDENTITY — AUTHORITATIVE:\n{self.self_identity_context()}\n\n"
                                f"KNOWN PEOPLE / OTHER POSSIBLE IDENTITIES:\n{self.known_people_names()}\n\n"
                                f"PINNED DURABLE MEMORIES:\n{pinned_text}\n\n"
                                f"AUTOMATIC DURABLE MEMORIES:\n{automatic_text}\n\n"
                                f"EPISODIC MEMORIES:\n{episode_text}"
                            ),
                        }
                    ],
                },
            ],
        )

        parsed = json.loads(response.output_text)
        durable = parsed.get("automatic_durable_memories") or []
        archive_ids = parsed.get("archive_ids") or []
        return durable, archive_ids

    async def consolidate_scope(
        self,
        scope_id: str,
        force_full: bool = False,
    ) -> Dict[str, Any]:
        state = await self._state(scope_id)

        if force_full or not state["initial_done"]:
            async with self.storage.connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, channel_id, channel_name, content,
                           significance, created_at
                    FROM memories
                    WHERE scope_id = $1
                      AND kind = 'episode'
                      AND status = 'active'
                    ORDER BY id ASC
                    """,
                    scope_id,
                )
            episodes = [dict(row) for row in rows]

            for start in range(0, len(episodes), self.consolidation_batch_size):
                batch = episodes[start:start + self.consolidation_batch_size]
                if not batch:
                    continue
                durable, archive_ids = await self._call_consolidator(scope_id, batch)
                await self._replace_automatic_durable(scope_id, durable)
                await self._archive_ids(
                    scope_id,
                    archive_ids,
                    [row["id"] for row in batch],
                )

            max_id = max([row["id"] for row in episodes], default=0)
            async with self.storage.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO memory_consolidation_state(
                        scope_id, initial_done, last_processed_id, last_run_at
                    )
                    VALUES($1, TRUE, $2, NOW())
                    ON CONFLICT (scope_id) DO UPDATE SET
                        initial_done = TRUE,
                        last_processed_id = GREATEST(
                            memory_consolidation_state.last_processed_id,
                            EXCLUDED.last_processed_id
                        ),
                        last_run_at = NOW()
                    """,
                    scope_id,
                    max_id,
                )
            return {
                "ran": True,
                "mode": "full",
                "episodes_seen": len(episodes),
            }

        last_processed = int(state["last_processed_id"] or 0)

        async with self.storage.connection() as conn:
            new_rows = await conn.fetch(
                """
                SELECT id, channel_id, channel_name, content,
                       significance, created_at
                FROM memories
                WHERE scope_id = $1
                  AND kind = 'episode'
                  AND status = 'active'
                  AND id > $2
                ORDER BY id ASC
                """,
                scope_id,
                last_processed,
            )

        if len(new_rows) < self.consolidate_every:
            return {
                "ran": False,
                "mode": "threshold",
                "new_count": len(new_rows),
            }

        # Process enough new memories to keep the next backlog manageable.
        new_batch = [dict(row) for row in new_rows[:self.consolidation_batch_size]]
        async with self.storage.connection() as conn:
            context_rows = await conn.fetch(
                """
                SELECT id, channel_id, channel_name, content,
                       significance, created_at
                FROM memories
                WHERE scope_id = $1
                  AND kind = 'episode'
                  AND status = 'active'
                  AND id <= $2
                ORDER BY id DESC
                LIMIT 8
                """,
                scope_id,
                last_processed,
            )

        context = [dict(row) for row in reversed(context_rows)]
        episodes = context + new_batch
        durable, archive_ids = await self._call_consolidator(scope_id, episodes)
        await self._replace_automatic_durable(scope_id, durable)
        await self._archive_ids(
            scope_id,
            archive_ids,
            [row["id"] for row in episodes],
        )

        processed_to = max(row["id"] for row in new_batch)
        async with self.storage.connection() as conn:
            await conn.execute(
                """
                UPDATE memory_consolidation_state
                SET last_processed_id = $2,
                    last_run_at = NOW()
                WHERE scope_id = $1
                """,
                scope_id,
                processed_to,
            )

        return {
            "ran": True,
            "mode": "incremental",
            "episodes_seen": len(episodes),
            "processed_to": processed_to,
        }
