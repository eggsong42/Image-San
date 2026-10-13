# Image-san Discord bot (MVP v3)

An image-native Discord companion using an OpenAI image model for replies and a cheap vision-capable helper model for factual first-person memory compression.

## Files

- `bot.py` — Discord bot + OpenAI calls + memory retrieval
- `schema.sql` — Neon/Postgres schema; the bot runs this automatically on startup
- `requirements.txt` — Python dependencies
- `.env.example` — recommended environment variables

## Railway

1. Put these files in the root of the GitHub repo connected to Railway.
2. Add the variables from `.env.example` in Railway Variables.
3. Set Railway's Start Command to:

   ```
   python bot.py
   ```

4. In the Discord Developer Portal, enable **Message Content Intent**.
5. Give the bot permission to View Channels, Send Messages, Attach Files, and Read Message History.

## Owner-only commands

Set:

```
OWNER_DISCORD_ID=<your numeric Discord user ID>
```

Only that Discord account can use the memory/admin commands. If the variable is missing, the commands stay locked for everyone. Unauthorized command attempts are silently consumed and are not added to Image-san's context.

Commands:

- `!memories` — newest 10 memories across the current server/DM scope, with channel tags
- `!memories 5` — newest 5
- `!memories 20` — maximum currently supported
- `!memorycontext` — shows exactly what Image-san would receive as the local recent-memory block and the server-wide RAG block for the current channel

These commands are not added to Image-san's conversational context.

## Memory architecture

Image-san now gets three distinct continuity layers:

1. **IMMEDIATE CONTEXT** — newest `RECENT_TEXT_MESSAGES` messages in the current channel. Image-san's own placeholders are excluded because her own recent outputs are supplied visually.
2. **RECENT MEMORIES** — newest `RECENT_MEMORIES` autobiographical memories from the current channel only. This is local episodic continuity: what has just been happening in this room/thread.
3. **RETRIEVED MEMORIES (OLDER / RAG)** — up to `MAX_RETRIEVED_MEMORIES` relevant memories from anywhere in the same Discord server. In DMs, retrieval stays inside that DM scope. RAG memories carry a channel tag such as `[channel: general]` so Image-san knows where the memory came from.
4. **REFERENCE IMAGES** — newest `RECENT_IMAGE_TURNS` self-images from the current channel, plus an optional canonical reference.

The defaults are 10 recent text messages, 10 channel-local recent memories, 6 server-wide RAG memories, and 3 channel-local recent self-images.

This means a busy `#general` channel does not overwrite the local recent-memory sequence of an ongoing scene in `#image-san-lab`, while long-term recall can still cross channels naturally.

Memory is scoped to the Discord server rather than globally across every server the bot joins, preventing unrelated communities from leaking memories into one another. DMs are isolated separately.

## Existing databases

`schema.sql` includes a safe migration for the earlier MVP schema. Old memories created before v3 cannot be mapped back to a Discord server because the old table did not store guild information, so they are tagged with a legacy channel-only scope. New memories will use proper server-wide scope automatically.

## Visual working memory

The newest `RECENT_IMAGE_TURNS` self-images are stored directly as Postgres `BYTEA` values and remain channel-local. Older image blobs are pruned. This makes the three recent images act like visual working memory for the current scene rather than dragging unrelated scenes from other channels into each generation.

## Reply behaviour

- DMs: replies to every human message when `RESPOND_IN_DMS=true`.
- Servers with `RESPOND_TO_ALL=false`: replies when mentioned or when someone replies directly to one of its messages.
- Servers with `RESPOND_TO_ALL=true`: replies to every non-bot message in every channel it can see.
