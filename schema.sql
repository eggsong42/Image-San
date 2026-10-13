CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    channel_id TEXT NOT NULL,
    discord_message_id TEXT UNIQUE NOT NULL,
    author_id TEXT NOT NULL,
    author_name TEXT NOT NULL,
    content TEXT NOT NULL,
    is_bot BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS outputs (
    id BIGSERIAL PRIMARY KEY,
    channel_id TEXT NOT NULL,
    discord_message_id TEXT UNIQUE NOT NULL,
    attachment_url TEXT NOT NULL,
    prompt_used TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memories (
    id BIGSERIAL PRIMARY KEY,
    scope_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    channel_name TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    significance INTEGER NOT NULL DEFAULT 1,
    source_output_id BIGINT REFERENCES outputs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

-- Safe migration for an earlier MVP database that already had `memories`
-- without server/DM scope or human-readable channel tags.
ALTER TABLE memories ADD COLUMN IF NOT EXISTS scope_id TEXT;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS channel_name TEXT;
UPDATE memories
SET scope_id = 'legacy-channel:' || channel_id
WHERE scope_id IS NULL;
UPDATE memories
SET channel_name = channel_id
WHERE channel_name IS NULL;
ALTER TABLE memories ALTER COLUMN scope_id SET NOT NULL;
ALTER TABLE memories ALTER COLUMN channel_name SET NOT NULL;

-- Short-term visual working memory. Only the newest N images per channel are kept.
CREATE TABLE IF NOT EXISTS recent_images (
    id BIGSERIAL PRIMARY KEY,
    channel_id TEXT NOT NULL,
    discord_message_id TEXT UNIQUE NOT NULL,
    image_bytes BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_channel_created
    ON messages(channel_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_outputs_channel_created
    ON outputs(channel_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_channel_created
    ON memories(channel_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_scope_created
    ON memories(scope_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_scope_significance
    ON memories(scope_id, significance DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_recent_images_channel_created
    ON recent_images(channel_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_fts
    ON memories
    USING GIN (to_tsvector('english', content));
