-- Tokenova proxy persistence schema.
--
-- One row per proxied AI request. The hypertable conversion at the
-- bottom is a no-op on plain Postgres (caught and logged) so this
-- migration runs cleanly against either TimescaleDB or vanilla PG.

CREATE TABLE IF NOT EXISTS log_records (
    request_id              UUID         NOT NULL,
    received_at             TIMESTAMPTZ  NOT NULL,
    provider                TEXT         NOT NULL,
    model                   TEXT         NOT NULL,

    attribution_user        TEXT,
    attribution_team        TEXT,
    attribution_department  TEXT,
    attribution_project     TEXT,
    attribution_application TEXT,
    attribution_cost_center TEXT,
    attribution_environment TEXT,

    prompt_tokens           BIGINT       NOT NULL,
    completion_tokens       BIGINT       NOT NULL,
    total_tokens            BIGINT       NOT NULL,

    cost_usd                DOUBLE PRECISION NOT NULL,
    intent                  TEXT         NOT NULL,
    latency_added_ms        DOUBLE PRECISION NOT NULL,
    upstream_status         INTEGER      NOT NULL,

    streamed                BOOLEAN      NOT NULL,
    stream_truncated        BOOLEAN      NOT NULL,
    stream_error            BOOLEAN      NOT NULL,
    stream_duration_ms      DOUBLE PRECISION,

    PRIMARY KEY (request_id, received_at)
);

-- Common dashboard queries: spend-by-team-over-time, spend-by-user,
-- per-provider rollups. All filter by received_at (DESC) so include it
-- in every index.
CREATE INDEX IF NOT EXISTS idx_log_records_provider_received_at
    ON log_records (provider, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_log_records_user_received_at
    ON log_records (attribution_user, received_at DESC)
    WHERE attribution_user IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_log_records_team_received_at
    ON log_records (attribution_team, received_at DESC)
    WHERE attribution_team IS NOT NULL;

-- TimescaleDB hypertable conversion. Wrapped in plpgsql so we degrade
-- cleanly to a regular table on plain Postgres (the function won't
-- exist and the EXCEPTION block swallows the error).
DO $$
BEGIN
    PERFORM create_hypertable('log_records', 'received_at',
                              if_not_exists => TRUE,
                              migrate_data  => TRUE);
EXCEPTION
    WHEN undefined_function THEN
        RAISE NOTICE 'TimescaleDB extension not present; log_records remains a regular table.';
END;
$$;
