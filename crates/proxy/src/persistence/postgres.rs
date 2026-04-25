//! Postgres / TimescaleDB sink.
//!
//! Owns a `sqlx::PgPool` and an unbounded mpsc channel. The proxy hot
//! path enqueues `LogRecord`s synchronously (no allocation beyond the
//! clone, no DB roundtrip). A background writer task drains the channel,
//! batches up to `batch_size` records or every `flush_interval`, and
//! issues a single multi-row INSERT per batch.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Postgres, QueryBuilder};
use tokenova_domain::LogRecord;
use tokio::sync::mpsc;

use super::LogRecordSink;

/// Tunables for the writer task.
#[derive(Debug, Clone)]
pub struct PostgresSinkConfig {
    pub database_url: String,
    pub max_connections: u32,
    pub batch_size: usize,
    pub flush_interval: Duration,
}

impl PostgresSinkConfig {
    pub fn new(database_url: String) -> Self {
        Self {
            database_url,
            max_connections: 8,
            batch_size: 100,
            flush_interval: Duration::from_millis(500),
        }
    }
}

/// Postgres-backed persistence sink. Cheap to clone (just an `Arc`'d
/// channel sender).
#[derive(Clone)]
pub struct PostgresSink {
    tx: mpsc::UnboundedSender<LogRecord>,
}

impl PostgresSink {
    /// Build the pool, run migrations, spawn the writer task. Returns
    /// the sink ready to be installed via
    /// [`super::install_sink`]. Run inside a tokio runtime.
    pub async fn start(config: PostgresSinkConfig) -> anyhow::Result<Arc<Self>> {
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .connect(&config.database_url)
            .await
            .context("connecting to TOKENOVA_DATABASE_URL")?;

        run_migrations(&pool)
            .await
            .context("running tokenova-proxy migrations")?;

        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(writer_loop(
            pool,
            rx,
            config.batch_size,
            config.flush_interval,
        ));

        Ok(Arc::new(Self { tx }))
    }
}

impl LogRecordSink for PostgresSink {
    fn enqueue(&self, record: LogRecord) {
        // Channel send is non-blocking. If the writer task has died,
        // `send` returns Err — log and drop the record. We don't retry
        // because the proxy hot path must never block on persistence.
        if let Err(err) = self.tx.send(record) {
            tracing::warn!(
                error = %err,
                "persistence channel closed; dropping LogRecord"
            );
        }
    }
}

async fn writer_loop(
    pool: PgPool,
    mut rx: mpsc::UnboundedReceiver<LogRecord>,
    batch_size: usize,
    flush_interval: Duration,
) {
    let mut batch: Vec<LogRecord> = Vec::with_capacity(batch_size);
    let mut interval = tokio::time::interval(flush_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            // Biased select so a flooded channel doesn't starve the
            // periodic flush.
            biased;

            received = rx.recv() => {
                match received {
                    Some(record) => {
                        batch.push(record);
                        if batch.len() >= batch_size {
                            flush(&pool, &mut batch).await;
                        }
                    }
                    None => {
                        // Channel closed — drain final batch and exit.
                        if !batch.is_empty() {
                            flush(&pool, &mut batch).await;
                        }
                        tracing::info!("persistence writer exiting (channel closed)");
                        return;
                    }
                }
            }

            _ = interval.tick() => {
                if !batch.is_empty() {
                    flush(&pool, &mut batch).await;
                }
            }
        }
    }
}

async fn flush(pool: &PgPool, batch: &mut Vec<LogRecord>) {
    if batch.is_empty() {
        return;
    }
    let records: Vec<LogRecord> = std::mem::take(batch);
    let count = records.len();

    if let Err(err) = insert_batch(pool, &records).await {
        tracing::error!(
            error = %err,
            dropped = count,
            "failed to persist LogRecord batch"
        );
    }
}

async fn insert_batch(pool: &PgPool, records: &[LogRecord]) -> anyhow::Result<()> {
    let mut qb: QueryBuilder<Postgres> = QueryBuilder::new(
        "INSERT INTO log_records (\
            request_id, received_at, provider, model, \
            attribution_user, attribution_team, attribution_department, \
            attribution_project, attribution_application, attribution_cost_center, \
            attribution_environment, \
            prompt_tokens, completion_tokens, total_tokens, \
            cost_usd, intent, latency_added_ms, upstream_status, \
            streamed, stream_truncated, stream_error, stream_duration_ms\
        ) ",
    );

    qb.push_values(records.iter(), |mut b, r| {
        b.push_bind(r.request_id)
            .push_bind(r.received_at)
            .push_bind(r.provider.as_str())
            .push_bind(&r.model)
            .push_bind(r.attribution.user.as_deref())
            .push_bind(r.attribution.team.as_deref())
            .push_bind(r.attribution.department.as_deref())
            .push_bind(r.attribution.project.as_deref())
            .push_bind(r.attribution.application.as_deref())
            .push_bind(r.attribution.cost_center.as_deref())
            .push_bind(r.attribution.environment.as_deref())
            .push_bind(i64::from(r.usage.prompt_tokens))
            .push_bind(i64::from(r.usage.completion_tokens))
            .push_bind(i64::from(r.usage.total_tokens))
            .push_bind(r.cost_usd)
            .push_bind(r.intent.as_str())
            .push_bind(r.latency_added_ms)
            .push_bind(i32::from(r.upstream_status))
            .push_bind(r.streamed)
            .push_bind(r.stream_truncated)
            .push_bind(r.stream_error)
            .push_bind(r.stream_duration_ms);
    });

    qb.push(" ON CONFLICT (request_id) DO NOTHING");
    qb.build().execute(pool).await?;
    Ok(())
}

async fn run_migrations(pool: &PgPool) -> anyhow::Result<()> {
    // Inline migration: keeps the proxy crate self-contained and lets
    // operators run the table creation without a separate sqlx-cli step.
    // Idempotent — safe to run on every startup.
    sqlx::query(include_str!("../../migrations/0001_create_log_records.sql"))
        .execute(pool)
        .await?;
    Ok(())
}
