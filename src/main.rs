mod backend;
mod ir;

use tracing_subscriber::EnvFilter;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Debug, Clone, Parser)]
pub struct CLI {
    #[arg(short, long)]
    pub source: String,
    #[arg(short, long)]
    pub destination: String,
}

fn setup_tracing() -> Result<()> {
    let subscriber = tracing_subscriber::fmt()
        .pretty()
        .with_env_filter(EnvFilter::from_default_env())
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .with_context(|| "Failed to setup tracing")?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = CLI::try_parse()?;
    setup_tracing()?;
    let model = ir::json::decode(cli.source.as_str())?;
    backend::c::emit(&model, cli.destination.as_str())
}
