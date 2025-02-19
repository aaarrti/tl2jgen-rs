mod backend;
mod cli;
mod ir;

use cli::{Backend, CLI};
use tracing;
use tracing_subscriber;
use tracing_subscriber::EnvFilter;

use anyhow::{Context, Result};
use clap::Parser;

fn setup_tracing() -> Result<()> {
    let subscriber = tracing_subscriber::fmt()
        .pretty()
        .with_env_filter(EnvFilter::from_default_env())
        .with_thread_ids(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .with_context(|| "Failed to setup tracing")?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = CLI::try_parse()?;
    setup_tracing()?;

    let model = ir::json::decode(cli.source.as_str())?;

    match cli.backend {
        Backend::C => backend::c::emit(&model, cli.destination.as_str()),
        Backend::JAVA => backend::java::emit(&model, cli.destination.as_str()),
    }
}
