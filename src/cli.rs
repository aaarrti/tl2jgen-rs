use clap::{Parser, ValueEnum};

#[derive(Clone, ValueEnum, Debug)]
pub enum Backend {
    C,
    JAVA,
}

#[derive(Debug, Clone, Parser)]
pub struct CLI {
    #[clap(long, default_value_t = Backend::C, short, long)]
    #[arg(value_enum, short, long)]
    pub backend: Backend,
    #[arg(short, long)]
    pub source: String,
    #[arg(short, long)]
    pub destination: String,
}
