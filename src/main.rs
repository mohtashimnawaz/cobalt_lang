mod ast;
mod parser;
mod lexer;
mod diagnostics;
mod ty;
mod codegen;
mod cli;

fn main() {
    env_logger::init();
    if let Err(e) = cli::run() {
        eprintln!("error: {:#}", e);
        std::process::exit(1);
    }
}
