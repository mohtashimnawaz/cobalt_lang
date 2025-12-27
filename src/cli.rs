use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::fs;
use anyhow::{Result, Context};

/// Cobalt CLI
#[derive(Parser, Debug)]
#[command(name = "cobalt", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Build a source file
    Build {
        /// Input source file
        input: PathBuf,
        /// Output path (object file or IR)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Build { input, output } => run_build(input, output),
    }
}

fn run_build(input: PathBuf, output: Option<PathBuf>) -> Result<()> {
    let src = fs::read_to_string(&input).with_context(|| format!("failed to read input file {:?}", input))?;

    match crate::parser::parse_module(&src) {
        Ok(module) => {
            // Run semantic checks (type checker) and emit diagnostics and warnings
            let (res, warnings) = crate::ty::type_check_module_with_warnings(&module);
            if !warnings.is_empty() {
                crate::diagnostics::report_type_warnings(&src, input.to_str().unwrap_or("<input>"), &warnings);
            }
            if let Err(type_errors) = res {
                crate::diagnostics::report_type_errors(&src, input.to_str().unwrap_or("<input>"), &type_errors);
                anyhow::bail!("type check failed with {} error(s)", type_errors.len())
            }

            let out = output.clone().unwrap_or_else(|| {
                let mut p = input.clone();
                p.set_extension("o");
                p
            });

            #[cfg(feature = "llvm")]
            {
                use crate::codegen;
                use std::path::Path;
                codegen::compile_module_to_object(&module, "cobalt_module", out.as_path())
                    .with_context(|| "codegen to object failed")?;
                println!("wrote object to {:?}", out);
                return Ok(());
            }

            #[cfg(not(feature = "llvm"))]
            {
                // Fallback: produce textual IR using compile_module_to_ir if available, otherwise save a .ll-like placeholder
                match crate::codegen::compile_module_to_ir(&module, "cobalt_module") {
                    Ok(ir) => {
                        let p = output.clone().unwrap_or_else(|| { let mut p = input.clone(); p.set_extension("ll"); p });
                        fs::write(&p, ir).with_context(|| format!("failed to write IR to {:?}", p))?;
                        println!("wrote IR to {:?} (build with --features llvm to emit object files)", p);
                        Ok(())
                    }
                    Err(e) => Err(e).with_context(|| "codegen feature disabled; enable `llvm` feature to generate object files"),
                }
            }
        }
        Err(errors) => {
            // Report diagnostics
            crate::diagnostics::report_parse_errors(&src, input.to_str().unwrap_or("<input>"), &errors);
            anyhow::bail!("parse failed with {} error(s)", errors.len())
        }
    }
}
