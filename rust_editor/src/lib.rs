# =========================================
# File: src/lib.rs
# =========================================
//! Ritsu Rust Editor Core
//! - Rope-based text buffer
//! - Optional GPU backend (feature = "gpu")
//! - Lightweight static analyzer + formatter
//! - JSON-RPC LSP bridge for smart IDE features


mod analyzer;
mod formatter;
mod lsp_bridge;


use anyhow::Result;
use ropey::Rope;
use std::{path::PathBuf, fs, io::Write};
use uuid::Uuid;


pub use analyzer::{AnalyzerIssue, AnalyzerSeverity, AnalyzerSummary};
pub use formatter::{FormatOptions, format_rust_like};
pub use lsp_bridge::{LspBridge, LspEvent, LspRequestId};


#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Position { pub line: usize, pub column: usize }


#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Range { pub start: Position, pub end: Position }


#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Edit { pub range: Range, pub text: String }


#[derive(Debug, Default)]
pub struct Editor {
id: Uuid,
rope: Rope,
file_path: Option<PathBuf>,
cursor: Position,
undo: Vec<Edit>,
redo: Vec<Edit>,
}


impl Editor {
pub fn new() -> Self {
Self { id: Uuid::new_v4(), rope: Rope::new(), file_path: None, cursor: Position{line:0, column:0}, undo: vec![], redo: vec![] }
}


pub fn open_from_str(path: Option<PathBuf>, text: &str) -> Self {
let mut ed = Self::new();
ed.rope = Rope::from_str(text);
ed.file_path = path;
ed
}


pub fn open_from_file(path: impl Into<PathBuf>) -> Result<Self> {
let path = path.into();
let text = fs::read_to_string(&path)?;
Ok(Self::open_from_str(Some(path), &text))
}


}