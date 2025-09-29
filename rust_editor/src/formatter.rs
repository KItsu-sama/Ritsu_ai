# =========================================
# File: src/formatter.rs
# =========================================
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FormatOptions {
pub indent_width: usize,
pub use_tabs: bool,
pub max_empty_lines: usize,
}


impl Default for FormatOptions {
fn default() -> Self { Self { indent_width: 4, use_tabs: false, max_empty_lines: 2 } }
}


pub fn format_rust_like(src: &str) -> String { format_rust_like_with_opts(src, FormatOptions::default()) }


pub fn format_rust_like_with_opts(src: &str, opts: FormatOptions) -> String {
let mut out = String::with_capacity(src.len());
let mut indent: isize = 0;
let mut empty_run = 0usize;


for raw_line in src.lines() {
let line = raw_line.trim();
if line.is_empty() {
empty_run += 1;
if empty_run > opts.max_empty_lines { continue; }
out.push('\n');
continue;
}
empty_run = 0;


// De-dent on leading closers
let mut effective_indent = indent;
if line.starts_with('}') || line.starts_with(']') || line.starts_with(')') {
effective_indent -= 1;
}
if effective_indent < 0 { effective_indent = 0; }


let unit = if opts.use_tabs { "\t".to_string() } else { " ".repeat(opts.indent_width) };
for _ in 0..effective_indent { out.push_str(&unit); }
out.push_str(line);
out.push('\n');


// Increase indent if line ends with an opener or contains opener without closer
let opens = line.chars().filter(|c| matches!(c, '{'|'['|'(')).count() as isize;
let closes = line.chars().filter(|c| matches!(c, '}'|']'|')')).count() as isize;
indent += opens - closes;
if indent < 0 { indent = 0; }
}


out
}