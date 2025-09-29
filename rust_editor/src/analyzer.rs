# =========================================
# File: src/analyzer.rs
# =========================================
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnalyzerSeverity { Info, Warning, Error }


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerIssue {
pub code: &'static str,
pub severity: AnalyzerSeverity,
pub message: String,
pub line: usize,
pub column: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnalyzerSummary {
pub issues: Vec<AnalyzerIssue>,
pub line_count: usize,
pub char_count: usize,
pub todos: usize,
}


pub fn analyze_text(src: &str) -> AnalyzerSummary {
let mut summary = AnalyzerSummary::default();
summary.line_count = src.lines().count();
summary.char_count = src.chars().count();


let todo_re = Regex::new(r"(?i)\\bTODO\\b").unwrap();
let long_line_threshold = 120usize;


for (i, line) in src.lines().enumerate() {
if line.len() > long_line_threshold {
summary.issues.push(AnalyzerIssue{ code: "L001", severity: AnalyzerSeverity::Warning, message: format!("Line exceeds {long_line_threshold} chars ({})", line.len()), line: i, column: long_line_threshold });
}
for m in todo_re.find_iter(line) {
summary.todos += 1;
summary.issues.push(AnalyzerIssue{ code: "N001", severity: AnalyzerSeverity::Info, message: "TODO found".into(), line: i, column: m.start() });
}
// Naive trailing whitespace check
if line.ends_with(' ') || line.ends_with('\t') {
summary.issues.push(AnalyzerIssue{ code: "S001", severity: AnalyzerSeverity::Info, message: "Trailing whitespace".into(), line: i, column: line.len().saturating_sub(1) });
}
}


// Simple bracket balance check
let mut stack = Vec::new();
for (i, line) in src.lines().enumerate() {
for (j, ch) in line.chars().enumerate() {
match ch {
'{' | '(' | '[' => stack.push((ch, i, j)),
'}' => { if !matches!(stack.pop().map(|s| s.0), Some('{')) {
summary.issues.push(AnalyzerIssue{ code: "P001", severity: AnalyzerSeverity::Error, message: "Unmatched '}'".into(), line: i, column: j });
}},
')' => { if !matches!(stack.pop().map(|s| s.0), Some('(')) {
summary.issues.push(AnalyzerIssue{ code: "P002", severity: AnalyzerSeverity::Error, message: "Unmatched ')'".into(), line: i, column: j });
}},
']' => { if !matches!(stack.pop().map(|s| s.0), Some('[')) {
summary.issues.push(AnalyzerIssue{ code: "P003", severity: AnalyzerSeverity::Error, message: "Unmatched ']'".into(), line: i, column: j });
}},
_ => {}
}
}
}
for (_, i, j) in stack.into_iter() {
// unclosed opener
summary.issues.push(AnalyzerIssue{ code: "P004", severity: AnalyzerSeverity::Error, message: "Unclosed delimiter".into(), line: i, column: j });
}


summary
}