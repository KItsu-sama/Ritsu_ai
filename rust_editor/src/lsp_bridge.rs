# =========================================
# File: src/lsp_bridge.rs
# =========================================
use serde_json::{json, Value};
use std::{io::{Read, Write}, process::{Child, ChildStdin, ChildStdout, Command, Stdio}, thread, sync::mpsc::{self, Sender, Receiver}};
use url::Url;
use uuid::Uuid;

pub type LspRequestId = String;

#[derive(Debug, Clone)]
pub enum LspEvent {
    Notification(Value),
    Response { id: LspRequestId, result: Value },
    Error { id: Option<LspRequestId>, error: Value },
}

pub struct LspBridge {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    tx: Sender<LspEvent>,
    rx: Receiver<LspEvent>,
}

impl LspBridge {
    pub fn start(server_cmd: &str, args: &[&str]) -> anyhow::Result<Self> {
        let mut child = Command::new(server_cmd)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().unwrap();
        let mut stdout = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();

        // Reader thread: parse JSON-RPC messages with Content-Length header
        let tx_clone = tx.clone();
        thread::spawn(move || {
            loop {
                match read_lsp_message(&mut stdout) {
                    Ok(v) => {
                        // Route by presence of id/result/error
                        if let Some(id) = v.get("id").and_then(|x| x.as_str()).map(|s| s.to_string()) {
                            if let Some(result) = v.get("result").cloned() {
                                let _ = tx_clone.send(LspEvent::Response{ id, result });
                            } else if let Some(err) = v.get("error").cloned() {
                                let _ = tx_clone.send(LspEvent::Error{ id: Some(id), error: err });
                            }
                        } else {
                            let _ = tx_clone.send(LspEvent::Notification(v));
                        }
                    }
                    Err(e) => {
                        let _ = tx_clone.send(LspEvent::Error{ id: None, error: json!({"message": e.to_string()}) });
                        break;
                    }
                }
            }
        });

        Ok(Self{ child, stdin, stdout, tx, rx })
    }

    pub fn initialize(&mut self, root_uri: &str) -> anyhow::Result<LspRequestId> {
        let id = Uuid::new_v4().to_string();
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "processId": std::process::id(),
                "rootUri": Url::parse(root_uri).ok().map(|u| u.as_str().to_string()),
                "capabilities": {"textDocument": {"synchronization": {"didSave": true}}}
            }
        });
        write_lsp_message(&mut self.stdin, &msg)?;
        Ok(msg.get("id").unwrap().as_str().unwrap().to_string())
    }

    pub fn did_open(&mut self, uri: &str, language_id: &str, version: i32, text: &str) -> anyhow::Result<()> {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {"uri": uri, "languageId": language_id, "version": version, "text": text}
            }
        });
        write_lsp_message(&mut self.stdin, &msg)
    }

    pub fn hover(&mut self, uri: &str, line: u32, character: u32) -> anyhow::Result<LspRequestId> {
        let id = Uuid::new_v4().to_string();
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "textDocument/hover",
            "params": {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}}
        });
        write_lsp_message(&mut self.stdin, &msg)?;
        Ok(id)
    }

    pub fn completion(&mut self, uri: &str, line: u32, character: u32) -> anyhow::Result<LspRequestId> {
        let id = Uuid::new_v4().to_string();
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "textDocument/completion",
            "params": {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}}
        });
        write_lsp_message(&mut self.stdin, &msg)?;
        Ok(id)
    }

    pub fn shutdown(&mut self) -> anyhow::Result<()> {
        let msg = json!({"jsonrpc": "2.0", "id": Uuid::new_v4().to_string(), "method": "shutdown"});
        write_lsp_message(&mut self.stdin, &msg)
    }

    pub fn recv(&self) -> Option<LspEvent> { self.rx.try_recv().ok() }
}

fn write_lsp_message(mut stdin: &ChildStdin, v: &serde_json::Value) -> anyhow::Result<()> {
    let body = v.to_string();
    let header = format!("Content-Length: {}\r\n\r\n", body.as_bytes().len());
    stdin.write_all(header.as_bytes())?;
    stdin.write_all(body.as_bytes())?;
    stdin.flush()?;
    Ok(())
}

fn read_lsp_message(stdout: &mut ChildStdout) -> anyhow::Result<serde_json::Value> {
    let mut header_buf = Vec::new();
    let mut tmp = [0u8; 1];
    // Read until CRLF CRLF
    let mut last4 = [0u8; 4];
    loop {
        stdout.read_exact(&mut tmp)?;
        header_buf.push(tmp[0]);
        last4.copy_within(1..4, 0);
        last4[3] = tmp[0];
        if &last4 == b"\r\n\r\n" { break; }
    }
    let header = String::from_utf8_lossy(&header_buf);
    let mut content_length = None;
    for line in header.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Content-Length:") { content_length = rest.trim().parse::<usize>().ok(); }
    }
    let len = content_length.ok_or_else(|| anyhow::anyhow!("Missing Content-Length"))?;
    let mut body = vec![0u8; len];
    stdout.read_exact(&mut body)?;
    let v: Value = serde_json::from_slice(&body)?;
    Ok(v)
}
