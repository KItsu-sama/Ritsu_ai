from __future__ import annotations

"""
core/tools.py

Toolbelt — utility calls (APIs, sys commands)
- System command execution
- API integration helpers
- File system operations
- Basic utility functions
"""



# core/tools.py
"""
Advanced Tool System for Ritsu AI
Provides hands-on capabilities similar to advanced AI assistants
"""

import os
import re
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import tempfile

@dataclass
class ToolResult:
    """Result of a tool operation"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    tool_name: str = ""
    
class RitsuToolSystem:
    """Advanced tool system providing hands-on capabilities"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_directory = Path.cwd()
        self.tool_usage_log = []
        
    # === FILE OPERATIONS ===
    
    def read_files(self, file_paths: List[str], max_lines: int = 5000) -> ToolResult:
        """Read contents of specified files"""
        try:
            results = {}
            for path_str in file_paths:
                file_path = Path(path_str)
                if not file_path.exists():
                    results[path_str] = {"error": "File not found"}
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        
                    lines = content.split('\n')
                    if len(lines) > max_lines:
                        content = '\n'.join(lines[:max_lines])
                        content += f'\n\n[... truncated at {max_lines} lines ...]'
                    
                    results[path_str] = {
                        "content": content,
                        "lines": len(lines),
                        "truncated": len(lines) > max_lines
                    }
                    
                except Exception as e:
                    results[path_str] = {"error": f"Read error: {e}"}
                    
            return ToolResult(True, results, tool_name="read_files")
            
        except Exception as e:
            return ToolResult(False, error=f"Read files failed: {e}", tool_name="read_files")
    
    def create_file(self, file_path: str, content: str, summary: str = "") -> ToolResult:
        """Create a new file with specified content"""
        try:
            path = Path(file_path)
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Created file: {file_path}")
            if summary:
                self.logger.info(f"File summary: {summary}")
                
            return ToolResult(True, {"file_path": file_path, "summary": summary}, tool_name="create_file")
            
        except Exception as e:
            return ToolResult(False, error=f"Create file failed: {e}", tool_name="create_file")
    
    def edit_files(self, edits: List[Dict[str, Any]]) -> ToolResult:
        """Apply diff-based edits to files"""
        try:
            results = []
            
            for edit in edits:
                file_path = Path(edit["file_path"])
                search_text = edit["search"]
                replace_text = edit["replace"]
                search_start_line = edit.get("search_start_line_number", 1)
                
                if not file_path.exists():
                    results.append({"file": str(file_path), "error": "File not found"})
                    continue
                
                try:
                    # Read current content
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    
                    original_lines = content.split('\n')
                    
                    # Find the search text starting from specified line
                    search_lines = search_text.split('\n')
                    found = False
                    
                    for start_idx in range(search_start_line - 1, len(original_lines)):
                        # Check if search pattern matches from this position
                        match = True
                        for i, search_line in enumerate(search_lines):
                            if start_idx + i >= len(original_lines) or original_lines[start_idx + i] != search_line:
                                match = False
                                break
                        
                        if match:
                            # Apply replacement
                            end_idx = start_idx + len(search_lines)
                            new_lines = original_lines[:start_idx]
                            
                            if replace_text.strip():
                                new_lines.extend(replace_text.split('\n'))
                            
                            new_lines.extend(original_lines[end_idx:])
                            
                            # Write back to file
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(new_lines))
                            
                            results.append({
                                "file": str(file_path),
                                "status": "success",
                                "lines_changed": len(search_lines),
                                "start_line": start_idx + 1
                            })
                            found = True
                            break
                    
                    if not found:
                        results.append({
                            "file": str(file_path),
                            "error": f"Search pattern not found starting from line {search_start_line}"
                        })
                        
                except Exception as e:
                    results.append({"file": str(file_path), "error": f"Edit error: {e}"})
            
            return ToolResult(True, {"edits": results}, tool_name="edit_files")
            
        except Exception as e:
            return ToolResult(False, error=f"Edit files failed: {e}", tool_name="edit_files")
    
    def find_files(self, patterns: List[str], search_dir: str = ".", max_matches: int = 50, max_depth: int = 0) -> ToolResult:
        """Find files matching patterns"""
        try:
            search_path = Path(search_dir)
            if not search_path.exists():
                return ToolResult(False, error=f"Search directory not found: {search_dir}", tool_name="find_files")
            
            matches = []
            
            def should_search_depth(path: Path, current_depth: int) -> bool:
                if max_depth == 0:
                    return True
                relative_path = path.relative_to(search_path)
                return len(relative_path.parts) <= max_depth
            
            for pattern in patterns:
                # Convert simple patterns to proper glob patterns
                if max_depth == 0:
                    glob_pattern = f"**/{pattern}"
                else:
                    glob_pattern = pattern
                
                try:
                    for match in search_path.rglob(glob_pattern):
                        if match.is_file():
                            if should_search_depth(match, len(match.relative_to(search_path).parts)):
                                relative_match = match.relative_to(search_path)
                                matches.append(str(relative_match))
                                
                                if len(matches) >= max_matches:
                                    break
                except Exception as e:
                    self.logger.warning(f"Pattern '{pattern}' search error: {e}")
                
                if len(matches) >= max_matches:
                    break
            
            return ToolResult(True, {"matches": matches, "total": len(matches)}, tool_name="find_files")
            
        except Exception as e:
            return ToolResult(False, error=f"Find files failed: {e}", tool_name="find_files")
    
    def grep(self, queries: List[str], search_dir: str = ".") -> ToolResult:
        """Search for text patterns in files"""
        try:
            search_path = Path(search_dir)
            if not search_path.exists():
                return ToolResult(False, error=f"Search directory not found: {search_dir}", tool_name="grep")
            
            results = {}
            
            for query in queries:
                matches = []
                pattern = re.compile(query, re.IGNORECASE | re.MULTILINE)
                
                # Search through all text files
                for file_path in search_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix in ['.py', '.js', '.ts', '.txt', '.md', '.json', '.yml', '.yaml', '.cfg', '.conf']:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                content = f.read()
                            
                            for line_num, line in enumerate(content.split('\n'), 1):
                                if pattern.search(line):
                                    matches.append({
                                        "file": str(file_path.relative_to(search_path)),
                                        "line": line_num,
                                        "content": line.strip(),
                                        "query": query
                                    })
                                    
                                    if len(matches) >= 100:  # Limit results
                                        break
                                        
                        except Exception as e:
                            continue  # Skip files that can't be read
                
                results[query] = matches
            
            return ToolResult(True, {"results": results}, tool_name="grep")
            
        except Exception as e:
            return ToolResult(False, error=f"Grep failed: {e}", tool_name="grep")
    
    # === SHELL OPERATIONS ===
    
    def run_command(self, command: str, timeout: int = 30, capture_output: bool = True) -> ToolResult:
        """Execute shell command with proper error handling"""
        try:
            # Parse command safely
            if isinstance(command, str):
                # Use shell=True for PowerShell compatibility on Windows
                use_shell = True
            else:
                use_shell = False
            
            self.logger.info(f"Executing command: {command}")
            
            result = subprocess.run(
                command,
                shell=use_shell,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace',
                cwd=str(self.current_directory)
            )
            
            return ToolResult(True, {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": command
            }, tool_name="run_command")
            
        except subprocess.TimeoutExpired:
            return ToolResult(False, error=f"Command timed out after {timeout}s", tool_name="run_command")
        except Exception as e:
            return ToolResult(False, error=f"Command execution failed: {e}", tool_name="run_command")
    
    # === CODE ANALYSIS ===
    
    def analyze_code_quality(self, project_path: str = ".") -> ToolResult:
        """Analyze code quality in project"""
        try:
            from core.code_analyzer import CodeAnalyzer
            analyzer = CodeAnalyzer()
            
            analysis = analyzer.analyze_project(Path(project_path))
            return ToolResult(True, analysis, tool_name="analyze_code_quality")
            
        except ImportError:
            return ToolResult(False, error="CodeAnalyzer not available", tool_name="analyze_code_quality")
        except Exception as e:
            return ToolResult(False, error=f"Code analysis failed: {e}", tool_name="analyze_code_quality")
    
    def suggest_code_fixes(self, file_path: str) -> ToolResult:
        """Suggest fixes for code issues"""
        try:
            from core.code_analyzer import CodeAnalyzer
            analyzer = CodeAnalyzer()
            
            issues = analyzer.analyze_file(Path(file_path))
            suggested_fixes = []
            
            for issue in issues:
                fix = analyzer.suggest_code_fix(issue)
                if fix:
                    suggested_fixes.append({
                        "issue": {
                            "type": issue.issue_type,
                            "severity": issue.severity,
                            "description": issue.description,
                            "line": issue.line_number
                        },
                        "fix": {
                            "reason": fix.reason,
                            "old_content": fix.old_content,
                            "new_content": fix.new_content,
                            "start_line": fix.start_line,
                            "end_line": fix.end_line
                        }
                    })
            
            return ToolResult(True, {"fixes": suggested_fixes}, tool_name="suggest_code_fixes")
            
        except ImportError:
            return ToolResult(False, error="CodeAnalyzer not available", tool_name="suggest_code_fixes")
        except Exception as e:
            return ToolResult(False, error=f"Code fix suggestion failed: {e}", tool_name="suggest_code_fixes")
    
    def apply_code_fixes(self, fixes: List[Dict]) -> ToolResult:
        """Apply suggested code fixes"""
        try:
            from core.code_analyzer import CodeAnalyzer, CodeChange
            analyzer = CodeAnalyzer()
            
            results = []
            for fix_data in fixes:
                fix_info = fix_data.get("fix", {})
                change = CodeChange(
                    file_path=fix_data.get("file_path"),
                    start_line=fix_info.get("start_line"),
                    end_line=fix_info.get("end_line"),
                    old_content=fix_info.get("old_content"),
                    new_content=fix_info.get("new_content"),
                    reason=fix_info.get("reason")
                )
                
                success = analyzer.apply_code_change(change)
                results.append({
                    "file": change.file_path,
                    "success": success,
                    "reason": change.reason
                })
            
            return ToolResult(True, {"applied_fixes": results}, tool_name="apply_code_fixes")
            
        except Exception as e:
            return ToolResult(False, error=f"Apply fixes failed: {e}", tool_name="apply_code_fixes")
    
    # === DIAGNOSTIC OPERATIONS ===
    
    def generate_diagnostic_report(self, monitor=None) -> ToolResult:
        """Generate comprehensive diagnostic report"""
        try:
            from core.troubleshooter import Troubleshooter
            troubleshooter = Troubleshooter()
            
            report = troubleshooter.generate_diagnostic_report(monitor)
            return ToolResult(True, report, tool_name="generate_diagnostic_report")
            
        except ImportError:
            return ToolResult(False, error="TroubleshootingEngine not available", tool_name="generate_diagnostic_report")
        except Exception as e:
            return ToolResult(False, error=f"Diagnostic report failed: {e}", tool_name="generate_diagnostic_report")
    
    def create_action_plan(self, problems: List) -> ToolResult:
        """Create action plan for detected problems"""
        try:
            from core.troubleshooter import TroubleshootingEngine
            troubleshooter = TroubleshootingEngine()
            
            action_plan = troubleshooter.create_action_plan(problems)
            return ToolResult(True, action_plan, tool_name="create_action_plan")
            
        except ImportError:
            return ToolResult(False, error="TroubleshootingEngine not available", tool_name="create_action_plan")
        except Exception as e:
            return ToolResult(False, error=f"Action plan creation failed: {e}", tool_name="create_action_plan")
    
    # === PROJECT MANAGEMENT ===
    
    def analyze_project_structure(self, project_path: str = ".") -> ToolResult:
        """Analyze project structure and provide insights"""
        try:
            path = Path(project_path)
            if not path.exists():
                return ToolResult(False, error=f"Project path not found: {project_path}", tool_name="analyze_project_structure")
            
            structure = {
                "root": str(path),
                "directories": [],
                "files": [],
                "languages": {},
                "frameworks": [],
                "dependencies": {},
                "config_files": [],
                "documentation": [],
                "tests": []
            }
            
            # Analyze directory structure
            for item in path.rglob("*"):
                if item.is_dir():
                    relative_path = item.relative_to(path)
                    structure["directories"].append(str(relative_path))
                elif item.is_file():
                    relative_path = item.relative_to(path)
                    file_str = str(relative_path)
                    structure["files"].append(file_str)
                    
                    # Analyze file types
                    suffix = item.suffix.lower()
                    if suffix:
                        structure["languages"][suffix] = structure["languages"].get(suffix, 0) + 1
                    
                    # Identify special files
                    file_name = item.name.lower()
                    if file_name in ['requirements.txt', 'package.json', 'setup.py', 'pyproject.toml', 'conda.yml']:
                        structure["dependencies"][file_name] = file_str
                    elif file_name in ['config.py', 'settings.py', '.env', 'config.json', 'config.yml']:
                        structure["config_files"].append(file_str)
                    elif file_name.startswith('readme') or suffix in ['.md', '.rst']:
                        structure["documentation"].append(file_str)
                    elif 'test' in file_name or file_str.startswith('tests/'):
                        structure["tests"].append(file_str)
            
            # Detect frameworks
            if 'package.json' in structure["dependencies"]:
                structure["frameworks"].append("Node.js")
            if 'requirements.txt' in structure["dependencies"] or 'setup.py' in structure["dependencies"]:
                structure["frameworks"].append("Python")
            if '.py' in structure["languages"] and structure["languages"]['.py'] > 0:
                structure["frameworks"].append("Python")
            
            return ToolResult(True, structure, tool_name="analyze_project_structure")
            
        except Exception as e:
            return ToolResult(False, error=f"Project analysis failed: {e}", tool_name="analyze_project_structure")
    
    # === UTILITY FUNCTIONS ===
    
    def log_tool_usage(self, tool_name: str, success: bool, details: str = "") -> None:
        """Log tool usage for learning and analysis"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "success": success,
            "details": details
        }
        self.tool_usage_log.append(log_entry)
        
        # Keep only recent entries (last 100)
        if len(self.tool_usage_log) > 100:
            self.tool_usage_log = self.tool_usage_log[-100:]
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get statistics on tool usage"""
        stats = {
            "total_uses": len(self.tool_usage_log),
            "success_rate": 0,
            "tool_counts": {},
            "recent_activity": self.tool_usage_log[-10:]
        }
        
        if self.tool_usage_log:
            successful = sum(1 for entry in self.tool_usage_log if entry["success"])
            stats["success_rate"] = successful / len(self.tool_usage_log)
            
            for entry in self.tool_usage_log:
                tool = entry["tool"]
                stats["tool_counts"][tool] = stats["tool_counts"].get(tool, 0) + 1
        
        return stats
    
    # === CONVENIENCE METHODS ===
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute any tool by name with parameters"""
        if not hasattr(self, tool_name):
            return ToolResult(False, error=f"Tool '{tool_name}' not found", tool_name=tool_name)
        
        try:
            tool_method = getattr(self, tool_name)
            result = tool_method(**kwargs)
            self.log_tool_usage(tool_name, result.success)
            return result
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            self.log_tool_usage(tool_name, False, error_msg)
            return ToolResult(False, error=error_msg, tool_name=tool_name)
