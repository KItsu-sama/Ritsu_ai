# core/code_analyzer.py
import ast
import re
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class CodeIssue:
    """Represents a detected code issue"""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    suggested_fix: str
    context: Dict[str, Any]

@dataclass 
class CodeChange:
    """Represents a proposed code change"""
    file_path: str
    start_line: int
    end_line: int
    old_content: str
    new_content: str
    reason: str

class CodeAnalyzer:
    """Advanced code analysis and modification system"""
    
    def __init__(self):
        self.code_patterns = self._load_problem_patterns()
        self.analysis_cache = {}
    
    def _load_problem_patterns(self) -> Dict[str, Dict]:
        """Load patterns for common code problems"""
        return {
            "unicode_subprocess": {
                "pattern": r"subprocess\.(run|Popen)\([^)]*text=True[^)]*\)",
                "severity": "medium",
                "description": "Subprocess call without proper encoding",
                "fix_template": "Add encoding='utf-8', errors='replace' parameters"
            },
            "infinite_loop_risk": {
                "pattern": r"while\s+True:\s*\n(?:\s*.*\n)*?\s*(?!.*sleep|break|return)",
                "severity": "high", 
                "description": "Potential infinite loop without sleep or break",
                "fix_template": "Add time.sleep() or proper exit condition"
            },
            "hardcoded_paths": {
                "pattern": r'["\'][C-Z]:\\\\[^"\']*["\']',
                "severity": "low",
                "description": "Hardcoded Windows path detected",
                "fix_template": "Use Path() or os.path.join() for cross-platform compatibility"
            },
            "missing_error_handling": {
                "pattern": r"(open\(|subprocess\.|requests\.)",
                "severity": "medium",
                "description": "Operation without try-catch block",
                "fix_template": "Add proper error handling with try-except"
            },
            "long_function": {
                "pattern": r"def\s+\w+\([^)]*\):(?:\s*.*\n){30,}",
                "severity": "low",
                "description": "Function too long (>30 lines)",
                "fix_template": "Consider breaking into smaller functions"
            }
        }
    
    def analyze_file(self, file_path: Path) -> List[CodeIssue]:
        """Analyze a Python file for potential issues"""
        issues = []
        
        if not file_path.exists() or file_path.suffix != '.py':
            return issues
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Pattern-based analysis
            for issue_type, pattern_info in self.code_patterns.items():
                pattern = pattern_info["pattern"]
                matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
                
                for match in matches:
                    # Find line number
                    line_num = content[:match.start()].count('\n') + 1
                    
                    issue = CodeIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        issue_type=issue_type,
                        severity=pattern_info["severity"],
                        description=pattern_info["description"],
                        suggested_fix=pattern_info["fix_template"],
                        context={
                            "matched_text": match.group(0),
                            "line_content": lines[line_num - 1] if line_num <= len(lines) else ""
                        }
                    )
                    issues.append(issue)
            
            # AST-based analysis
            try:
                tree = ast.parse(content)
                ast_issues = self._analyze_ast(tree, str(file_path), lines)
                issues.extend(ast_issues)
            except SyntaxError as e:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=e.lineno or 1,
                    issue_type="syntax_error",
                    severity="critical",
                    description=f"Syntax error: {e.msg}",
                    suggested_fix="Fix syntax error",
                    context={"error": str(e)}
                ))
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: str, lines: List[str]) -> List[CodeIssue]:
        """Analyze AST for code issues"""
        issues = []
        
        class IssueVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check function complexity
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    func_length = node.end_lineno - node.lineno
                    if func_length > 50:
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            issue_type="function_too_long",
                            severity="medium",
                            description=f"Function '{node.name}' is {func_length} lines long",
                            suggested_fix="Break into smaller functions",
                            context={"function_name": node.name, "length": func_length}
                        ))
                
                # Check for too many arguments
                if len(node.args.args) > 5:
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type="too_many_parameters",
                        severity="low",
                        description=f"Function '{node.name}' has {len(node.args.args)} parameters",
                        suggested_fix="Consider using a config object or reducing parameters",
                        context={"function_name": node.name, "param_count": len(node.args.args)}
                    ))
                
                self.generic_visit(node)
            
            def visit_Try(self, node):
                # Check for empty except blocks
                try:
                    for handler in node.handlers:
                        if not handler.body or (len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass)):
                            exc_type = "all"
                            try:
                                if handler.type and hasattr(handler.type, 'id'):
                                    exc_type = handler.type.id
                                elif handler.type:
                                    exc_type = str(handler.type)
                            except (AttributeError, TypeError):
                                pass
                            
                            issues.append(CodeIssue(
                                file_path=file_path,
                                line_number=handler.lineno,
                                issue_type="empty_except",
                                severity="medium",
                                description="Empty except block",
                                suggested_fix="Add proper error handling or logging",
                                context={"exception_type": exc_type}
                            ))
                except (AttributeError, TypeError) as e:
                    # Skip problematic try blocks
                    pass
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for unused imports (basic check)
                try:
                    for alias in node.names:
                        module_name = alias.name if hasattr(alias, 'name') else str(alias)
                        # Simple heuristic: if import not found in rest of file
                        if isinstance(module_name, str) and module_name not in ' '.join(lines[node.lineno:]):
                            issues.append(CodeIssue(
                                file_path=file_path,
                                line_number=node.lineno,
                                issue_type="unused_import",
                                severity="low",
                                description=f"Potentially unused import: {module_name}",
                                suggested_fix="Remove if not needed",
                                context={"module": module_name}
                            ))
                except (AttributeError, TypeError) as e:
                    # Skip problematic imports
                    pass
                
                self.generic_visit(node)
        
        visitor = IssueVisitor()
        visitor.visit(tree)
        
        return issues
    
    def suggest_code_fix(self, issue: CodeIssue) -> Optional[CodeChange]:
        """Suggest a specific code fix for an issue"""
        
        if issue.issue_type == "unicode_subprocess":
            return self._fix_subprocess_encoding(issue)
        elif issue.issue_type == "infinite_loop_risk":
            return self._fix_infinite_loop(issue)
        elif issue.issue_type == "missing_error_handling":
            return self._add_error_handling(issue)
        elif issue.issue_type == "empty_except":
            return self._fix_empty_except(issue)
        
        return None
    
    def _fix_subprocess_encoding(self, issue: CodeIssue) -> CodeChange:
        """Fix subprocess encoding issues"""
        file_path = Path(issue.file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            line_content = lines[issue.line_number - 1]
            
            # Add encoding parameters
            if 'encoding=' not in line_content and 'text=True' in line_content:
                new_content = line_content.replace(
                    'text=True',
                    'text=True, encoding=\'utf-8\', errors=\'replace\''
                )
                
                return CodeChange(
                    file_path=issue.file_path,
                    start_line=issue.line_number,
                    end_line=issue.line_number,
                    old_content=line_content.strip(),
                    new_content=new_content.strip(),
                    reason="Fix Unicode encoding issues in subprocess calls"
                )
                
        except Exception as e:
            print(f"Error generating fix for {issue.file_path}: {e}")
        
        return None
    
    def _fix_infinite_loop(self, issue: CodeIssue) -> CodeChange:
        """Fix potential infinite loops"""
        file_path = Path(issue.file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find the while loop and add sleep
            loop_line = issue.line_number - 1
            indent = len(lines[loop_line]) - len(lines[loop_line].lstrip())
            
            # Look for the first line inside the loop
            insert_line = loop_line + 1
            while insert_line < len(lines) and lines[insert_line].strip() == '':
                insert_line += 1
            
            if insert_line < len(lines):
                inner_indent = len(lines[insert_line]) - len(lines[insert_line].lstrip())
                sleep_line = ' ' * inner_indent + 'time.sleep(0.1)  # Prevent high CPU usage\n'
                
                return CodeChange(
                    file_path=issue.file_path,
                    start_line=insert_line + 1,
                    end_line=insert_line + 1,
                    old_content='',
                    new_content=sleep_line,
                    reason="Add sleep to prevent high CPU usage in loop"
                )
                
        except Exception as e:
            print(f"Error generating loop fix: {e}")
        
        return None
    
    def _add_error_handling(self, issue: CodeIssue) -> CodeChange:
        """Add error handling to risky operations"""
        file_path = Path(issue.file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            problem_line = lines[issue.line_number - 1]
            indent = ' ' * (len(problem_line) - len(problem_line.lstrip()))
            
            # Wrap in try-except
            new_content = f"{indent}try:\n{problem_line}{indent}except Exception as e:\n{indent}    logging.error(f'Error: {{e}}')\n{indent}    # Handle error appropriately\n"
            
            return CodeChange(
                file_path=issue.file_path,
                start_line=issue.line_number,
                end_line=issue.line_number,
                old_content=problem_line.strip(),
                new_content=new_content.strip(),
                reason="Add error handling to prevent crashes"
            )
            
        except Exception as e:
            print(f"Error generating error handling fix: {e}")
        
        return None
    
    def _fix_empty_except(self, issue: CodeIssue) -> CodeChange:
        """Fix empty except blocks"""
        file_path = Path(issue.file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find the except block and add logging
            except_line = issue.line_number - 1
            indent = len(lines[except_line]) - len(lines[except_line].lstrip())
            
            # Add logging to the except block
            new_content = f"{' ' * (indent + 4)}logging.error(f'Error in {{__name__}}: {{e}}')\n"
            
            return CodeChange(
                file_path=issue.file_path,
                start_line=issue.line_number + 1,
                end_line=issue.line_number + 1,
                old_content='pass',
                new_content=new_content.strip(),
                reason="Add logging to empty except block"
            )
            
        except Exception as e:
            print(f"Error generating except fix: {e}")
        
        return None
    
    def apply_code_change(self, change: CodeChange) -> bool:
        """Apply a code change to a file"""
        try:
            file_path = Path(change.file_path)
            
            # Backup the original file
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
            
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Apply the change
            if change.start_line == change.end_line and change.old_content == '':
                # Insert new line
                lines.insert(change.start_line - 1, change.new_content + '\n')
            else:
                # Replace lines
                for i in range(change.start_line - 1, change.end_line):
                    if i < len(lines):
                        lines[i] = change.new_content + '\n'
            
            # Write modified content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print(f"Applied fix to {file_path}: {change.reason}")
            return True
            
        except Exception as e:
            print(f"Failed to apply fix to {change.file_path}: {e}")
            return False
    
    def analyze_project(self, project_path: Path) -> Dict[str, Any]:
        """Analyze entire project for issues"""
        analysis = {
            "timestamp": "2025-08-19T15:16:56Z",
            "project_path": str(project_path),
            "files_analyzed": 0,
            "issues_found": [],
            "summary": {},
            "suggested_fixes": []
        }
        
        # Find all Python files
        python_files = list(project_path.rglob("*.py"))
        analysis["files_analyzed"] = len(python_files)
        
        all_issues = []
        for py_file in python_files:
            file_issues = self.analyze_file(py_file)
            all_issues.extend(file_issues)
        
        # Categorize issues
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        issue_types = {}
        
        for issue in all_issues:
            severity_counts[issue.severity] += 1
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
            
            analysis["issues_found"].append({
                "file": issue.file_path,
                "line": issue.line_number,
                "type": issue.issue_type,
                "severity": issue.severity,
                "description": issue.description,
                "fix": issue.suggested_fix
            })
            
            # Generate specific fix
            fix = self.suggest_code_fix(issue)
            if fix:
                analysis["suggested_fixes"].append({
                    "file": fix.file_path,
                    "change": fix.reason,
                    "priority": severity_counts[issue.severity]
                })
        
        analysis["summary"] = {
            "total_issues": len(all_issues),
            "by_severity": severity_counts,
            "by_type": issue_types,
            "files_with_issues": len(set(issue.file_path for issue in all_issues))
        }
        
        return analysis
