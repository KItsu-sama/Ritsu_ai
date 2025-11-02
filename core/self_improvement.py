# core/self_improvement.py
"""
Self-Improvement System for Ritsu AI
Allows Ritsu to analyze, understand, and improve its own codebase over time
"""

import ast
import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib

@dataclass
class CodeAnalysis:
    """Analysis of code quality and potential improvements"""
    file_path: str
    complexity_score: float
    maintainability_score: float
    issues: List[Dict]
    suggestions: List[Dict]
    metrics: Dict[str, Any]
    timestamp: str

@dataclass
class Improvement:
    """Represents a potential improvement to the codebase"""
    id: str
    category: str  # performance, maintainability, functionality, security
    priority: str  # critical, high, medium, low
    title: str
    description: str
    target_files: List[str]
    proposed_changes: List[Dict]
    benefits: List[str]
    risks: List[str]
    estimated_effort: str
    success_probability: float
    created_at: str
    status: str = "proposed"  # proposed, approved, implemented, tested, rejected

@dataclass
class SelfAnalysisResult:
    """Result of self-analysis"""
    overall_health: float  # 0-100 score
    code_quality: float
    performance_score: float
    maintainability: float
    security_score: float
    identified_improvements: List[Improvement]
    analysis_timestamp: str

class SelfImprovementEngine:
    """Engine for analyzing and improving Ritsu's own codebase"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        self.analysis_history = []
        self.implemented_improvements = []
        self.rejected_improvements = []
        self.improvement_db_file = self.project_root / "improvements.json"
        self.load_improvement_database()
        
    def load_improvement_database(self):
        """Load previous improvements and analysis history"""
        try:
            if self.improvement_db_file.exists():
                with open(self.improvement_db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.analysis_history = data.get('analysis_history', [])
                    self.implemented_improvements = data.get('implemented', [])
                    self.rejected_improvements = data.get('rejected', [])
        except Exception as e:
            self.logger.warning(f"Failed to load improvement database: {e}")
    
    def save_improvement_database(self):
        """Save improvements and analysis to persistent storage"""
        try:
            data = {
                'analysis_history': self.analysis_history[-50:],  # Keep last 50
                'implemented': self.implemented_improvements[-100:],  # Keep last 100
                'rejected': self.rejected_improvements[-50:],  # Keep last 50
                'last_updated': datetime.now().isoformat()
            }
            with open(self.improvement_db_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save improvement database: {e}")
    
    def analyze_self(self, include_tests: bool = True) -> SelfAnalysisResult:
        """Perform comprehensive self-analysis of the codebase"""
        try:
            self.logger.info("Starting self-analysis...")
            
            # Find all Python files in the project
            python_files = list(self.project_root.rglob("*.py"))
            if not include_tests:
                python_files = [f for f in python_files if not self._is_test_file(f)]
            
            # Analyze each file
            file_analyses = []
            for py_file in python_files:
                try:
                    analysis = self._analyze_file(py_file)
                    file_analyses.append(analysis)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {py_file}: {e}")
            
            # Calculate overall metrics
            overall_health = self._calculate_overall_health(file_analyses)
            code_quality = self._calculate_code_quality(file_analyses)
            performance_score = self._calculate_performance_score(file_analyses)
            maintainability = self._calculate_maintainability(file_analyses)
            security_score = self._calculate_security_score(file_analyses)
            
            # Identify potential improvements
            improvements = self._identify_improvements(file_analyses)
            
            result = SelfAnalysisResult(
                overall_health=overall_health,
                code_quality=code_quality,
                performance_score=performance_score,
                maintainability=maintainability,
                security_score=security_score,
                identified_improvements=improvements,
                analysis_timestamp=datetime.now().isoformat()
            )
            
            # Save to history
            self.analysis_history.append(asdict(result))
            self.save_improvement_database()
            
            self.logger.info(f"Self-analysis complete. Health score: {overall_health:.1f}/100")
            return result
            
        except Exception as e:
            self.logger.error(f"Self-analysis failed: {e}")
            raise
    
    def _analyze_file(self, file_path: Path) -> CodeAnalysis:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Calculate metrics
            metrics = self._calculate_file_metrics(tree, content)
            
            # Identify issues
            issues = self._identify_file_issues(tree, content, file_path)
            
            # Generate suggestions
            suggestions = self._generate_file_suggestions(tree, content, metrics, issues)
            
            # Calculate scores
            complexity_score = self._calculate_complexity_score(metrics)
            maintainability_score = self._calculate_maintainability_score(metrics, issues)
            
            return CodeAnalysis(
                file_path=str(file_path),
                complexity_score=complexity_score,
                maintainability_score=maintainability_score,
                issues=issues,
                suggestions=suggestions,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"File analysis failed for {file_path}: {e}")
            raise
    
    def _calculate_file_metrics(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Calculate various metrics for a file"""
        metrics = {
            'lines_of_code': len(content.split('\n')),
            'functions': 0,
            'classes': 0,
            'complexity': 0,
            'max_function_length': 0,
            'docstring_coverage': 0,
            'import_count': 0,
            'nested_depth': 0
        }
        
        class MetricsVisitor(ast.NodeVisitor):
            def __init__(self):
                self.function_lengths = []
                self.docstrings = 0
                self.total_functions = 0
                self.current_depth = 0
                self.max_depth = 0
                
            def visit_FunctionDef(self, node):
                metrics['functions'] += 1
                self.total_functions += 1
                
                # Calculate function length
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    length = node.end_lineno - node.lineno
                    self.function_lengths.append(length)
                    metrics['max_function_length'] = max(metrics['max_function_length'], length)
                
                # Check for docstring (handle both old and new AST formats)
                if node.body and isinstance(node.body[0], ast.Expr):
                    value = node.body[0].value
                    if (isinstance(value, ast.Str) or  # Python < 3.8
                        (isinstance(value, ast.Constant) and isinstance(value.value, str))):  # Python >= 3.8
                        self.docstrings += 1
                
                # Calculate complexity (simplified McCabe)
                complexity = self._calculate_function_complexity(node)
                metrics['complexity'] += complexity
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                metrics['classes'] += 1
                # Check for class docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Str)):
                    self.docstrings += 1
                self.generic_visit(node)
            
            def visit_Import(self, node):
                metrics['import_count'] += len(node.names)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                metrics['import_count'] += len(node.names)
                self.generic_visit(node)
            
            def visit(self, node):
                # Preserve the normal dispatch to specialized visit_* methods
                return super().visit(node)

            def generic_visit(self, node):
                # Track nesting depth for compound statements when recursing
                is_compound = isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try))
                if is_compound:
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    super().generic_visit(node)
                    self.current_depth -= 1
                else:
                    super().generic_visit(node)
            
            def _calculate_function_complexity(self, node):
                """Calculate cyclomatic complexity for a function"""
                complexity = 1  # Base complexity
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                return complexity
        
        visitor = MetricsVisitor()
        visitor.visit(tree)
        
        # Calculate derived metrics
        if visitor.total_functions > 0:
            metrics['docstring_coverage'] = (visitor.docstrings / visitor.total_functions) * 100
        
        metrics['avg_function_length'] = sum(visitor.function_lengths) / len(visitor.function_lengths) if visitor.function_lengths else 0
        metrics['nested_depth'] = visitor.max_depth
        
        return metrics
    
    def _identify_file_issues(self, tree: ast.AST, content: str, file_path: Path) -> List[Dict]:
        """Identify issues in a file"""
        issues = []
        
        # Use existing code analyzer if available
        try:
            from core.code_analyzer import CodeAnalyzer
            analyzer = CodeAnalyzer()
            code_issues = analyzer.analyze_file(file_path)
            
            for issue in code_issues:
                issues.append({
                    'type': issue.issue_type,
                    'severity': issue.severity,
                    'description': issue.description,
                    'line': issue.line_number,
                    'fix_suggestion': issue.suggested_fix
                })
        except ImportError:
            self.logger.warning("CodeAnalyzer not available for detailed issue detection")
        
        # Add self-improvement specific checks
        issues.extend(self._check_self_improvement_issues(tree, content))
        
        return issues
    
    def _check_self_improvement_issues(self, tree: ast.AST, content: str) -> List[Dict]:
        """Check for issues specific to self-improvement"""
        issues = []
        
        # Check for outdated print() usage per-line and report correct line numbers
        lines = content.split('\n')
        for i, ln in enumerate(lines, 1):
            if 'print(' in ln and 'logging.' not in content:
                issues.append({
                    'type': 'outdated_logging',
                    'severity': 'medium',
                    'description': 'Using print() instead of proper logging',
                    'line': i,
                    'fix_suggestion': 'Replace print() with logging.info() or appropriate level'
                })
        
        # Check for hardcoded values
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if any(pattern in line for pattern in ['= "http://', '= "https://', 'password =', 'api_key =']):
                issues.append({
                    'type': 'hardcoded_config',
                    'severity': 'high',
                    'description': 'Hardcoded configuration values detected',
                    'line': i,
                    'fix_suggestion': 'Move configuration to config file or environment variables'
                })
        
        # Check for missing error handling in critical operations
        class ErrorHandlingChecker(ast.NodeVisitor):
            def visit_Call(self, node):
                if (isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id in ['subprocess', 'requests', 'urllib']):
                    
                    # Check if this call is inside a try block
                    # This is a simplified check
                    issues.append({
                        'type': 'missing_error_handling',
                        'severity': 'medium',
                        'description': f'External call without error handling: {node.func.attr}',
                        'line': node.lineno,
                        'fix_suggestion': 'Wrap in try-except block'
                    })
                
                self.generic_visit(node)
        
        checker = ErrorHandlingChecker()
        checker.visit(tree)
        
        return issues
    
    def _generate_file_suggestions(self, tree: ast.AST, content: str, metrics: Dict, issues: List[Dict]) -> List[Dict]:
        """Generate improvement suggestions for a file"""
        suggestions = []
        
        # Function length suggestions
        if metrics['max_function_length'] > 50:
            suggestions.append({
                'type': 'refactor',
                'priority': 'medium',
                'description': f'Function too long ({metrics["max_function_length"]} lines). Consider breaking into smaller functions.',
                'benefit': 'Improved readability and maintainability'
            })
        
        # Complexity suggestions
        if metrics['complexity'] > metrics['functions'] * 10:  # Average complexity > 10
            suggestions.append({
                'type': 'refactor',
                'priority': 'high',
                'description': 'High complexity detected. Consider simplifying logic.',
                'benefit': 'Reduced complexity, easier testing and maintenance'
            })
        
        # Documentation suggestions
        if metrics['docstring_coverage'] < 50:
            suggestions.append({
                'type': 'documentation',
                'priority': 'medium',
                'description': f'Low docstring coverage ({metrics["docstring_coverage"]:.1f}%). Add more documentation.',
                'benefit': 'Better code understanding and maintenance'
            })
        
        # Import optimization
        if metrics['import_count'] > 20:
            suggestions.append({
                'type': 'optimization',
                'priority': 'low',
                'description': f'Many imports ({metrics["import_count"]}). Consider organizing or reducing dependencies.',
                'benefit': 'Faster import times and cleaner code'
            })
        
        return suggestions
    
    def _identify_improvements(self, file_analyses: List[CodeAnalysis]) -> List[Improvement]:
        """Identify potential improvements across the codebase"""
        improvements = []
        
        # Aggregate issues and suggestions
        all_issues = []
        all_suggestions = []
        
        for analysis in file_analyses:
            all_issues.extend([(analysis.file_path, issue) for issue in analysis.issues])
            all_suggestions.extend([(analysis.file_path, suggestion) for suggestion in analysis.suggestions])
        
        # Group similar issues
        issue_groups = self._group_similar_issues(all_issues)
        
        for issue_type, issue_list in issue_groups.items():
            if len(issue_list) >= 3:  # If issue appears in multiple files
                improvement = self._create_improvement_from_issues(issue_type, issue_list)
                improvements.append(improvement)
        
        # Create improvements from high-impact suggestions
        for file_path, suggestion in all_suggestions:
            if suggestion.get('priority') == 'high':
                improvement = self._create_improvement_from_suggestion(file_path, suggestion)
                improvements.append(improvement)
        
        # Add architecture-level improvements
        improvements.extend(self._identify_architecture_improvements(file_analyses))
        
        return improvements
    
    def _group_similar_issues(self, issues: List[Tuple[str, Dict]]) -> Dict[str, List]:
        """Group similar issues together"""
        groups = {}
        
        for file_path, issue in issues:
            issue_type = issue.get('type', 'unknown')
            if issue_type not in groups:
                groups[issue_type] = []
            groups[issue_type].append((file_path, issue))
        
        return groups
    
    def _create_improvement_from_issues(self, issue_type: str, issue_list: List) -> Improvement:
        """Create an improvement from a group of similar issues"""
        affected_files = [file_path for file_path, _ in issue_list]
        
        severity_map = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        max_severity = max([severity_map.get(issue.get('severity', 'low'), 1) for _, issue in issue_list])
        priority = ['low', 'medium', 'high', 'critical'][max_severity - 1]
        
        improvement_id = hashlib.md5(f"{issue_type}_{len(issue_list)}_{datetime.now()}".encode()).hexdigest()[:8]
        
        return Improvement(
            id=improvement_id,
            category='maintainability',
            priority=priority,
            title=f"Fix {issue_type} issues across {len(affected_files)} files",
            description=f"Address {len(issue_list)} instances of {issue_type} issues in the codebase",
            target_files=affected_files,
            proposed_changes=[{
                'type': 'fix_issue',
                'issue_type': issue_type,
                'files': affected_files
            }],
            benefits=[
                'Improved code quality',
                'Reduced maintenance burden',
                'Better error handling'
            ],
            risks=['Potential introduction of new bugs if not carefully implemented'],
            estimated_effort='2-4 hours',
            success_probability=0.85,
            created_at=datetime.now().isoformat()
        )
    
    def _create_improvement_from_suggestion(self, file_path: str, suggestion: Dict) -> Improvement:
        """Create an improvement from a high-priority suggestion"""
        improvement_id = hashlib.md5(f"{file_path}_{suggestion['type']}_{datetime.now()}".encode()).hexdigest()[:8]
        
        return Improvement(
            id=improvement_id,
            category=suggestion.get('type', 'maintainability'),
            priority=suggestion.get('priority', 'medium'),
            title=f"Improve {suggestion['type']} in {Path(file_path).name}",
            description=suggestion.get('description', 'No description'),
            target_files=[file_path],
            proposed_changes=[{
                'type': suggestion.get('type'),
                'file': file_path,
                'description': suggestion.get('description')
            }],
            benefits=[suggestion.get('benefit', 'Code improvement')],
            risks=['Minimal risk if implemented carefully'],
            estimated_effort='1-2 hours',
            success_probability=0.90,
            created_at=datetime.now().isoformat()
        )
    
    def _identify_architecture_improvements(self, file_analyses: List[CodeAnalysis]) -> List[Improvement]:
        """Identify architecture-level improvements"""
        improvements = []
        
        # Check for missing components
        has_tests = any('test' in analysis.file_path.lower() for analysis in file_analyses)
        has_logging = any('logging' in analysis.file_path.lower() for analysis in file_analyses)
        has_config = any('config' in analysis.file_path.lower() for analysis in file_analyses)
        
        if not has_tests:
            improvements.append(Improvement(
                id=hashlib.md5(f"add_tests_{datetime.now()}".encode()).hexdigest()[:8],
                category='functionality',
                priority='high',
                title='Add comprehensive test suite',
                description='The project lacks comprehensive test coverage',
                target_files=[],
                proposed_changes=[{
                    'type': 'add_tests',
                    'description': 'Create unit tests for all major components'
                }],
                benefits=['Better code reliability', 'Easier refactoring', 'Bug prevention'],
                risks=['Initial time investment'],
                estimated_effort='1-2 days',
                success_probability=0.95,
                created_at=datetime.now().isoformat()
            ))
        
        # Check for performance optimization opportunities
        total_complexity = sum(analysis.metrics.get('complexity', 0) for analysis in file_analyses)
        avg_complexity = total_complexity / len(file_analyses) if file_analyses else 0
        
        if avg_complexity > 15:
            improvements.append(Improvement(
                id=hashlib.md5(f"optimize_performance_{datetime.now()}".encode()).hexdigest()[:8],
                category='performance',
                priority='medium',
                title='Optimize high-complexity functions',
                description=f'Average complexity is {avg_complexity:.1f}, consider optimization',
                target_files=[a.file_path for a in file_analyses if a.metrics.get('complexity', 0) > avg_complexity],
                proposed_changes=[{
                    'type': 'performance_optimization',
                    'description': 'Refactor complex functions and optimize algorithms'
                }],
                benefits=['Better performance', 'Reduced resource usage', 'Improved scalability'],
                risks=['Potential behavior changes if not carefully implemented'],
                estimated_effort='4-8 hours',
                success_probability=0.75,
                created_at=datetime.now().isoformat()
            ))
        
        return improvements
    
    def implement_improvement(self, improvement_id: str, auto_apply: bool = False) -> bool:
        """Implement a specific improvement"""
        try:
            # Find the improvement
            improvement = None
            for analysis in self.analysis_history:
                for imp in analysis.get('identified_improvements', []):
                    if imp.get('id') == improvement_id:
                        improvement = Improvement(**imp)
                        break
                if improvement:
                    break
            
            if not improvement:
                self.logger.error(f"Improvement {improvement_id} not found")
                return False
            
            self.logger.info(f"Implementing improvement: {improvement.title}")
            
            # Generate and apply fixes based on improvement type
            success = self._apply_improvement_changes(improvement, auto_apply)
            
            if success:
                improvement.status = "implemented"
                self.implemented_improvements.append(asdict(improvement))
                self.logger.info(f"Successfully implemented improvement: {improvement.title}")
            else:
                self.logger.error(f"Failed to implement improvement: {improvement.title}")
            
            self.save_improvement_database()
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to implement improvement {improvement_id}: {e}")
            return False
    
    def _apply_improvement_changes(self, improvement: Improvement, auto_apply: bool) -> bool:
        """Apply the changes for an improvement"""
        try:
            from core.code_generator import CodeGenerator
            from core.tools import RitsuToolSystem
            
            generator = CodeGenerator()
            tools = RitsuToolSystem()
            
            for change in improvement.proposed_changes:
                change_type = change.get('type')
                
                if change_type == 'fix_issue':
                    # Apply issue fixes
                    issue_type = change.get('issue_type')
                    files = change.get('files', [])
                    
                    for file_path in files:
                        self._apply_issue_fix(file_path, issue_type, generator, tools)
                
                elif change_type == 'add_tests':
                    # Generate test files
                    self._generate_test_files(generator, tools)
                
                elif change_type == 'performance_optimization':
                    # Apply performance optimizations
                    self._apply_performance_optimizations(improvement.target_files, generator, tools)
                
                elif change_type in ['refactor', 'documentation', 'optimization']:
                    # Apply general improvements
                    self._apply_general_improvements(change, generator, tools)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply improvement changes: {e}")
            return False
    
    def _apply_issue_fix(self, file_path: str, issue_type: str, generator, tools):
        """Apply fix for a specific issue type"""
        # This would use the code generator to create and apply fixes
        # Implementation depends on specific issue types
        self.logger.info(f"Applying {issue_type} fix to {file_path}")
    
    def _generate_test_files(self, generator, tools):
        """Generate test files for the project"""
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            if not self._is_test_file(py_file) and 'test_' not in py_file.name:
                try:
                    test_code = generator.generate_test_suite(str(py_file))
                    if test_code.file_path:
                        test_path = self.project_root / test_code.file_path
                        tools.create_file(str(test_path), test_code.code, f"Test for {py_file.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate test for {py_file}: {e}")
    
    def _apply_performance_optimizations(self, target_files: List[str], generator, tools):
        """Apply performance optimizations to target files"""
        for file_path in target_files:
            try:
                optimized = generator.refactor_code(file_path, "optimize_performance")
                if optimized.code:
                    tools.edit_files([{
                        'file_path': file_path,
                        'search': '',  # Full file replacement for major refactoring
                        'replace': optimized.code,
                        'search_start_line_number': 1
                    }])
            except Exception as e:
                self.logger.warning(f"Failed to optimize {file_path}: {e}")
    
    def _apply_general_improvements(self, change: Dict, generator, tools):
        """Apply general improvements"""
        # Implementation for general improvements
        self.logger.info(f"Applying general improvement: {change.get('description')}")
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if a file is a test file"""
        return (file_path.name.startswith('test_') or 
                file_path.name.endswith('_test.py') or 
                'test' in file_path.parts)
    
    def _calculate_overall_health(self, analyses: List[CodeAnalysis]) -> float:
        """Calculate overall codebase health score"""
        if not analyses:
            return 0.0
        
        quality_scores = [a.maintainability_score for a in analyses]
        complexity_scores = [100 - min(a.complexity_score, 100) for a in analyses]  # Invert complexity
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        
        # Weight quality more heavily
        return (avg_quality * 0.7 + avg_complexity * 0.3)
    
    def _calculate_code_quality(self, analyses: List[CodeAnalysis]) -> float:
        """Calculate code quality score"""
        if not analyses:
            return 0.0
        
        scores = []
        for analysis in analyses:
            # Base score from maintainability
            score = analysis.maintainability_score
            
            # Penalty for critical issues
            critical_issues = sum(1 for issue in analysis.issues if issue.get('severity') == 'critical')
            score -= critical_issues * 10
            
            scores.append(max(0, score))
        
        return sum(scores) / len(scores)
    
    def _calculate_performance_score(self, analyses: List[CodeAnalysis]) -> float:
        """Calculate performance score"""
        if not analyses:
            return 0.0
        
        scores = []
        for analysis in analyses:
            score = 100
            
            # Penalty for high complexity
            complexity = analysis.metrics.get('complexity', 0)
            functions = analysis.metrics.get('functions', 1)
            avg_complexity = complexity / functions if functions > 0 else 0
            
            if avg_complexity > 10:
                score -= (avg_complexity - 10) * 5
            
            # Penalty for very long functions
            max_length = analysis.metrics.get('max_function_length', 0)
            if max_length > 50:
                score -= (max_length - 50) * 0.5
            
            scores.append(max(0, score))
        
        return sum(scores) / len(scores)
    
    def _calculate_maintainability(self, analyses: List[CodeAnalysis]) -> float:
        """Calculate maintainability score"""
        if not analyses:
            return 0.0
        
        return sum(a.maintainability_score for a in analyses) / len(analyses)
    
    def _calculate_security_score(self, analyses: List[CodeAnalysis]) -> float:
            """Calculate security score"""
            if not analyses:
                return 0.0
            
            scores = []
            for analysis in analyses:
                score = 100.0
                
                # Penalties based on detected issue types
                issue_counts = {}
                for issue in analysis.issues:
                    issue_type = issue.get('type')
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                
                # High-severity penalties
                score -= issue_counts.get('hardcoded_config', 0) * 20.0  # Significant penalty
                
                # Medium-severity penalties
                score -= issue_counts.get('missing_error_handling', 0) * 5.0
                
                # Low-severity penalties (e.g., using print for sensitive info, though not currently checked)
                score -= issue_counts.get('outdated_logging', 0) * 1.0 
                
                # Penalty for low documentation/lack of tests (indirect security risk)
                doc_coverage = analysis.metrics.get('docstring_coverage', 0)
                if doc_coverage < 30:
                    score -= 5.0
                
                scores.append(max(0.0, score))
            
            # Security score is the average of file scores
            return sum(scores) / len(scores)

    def _calculate_complexity_score(self, metrics: Dict) -> float:
        """Calculate complexity score for a file"""
        complexity = metrics.get('complexity', 0)
        functions = metrics.get('functions', 1)
        avg_complexity = complexity / functions if functions > 0 else 0
        
        # Scale complexity to 0-100 score (higher = more complex)
        return min(avg_complexity * 10, 100)
    
    def _calculate_maintainability_score(self, metrics: Dict, issues: List[Dict]) -> float:
        """Calculate maintainability score for a file"""
        score = 100
        
        # Penalty for issues
        critical_issues = sum(1 for issue in issues if issue.get('severity') == 'critical')
        high_issues = sum(1 for issue in issues if issue.get('severity') == 'high')
        medium_issues = sum(1 for issue in issues if issue.get('severity') == 'medium')
        
        score -= critical_issues * 20
        score -= high_issues * 10
        score -= medium_issues * 5
        
        # Penalty for poor documentation
        doc_coverage = metrics.get('docstring_coverage', 0)
        if doc_coverage < 50:
            score -= (50 - doc_coverage) * 0.5
        
        # Penalty for long functions
        max_func_length = metrics.get('max_function_length', 0)
        if max_func_length > 50:
            score -= (max_func_length - 50) * 0.3
        
        return max(0, score)
    
    def get_improvement_suggestions(self, priority_filter: str = None) -> List[Dict]:
        """Get current improvement suggestions"""
        if not self.analysis_history:
            return []
        
        latest_analysis = self.analysis_history[-1]
        improvements = latest_analysis.get('identified_improvements', [])
        
        if priority_filter:
            improvements = [imp for imp in improvements if imp.get('priority') == priority_filter]
        
        return improvements
    
    def get_health_trend(self, days: int = 30) -> Dict[str, List]:
        """Get health trend over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_analyses = [
            analysis for analysis in self.analysis_history
            if datetime.fromisoformat(analysis['analysis_timestamp']) > cutoff_date
        ]
        
        trend = {
            'dates': [],
            'overall_health': [],
            'code_quality': [],
            'performance': [],
            'maintainability': []
        }
        
        for analysis in recent_analyses:
            trend['dates'].append(analysis['analysis_timestamp'])
            trend['overall_health'].append(analysis['overall_health'])
            trend['code_quality'].append(analysis['code_quality'])
            trend['performance'].append(analysis['performance_score'])
            trend['maintainability'].append(analysis['maintainability'])
        
        return trend
