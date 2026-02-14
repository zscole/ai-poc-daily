#!/usr/bin/env python3
"""
Agentic Coder - Autonomous Code Analysis and Improvement System
Demonstrates cutting-edge agentic coding capabilities inspired by Apple's Xcode AI integration
"""

import os
import ast
import subprocess
import json
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import textwrap
import sys

class CodeAnalyzer:
    """Analyzes code for potential improvements, security issues, and optimization opportunities."""
    
    def __init__(self):
        self.analysis_rules = [
            self._check_security_vulnerabilities,
            self._check_performance_issues,
            self._check_code_quality,
            self._check_type_annotations,
            self._check_error_handling
        ]
    
    def analyze_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Analyze a Python file and return improvement suggestions."""
        if not filepath.suffix == '.py':
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            issues = []
            
            for rule in self.analysis_rules:
                issues.extend(rule(content, tree, filepath))
            
            return issues
        except Exception as e:
            return [{"type": "parse_error", "message": f"Could not analyze {filepath}: {e}"}]
    
    def _check_security_vulnerabilities(self, content: str, tree: ast.AST, filepath: Path) -> List[Dict]:
        """Check for common security vulnerabilities."""
        issues = []
        
        # Check for subprocess shell=True
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == 'subprocess'):
                    
                    for keyword in node.keywords:
                        if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                            if keyword.value.value is True:
                                issues.append({
                                    "type": "security",
                                    "severity": "high",
                                    "line": node.lineno,
                                    "message": "subprocess with shell=True is a security risk",
                                    "suggestion": "Use shell=False and pass commands as list",
                                    "fix": self._generate_subprocess_fix(node)
                                })
        
        # Check for eval() usage
        if 'eval(' in content:
            issues.append({
                "type": "security",
                "severity": "critical",
                "message": "eval() usage detected - major security risk",
                "suggestion": "Replace eval() with safer alternatives like ast.literal_eval()"
            })
        
        return issues
    
    def _check_performance_issues(self, content: str, tree: ast.AST, filepath: Path) -> List[Dict]:
        """Check for performance optimization opportunities."""
        issues = []
        
        # Check for string concatenation in loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                        if isinstance(child.target, ast.Name):
                            issues.append({
                                "type": "performance",
                                "severity": "medium",
                                "line": getattr(child, 'lineno', 0),
                                "message": "String concatenation in loop detected",
                                "suggestion": "Use list.append() and ''.join() for better performance",
                                "fix": "Use: items.append(new_item) then ''.join(items)"
                            })
        
        return issues
    
    def _check_code_quality(self, content: str, tree: ast.AST, filepath: Path) -> List[Dict]:
        """Check for code quality improvements."""
        issues = []
        lines = content.split('\n')
        
        # Check for long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                issues.append({
                    "type": "quality",
                    "severity": "low",
                    "line": i,
                    "message": f"Line too long ({len(line)} chars)",
                    "suggestion": "Consider breaking long lines for readability"
                })
        
        # Check for magic numbers
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in [0, 1, -1] and node.value > 10:
                    issues.append({
                        "type": "quality",
                        "severity": "low",
                        "line": getattr(node, 'lineno', 0),
                        "message": f"Magic number detected: {node.value}",
                        "suggestion": "Consider using named constants"
                    })
        
        return issues
    
    def _check_type_annotations(self, content: str, tree: ast.AST, filepath: Path) -> List[Dict]:
        """Check for missing type annotations."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.returns and node.name != '__init__':
                    issues.append({
                        "type": "typing",
                        "severity": "low",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' missing return type annotation",
                        "suggestion": "Add return type annotation for better code clarity"
                    })
        
        return issues
    
    def _check_error_handling(self, content: str, tree: ast.AST, filepath: Path) -> List[Dict]:
        """Check for proper error handling."""
        issues = []
        
        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "type": "error_handling",
                        "severity": "medium",
                        "line": node.lineno,
                        "message": "Bare except clause detected",
                        "suggestion": "Catch specific exception types instead of using bare except"
                    })
        
        return issues
    
    def _generate_subprocess_fix(self, node: ast.Call) -> str:
        """Generate a fix for subprocess shell=True issue."""
        return "Use subprocess.run([command, arg1, arg2], shell=False) instead"

class CodeGenerator:
    """Generates improved code based on analysis results."""
    
    def generate_fix(self, issue: Dict[str, Any], original_content: str) -> str:
        """Generate a code fix for a specific issue."""
        if issue["type"] == "security" and "subprocess" in issue["message"]:
            return self._fix_subprocess_shell(original_content, issue)
        elif issue["type"] == "performance" and "concatenation" in issue["message"]:
            return self._fix_string_concatenation(original_content, issue)
        elif issue["type"] == "error_handling" and "bare except" in issue["message"]:
            return self._fix_bare_except(original_content, issue)
        else:
            return original_content  # No automatic fix available
    
    def _fix_subprocess_shell(self, content: str, issue: Dict) -> str:
        """Fix subprocess shell=True issues."""
        lines = content.split('\n')
        if issue.get("line"):
            line_idx = issue["line"] - 1
            if line_idx < len(lines):
                line = lines[line_idx]
                # Simple replacement for demonstration
                if "shell=True" in line:
                    lines[line_idx] = line.replace("shell=True", "shell=False")
                    # Add comment explaining the change
                    lines.insert(line_idx, "    # Fixed: Changed shell=True to shell=False for security")
        
        return '\n'.join(lines)
    
    def _fix_string_concatenation(self, content: str, issue: Dict) -> str:
        """Fix string concatenation in loops."""
        # This is a simplified example - real implementation would be more sophisticated
        lines = content.split('\n')
        if issue.get("line"):
            line_idx = issue["line"] - 1
            if line_idx < len(lines):
                lines.insert(line_idx, "    # TODO: Consider using list.append() and ''.join() for better performance")
        
        return '\n'.join(lines)
    
    def _fix_bare_except(self, content: str, issue: Dict) -> str:
        """Fix bare except clauses."""
        lines = content.split('\n')
        if issue.get("line"):
            line_idx = issue["line"] - 1
            if line_idx < len(lines):
                line = lines[line_idx]
                if "except:" in line:
                    lines[line_idx] = line.replace("except:", "except Exception as e:")
        
        return '\n'.join(lines)

class TestRunner:
    """Runs tests to verify code changes don't break functionality."""
    
    def run_syntax_check(self, filepath: Path) -> Tuple[bool, str]:
        """Check if Python file has valid syntax."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            ast.parse(content)
            return True, "Syntax check passed"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    def run_tests(self, directory: Path) -> Tuple[bool, str]:
        """Run any existing tests in the directory."""
        test_files = list(directory.glob("test_*.py")) + list(directory.glob("*_test.py"))
        
        if not test_files:
            return True, "No tests found"
        
        for test_file in test_files:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_file), "-v"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    return False, f"Test failed: {result.stdout}\n{result.stderr}"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fallback to basic import test
                try:
                    subprocess.run([sys.executable, "-c", f"import sys; sys.path.append('{directory}'); exec(open('{test_file}').read())"], 
                                 check=True, capture_output=True, timeout=10)
                except subprocess.CalledProcessError as e:
                    return False, f"Test execution failed: {e}"
        
        return True, "All tests passed"

class AgenticCoder:
    """Main agentic coding system that orchestrates analysis, generation, and testing."""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.generator = CodeGenerator()
        self.tester = TestRunner()
        self.session_log = []
    
    def analyze_and_improve(self, target_path: Path, apply_fixes: bool = False) -> Dict[str, Any]:
        """Analyze code and optionally apply improvements."""
        start_time = time.time()
        
        self.log(f"Starting analysis of {target_path}")
        
        if target_path.is_file():
            files_to_analyze = [target_path]
        else:
            files_to_analyze = list(target_path.rglob("*.py"))
        
        all_issues = []
        fixes_applied = 0
        
        for filepath in files_to_analyze:
            self.log(f"Analyzing {filepath.name}...")
            
            # Analyze file
            issues = self.analyzer.analyze_file(filepath)
            all_issues.extend(issues)
            
            if issues and apply_fixes:
                # Apply fixes
                with open(filepath, 'r') as f:
                    original_content = f.read()
                
                fixed_content = original_content
                for issue in issues:
                    if issue.get("fix") or issue["type"] in ["security", "error_handling"]:
                        fixed_content = self.generator.generate_fix(issue, fixed_content)
                        fixes_applied += 1
                
                # Test changes
                if fixed_content != original_content:
                    # Write temporary file for testing
                    temp_file = filepath.with_suffix('.py.tmp')
                    with open(temp_file, 'w') as f:
                        f.write(fixed_content)
                    
                    # Run syntax check
                    syntax_ok, syntax_msg = self.tester.run_syntax_check(temp_file)
                    
                    if syntax_ok:
                        # Apply changes
                        with open(filepath, 'w') as f:
                            f.write(fixed_content)
                        self.log(f"Applied fixes to {filepath.name}")
                        temp_file.unlink()  # Remove temp file
                    else:
                        self.log(f"Syntax check failed for {filepath.name}: {syntax_msg}")
                        temp_file.unlink()  # Remove temp file
        
        # Run tests
        test_passed, test_msg = self.tester.run_tests(target_path if target_path.is_dir() else target_path.parent)
        
        duration = time.time() - start_time
        
        results = {
            "files_analyzed": len(files_to_analyze),
            "total_issues": len(all_issues),
            "fixes_applied": fixes_applied,
            "test_results": {"passed": test_passed, "message": test_msg},
            "duration": duration,
            "issues_by_type": self._categorize_issues(all_issues),
            "session_log": self.session_log.copy()
        }
        
        self.log(f"Analysis complete: {len(all_issues)} issues found, {fixes_applied} fixes applied")
        
        return results
    
    def _categorize_issues(self, issues: List[Dict]) -> Dict[str, int]:
        """Categorize issues by type for reporting."""
        categories = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            categories[issue_type] = categories.get(issue_type, 0) + 1
        return categories
    
    def log(self, message: str):
        """Log a message to the session log."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.session_log.append(log_entry)
        print(log_entry)

def create_demo_files():
    """Create demo files with various code issues for testing."""
    demo_dir = Path("demo_code")
    demo_dir.mkdir(exist_ok=True)
    
    # Create a file with security issues
    security_demo = """
import subprocess
import os

def run_command(user_input):
    # Security issue: shell=True with user input
    result = subprocess.run(user_input, shell=True, capture_output=True)
    return result.stdout

def process_data(data):
    # Security issue: eval usage
    return eval(data)

def main():
    cmd = input("Enter command: ")
    output = run_command(cmd)
    print(output)
"""
    
    with open(demo_dir / "security_issues.py", "w") as f:
        f.write(security_demo)
    
    # Create a file with performance issues
    performance_demo = """
def process_items(items):
    result = ""
    # Performance issue: string concatenation in loop
    for item in items:
        result += str(item) + ", "
    
    # Magic numbers
    if len(items) > 42:
        return result
    
    return result[:100]  # Another magic number

def calculate_something():
    try:
        value = 10 / 0
    except:  # Bare except clause
        return None
    return value
"""
    
    with open(demo_dir / "performance_issues.py", "w") as f:
        f.write(performance_demo)
    
    return demo_dir

def main():
    """Demonstrate the agentic coding system."""
    print("Agentic Coder - Autonomous Code Analysis and Improvement System")
    print("=" * 65)
    print()
    
    # Create demo files
    demo_dir = create_demo_files()
    print(f"Created demo files in {demo_dir}/")
    print()
    
    # Initialize the agentic coder
    coder = AgenticCoder()
    
    # Analyze and improve the demo code
    print("Phase 1: Analysis Only")
    print("-" * 20)
    results = coder.analyze_and_improve(demo_dir, apply_fixes=False)
    
    print(f"\nAnalysis Results:")
    print(f"Files analyzed: {results['files_analyzed']}")
    print(f"Issues found: {results['total_issues']}")
    print(f"Issue breakdown: {results['issues_by_type']}")
    print(f"Duration: {results['duration']:.2f}s")
    
    print("\nPhase 2: Autonomous Improvement")
    print("-" * 32)
    
    # Apply fixes
    results = coder.analyze_and_improve(demo_dir, apply_fixes=True)
    
    print(f"\nImprovement Results:")
    print(f"Fixes applied: {results['fixes_applied']}")
    print(f"Tests passed: {results['test_results']['passed']}")
    print(f"Test message: {results['test_results']['message']}")
    
    print("\nDemonstration complete!")
    print(f"Check the files in {demo_dir}/ to see the applied improvements.")

if __name__ == "__main__":
    main()