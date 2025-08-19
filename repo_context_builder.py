#!/usr/bin/env python3
"""
enhanced_repo_context_builder.py

Enhanced repository context builder that creates comprehensive documentation
of a Python codebase with deep cross-referencing and dependency analysis.
"""

import ast
import os
import re
import yaml
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict, Counter
import argparse

# ---------- CONFIGURATION ----------
REPO_ROOT = Path.cwd()
OUTPUT_FILE = REPO_ROOT / "REPOSITORY_CONTEXT.md"
CACHE_DIR = REPO_ROOT / ".cache"
DS_MODEL = "deepseek-chat"
DS_BASE = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise SystemExit("Please set DEEPSEEK_API_KEY environment variable.")

CACHE_DIR.mkdir(exist_ok=True)

# Default ignore patterns
DEFAULT_IGNORE_PATTERNS = [
    "*.pyc", "__pycache__", ".git", ".cache", "venv", ".venv", 
    "env", ".env", "node_modules", "build", "dist", "*.egg-info",
    "test_*", "*_test.py", "tests/", "migrations/"
]

# ---------- UTILITIES ----------

def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def cached_request(messages: List[Dict[str, str]]) -> str:
    payload = json.dumps(messages, sort_keys=True)
    key = cache_key(payload)
    cache_file = CACHE_DIR / f"{key}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    import httpx
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DS_MODEL,
        "messages": messages,
        "temperature": 0.0,
    }
    resp = httpx.post(DS_BASE, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    cache_file.write_text(content, encoding="utf-8")
    return content

def should_ignore_file(file_path: Path, ignore_patterns: List[str]) -> bool:
    """Check if a file should be ignored based on patterns."""
    path_str = str(file_path)
    for pattern in ignore_patterns:
        if "*" in pattern:
            import fnmatch
            if fnmatch.fnmatch(path_str, pattern):
                return True
        else:
            if pattern in path_str:
                return True
    return False

def get_file_importance_score(file_path: Path, code: str) -> int:
    """Calculate importance score for a file based on various factors."""
    score = 0
    
    # Size factor (larger files are often more important)
    lines = len(code.splitlines())
    if lines > 100:
        score += 3
    elif lines > 50:
        score += 2
    elif lines > 20:
        score += 1
    
    # Special file names
    name = file_path.name.lower()
    if name in ['__init__.py', 'main.py', 'app.py', 'server.py', 'client.py']:
        score += 3
    elif name.startswith('__') and name.endswith('__.py'):
        score += 2
    
    # Code complexity indicators
    if 'class ' in code:
        score += len(re.findall(r'\nclass ', code))
    if 'def ' in code:
        score += min(len(re.findall(r'\ndef ', code)) // 2, 3)
    if 'import ' in code or 'from ' in code:
        score += min(len(re.findall(r'\n(import|from) ', code)) // 3, 2)
    
    return score

# ---------- ENHANCED AST ANALYZER ----------

class EnhancedAnalyzer(ast.NodeVisitor):
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.classes: List[Dict[str, Any]] = []
        self.functions: List[Dict[str, Any]] = []
        self.constants: List[Dict[str, Any]] = []
        self.vars: List[Dict[str, Any]] = []
        self.imports: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []
        self.decorators: List[str] = []
        self.inheritance: Dict[str, List[str]] = {}
        self.complexity_score = 0
        
    def visit_ClassDef(self, node: ast.ClassDef):
        doc = ast.get_docstring(node) or ""
        bases = [self._get_name(base) for base in node.bases]
        methods = []
        
        # Extract methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_doc = ast.get_docstring(item) or ""
                methods.append({
                    "name": item.name,
                    "doc": method_doc[:100] + "..." if len(method_doc) > 100 else method_doc,
                    "is_async": isinstance(item, ast.AsyncFunctionDef),
                    "is_property": any(self._get_name(d) == "property" for d in item.decorator_list),
                    "is_static": any(self._get_name(d) == "staticmethod" for d in item.decorator_list),
                    "is_class": any(self._get_name(d) == "classmethod" for d in item.decorator_list)
                })
        
        self.classes.append({
            "name": node.name,
            "doc": doc[:200] + "..." if len(doc) > 200 else doc,
            "bases": bases,
            "methods": methods,
            "decorators": [self._get_name(d) for d in node.decorator_list],
            "line": node.lineno
        })
        
        if bases:
            self.inheritance[node.name] = bases
            
        self.complexity_score += len(methods) + 2
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Only top-level functions (not methods)
        if not isinstance(self._get_parent(node), ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            args = [arg.arg for arg in node.args.args]
            
            self.functions.append({
                "name": node.name,
                "doc": doc[:150] + "..." if len(doc) > 150 else doc,
                "args": args,
                "decorators": [self._get_name(d) for d in node.decorator_list],
                "is_async": False,
                "line": node.lineno,
                "returns": self._get_name(node.returns) if node.returns else None
            })
            
            self.complexity_score += len(node.body) // 3
            
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if not isinstance(self._get_parent(node), ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            args = [arg.arg for arg in node.args.args]
            
            self.functions.append({
                "name": node.name,
                "doc": doc[:150] + "..." if len(doc) > 150 else doc,
                "args": args,
                "decorators": [self._get_name(d) for d in node.decorator_list],
                "is_async": True,
                "line": node.lineno,
                "returns": self._get_name(node.returns) if node.returns else None
            })
            
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append({
                "type": "import",
                "module": alias.name,
                "alias": alias.asname,
                "from": None,
                "line": node.lineno
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.imports.append({
                "type": "from",
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "from": module,
                "line": node.lineno
            })
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                try:
                    value_preview = ast.unparse(node.value)[:60] if hasattr(ast, "unparse") else str(node.value)[:60]
                except:
                    value_preview = "<complex expression>"
                
                if re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                    self.constants.append({
                        "name": name,
                        "value_preview": value_preview,
                        "line": node.lineno,
                        "type": self._infer_type(node.value)
                    })
                else:
                    self.vars.append({
                        "name": name,
                        "value_preview": value_preview,
                        "line": node.lineno,
                        "type": self._infer_type(node.value)
                    })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = self._get_name(node.func)
        if func_name:
            self.calls.append({
                "name": func_name,
                "line": node.lineno,
                "args_count": len(node.args),
                "has_kwargs": len(node.keywords) > 0
            })
        self.generic_visit(node)

    def _get_name(self, node) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return None
    
    def _get_parent(self, node):
        # This is a simplified parent detection
        return None
    
    def _infer_type(self, node) -> str:
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Call):
            func_name = self._get_name(node.func)
            return func_name or "unknown"
        return "unknown"

# ---------- CROSS-REFERENCE ANALYZER ----------

class CrossReferenceAnalyzer:
    def __init__(self, file_data: Dict[str, Any]):
        self.file_data = file_data
        self.symbol_map = self._build_symbol_map()
        self.dependencies = self._analyze_dependencies()
        
    def _build_symbol_map(self) -> Dict[str, List[str]]:
        """Build a map of symbols to files where they're defined."""
        symbol_map = defaultdict(list)
        
        for file_path, data in self.file_data.items():
            for cls in data.get("classes", []):
                symbol_map[cls["name"]].append(file_path)
            for func in data.get("functions", []):
                symbol_map[func["name"]].append(file_path)
            for const in data.get("constants", []):
                symbol_map[const["name"]].append(file_path)
                
        return dict(symbol_map)
    
    def _analyze_dependencies(self) -> Dict[str, Dict[str, int]]:
        """Analyze cross-file dependencies."""
        dependencies = defaultdict(lambda: defaultdict(int))
        
        for file_path, data in self.file_data.items():
            for call in data.get("calls", []):
                call_name = call["name"]
                # Check if this call references a symbol from another file
                for symbol, files in self.symbol_map.items():
                    if symbol in call_name and len(files) > 1:
                        for target_file in files:
                            if target_file != file_path:
                                dependencies[file_path][target_file] += 1
                                
        return dict(dependencies)
    
    def get_file_dependencies(self, file_path: str) -> List[Tuple[str, int]]:
        """Get dependencies for a specific file."""
        deps = self.dependencies.get(file_path, {})
        return sorted(deps.items(), key=lambda x: x[1], reverse=True)
    
    def get_most_referenced_symbols(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get the most referenced symbols across the codebase."""
        symbol_refs = Counter()
        
        for file_data in self.file_data.values():
            for call in file_data.get("calls", []):
                symbol_refs[call["name"]] += 1
                
        return symbol_refs.most_common(limit)

# ---------- LLM INTERACTION ----------

def analyze_file_purpose(file_path: str, code_structure: Dict[str, Any]) -> str:
    """Get detailed analysis of file's purpose and role."""
    prompt = f"""Analyze this Python file structure and provide a comprehensive analysis:

File: {file_path}

Structure:
{yaml.dump(code_structure, default_flow_style=False)}

Provide a detailed analysis in the following format:

**Primary Purpose:** (1-2 sentences about what this file accomplishes)

**Key Responsibilities:** (bullet points of main functions/responsibilities)

**Architecture Role:** (how this file fits into the larger system - is it a core module, utility, interface, etc.)

**Notable Patterns:** (any important design patterns, architectural decisions, or code organization patterns)

Keep the analysis technical but concise. Focus on the file's role in the codebase."""

    messages = [{"role": "user", "content": prompt}]
    return cached_request(messages)

def analyze_module_interactions(file_path: str, dependencies: List[Tuple[str, int]], 
                              imported_modules: List[str]) -> str:
    """Analyze how this module interacts with others."""
    prompt = f"""Analyze the interactions and dependencies for this Python module:

File: {file_path}

Dependencies on other files:
{yaml.dump(dependencies)}

Imported modules:
{yaml.dump(imported_modules)}

Provide analysis in this format:

**Internal Dependencies:** (how this file depends on other files in the codebase)

**External Dependencies:** (key third-party libraries and their roles)

**Integration Patterns:** (how this module typically interacts with others)

Be concise but insightful about the module's integration patterns."""

    messages = [{"role": "user", "content": prompt}]
    return cached_request(messages)

def generate_architecture_overview(cross_ref: CrossReferenceAnalyzer, 
                                 file_summaries: Dict[str, str]) -> str:
    """Generate high-level architecture overview."""
    most_referenced = cross_ref.get_most_referenced_symbols(15)
    
    prompt = f"""Based on this repository analysis, provide a comprehensive architecture overview:

Most Referenced Symbols:
{yaml.dump(most_referenced)}

File Summaries:
{yaml.dump({k: v[:200] for k, v in file_summaries.items()})}

Provide a structured analysis:

## Repository Architecture Overview

**Core Purpose:** (What does this repository accomplish?)

**Architectural Style:** (What patterns/styles are used? MVC, microservices, monolithic, etc.)

**Key Components:** (Main modules and their roles)

**Data Flow:** (How information flows through the system)

**Integration Points:** (How different parts connect)

**Design Patterns:** (Notable patterns used throughout)

Write in markdown format, be comprehensive but concise (~400-500 words)."""

    messages = [{"role": "user", "content": prompt}]
    return cached_request(messages)

# ---------- MAIN PIPELINE ----------

def select_files(repo_root: Path, include_patterns: List[str] = None, 
                exclude_patterns: List[str] = None, max_files: int = None) -> List[Path]:
    """Select files to analyze based on patterns and importance."""
    all_files = list(repo_root.rglob("*.py"))
    
    # Apply exclude patterns
    exclude_patterns = exclude_patterns or DEFAULT_IGNORE_PATTERNS
    filtered_files = [f for f in all_files if not should_ignore_file(f, exclude_patterns)]
    
    # Apply include patterns if specified
    if include_patterns:
        include_files = []
        for pattern in include_patterns:
            import fnmatch
            # Convert the pattern to work with relative paths from repo_root
            for f in filtered_files:
                rel_path = str(f.relative_to(repo_root))
                # Try both the relative path and the full path
                if (fnmatch.fnmatch(rel_path, pattern) or 
                    fnmatch.fnmatch(str(f), pattern) or
                    fnmatch.fnmatch(f.name, pattern)):
                    include_files.append(f)
        filtered_files = list(set(include_files))  # Remove duplicates
    
    # Score and sort by importance
    file_scores = []
    for file_path in filtered_files:
        try:
            code = file_path.read_text(encoding="utf-8")
            score = get_file_importance_score(file_path, code)
            file_scores.append((file_path, score, len(code)))
        except:
            continue
    
    file_scores.sort(key=lambda x: x[1], reverse=True)
    
    if max_files:
        file_scores = file_scores[:max_files]
    
    return [f[0] for f in file_scores]

def main():
    parser = argparse.ArgumentParser(description="Enhanced Repository Context Builder")
    parser.add_argument("--include", nargs="*", help="Include patterns (e.g., 'src/*.py')")
    parser.add_argument("--exclude", nargs="*", help="Additional exclude patterns")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to analyze")
    parser.add_argument("--output", help="Output file path", default=str(OUTPUT_FILE))
    
    args = parser.parse_args()
    
    # Select files to analyze
    selected_files = select_files(
        REPO_ROOT, 
        args.include, 
        args.exclude, 
        args.max_files
    )
    
    print(f"Selected {len(selected_files)} files for analysis...")
    
    # Analyze each file
    file_data = {}
    file_analyses = {}
    
    for pyfile in selected_files:
        print(f"Analyzing {pyfile.relative_to(REPO_ROOT)}...")
        
        try:
            code = pyfile.read_text(encoding="utf-8")
            analyzer = EnhancedAnalyzer(pyfile)
            analyzer.visit(ast.parse(code))
            
            rel_path = str(pyfile.relative_to(REPO_ROOT))
            
            # Store structured data
            file_data[rel_path] = {
                "classes": analyzer.classes,
                "functions": analyzer.functions,
                "constants": analyzer.constants,
                "vars": analyzer.vars,
                "imports": analyzer.imports,
                "calls": analyzer.calls,
                "inheritance": analyzer.inheritance,
                "complexity_score": analyzer.complexity_score,
                "file_size": len(code),
                "line_count": len(code.splitlines())
            }
            
            # Get detailed analysis from LLM
            file_analyses[rel_path] = analyze_file_purpose(rel_path, file_data[rel_path])
            
        except Exception as e:
            print(f"Error analyzing {pyfile}: {e}")
            continue
    
    # Cross-reference analysis
    cross_ref = CrossReferenceAnalyzer(file_data)
    
    # Analyze module interactions
    interaction_analyses = {}
    for file_path, data in file_data.items():
        dependencies = cross_ref.get_file_dependencies(file_path)
        imported_modules = [imp["module"] for imp in data["imports"]]
        
        if dependencies or imported_modules:
            interaction_analyses[file_path] = analyze_module_interactions(
                file_path, dependencies, imported_modules
            )
    
    # Generate architecture overview
    architecture_overview = generate_architecture_overview(cross_ref, file_analyses)
    
    # Generate final markdown
    md_content = generate_markdown_report(
        file_data, file_analyses, interaction_analyses, 
        cross_ref, architecture_overview
    )
    
    output_path = Path(args.output)
    output_path.write_text(md_content, encoding="utf-8")
    print(f"Enhanced context written to {output_path}")

def generate_markdown_report(file_data, file_analyses, interaction_analyses, 
                           cross_ref, architecture_overview):
    """Generate the final markdown report."""
    md = f"# Repository Context\n\n"
    md += f"Generated on: {os.popen('date').read().strip()}\n"
    md += f"Files analyzed: {len(file_data)}\n\n"
    
    # Architecture overview
    md += architecture_overview + "\n\n"
    
    # Most referenced symbols
    most_ref = cross_ref.get_most_referenced_symbols(10)
    if most_ref:
        md += "## Most Referenced Symbols\n\n"
        for symbol, count in most_ref:
            md += f"- `{symbol}` ({count} references)\n"
        md += "\n"
    
    # File-by-file analysis
    md += "## Detailed File Analysis\n\n"
    
    for file_path in sorted(file_data.keys()):
        data = file_data[file_path]
        analysis = file_analyses.get(file_path, "")
        
        md += f"### `{file_path}`\n\n"
        md += f"**Complexity Score:** {data['complexity_score']} | "
        md += f"**Lines:** {data['line_count']} | "
        md += f"**Size:** {data['file_size']} bytes\n\n"
        
        # LLM analysis
        if analysis:
            md += analysis + "\n\n"
        
        # Module interactions
        if file_path in interaction_analyses:
            md += "#### Module Interactions\n\n"
            md += interaction_analyses[file_path] + "\n\n"
        
        # Detailed structure
        if data["classes"]:
            md += "#### Classes\n\n"
            for cls in data["classes"]:
                md += f"- **`{cls['name']}`** (line {cls['line']})"
                if cls["bases"]:
                    md += f" extends {', '.join(cls['bases'])}"
                md += f"\n  - {cls['doc']}\n"
                if cls["methods"]:
                    methods_summary = [m["name"] for m in cls["methods"][:5]]
                    if len(cls["methods"]) > 5:
                        methods_summary.append(f"...+{len(cls['methods'])-5} more")
                    md += f"  - Methods: {', '.join(methods_summary)}\n"
                md += "\n"
        
        if data["functions"]:
            md += "#### Functions\n\n"
            for func in data["functions"]:
                md += f"- **`{func['name']}`** (line {func['line']})"
                if func["args"]:
                    md += f" - Args: {', '.join(func['args'])}"
                if func["is_async"]:
                    md += " [async]"
                md += f"\n  - {func['doc']}\n"
        
        # Dependencies
        deps = cross_ref.get_file_dependencies(file_path)
        if deps:
            md += "#### Internal Dependencies\n\n"
            for dep_file, count in deps[:5]:
                md += f"- `{dep_file}` ({count} references)\n"
            md += "\n"
        
        md += "---\n\n"
    
    return md

if __name__ == "__main__":
    main()