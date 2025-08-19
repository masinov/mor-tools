#!/usr/bin/env python3
"""
comprehensive_repo_analyzer.py

Creates a comprehensive functional documentation of a Python repository
that captures all semantic details, cross-references, and usage patterns
needed to understand and use the codebase without seeing the actual code.
"""

import ast
import os
import re
import yaml
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple, Union
from collections import defaultdict, Counter
import argparse

# ---------- CONFIGURATION ----------
REPO_ROOT = Path.cwd()
OUTPUT_FILE = REPO_ROOT / "COMPREHENSIVE_CONTEXT.md"
CACHE_DIR = REPO_ROOT / ".cache"
DS_MODEL = "deepseek-chat"
DS_BASE = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise SystemExit("Please set DEEPSEEK_API_KEY environment variable.")

CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_IGNORE_PATTERNS = [
    "*.pyc", "__pycache__", ".git", ".cache", "venv", ".venv", 
    "env", ".env", "node_modules", "build", "dist", "*.egg-info"
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

# ---------- COMPREHENSIVE AST ANALYZER ----------

class ComprehensiveAnalyzer(ast.NodeVisitor):
    def __init__(self, file_path: Path, source_code: str):
        self.file_path = file_path
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        
        # Core structures
        self.classes: List[Dict[str, Any]] = []
        self.functions: List[Dict[str, Any]] = []
        self.constants: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.imports: List[Dict[str, Any]] = []
        
        # Cross-reference data
        self.function_calls: List[Dict[str, Any]] = []
        self.attribute_accesses: List[Dict[str, Any]] = []
        self.method_calls: List[Dict[str, Any]] = []
        self.class_instantiations: List[Dict[str, Any]] = []
        
        # Context tracking
        self.current_class = None
        self.current_function = None
        self.scope_stack = []
        
        # Symbol definitions
        self.defined_symbols = set()
        self.used_symbols = set()
        
    def get_source_segment(self, node: ast.AST) -> str:
        """Extract source code for a node."""
        try:
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                start_line = node.lineno - 1
                end_line = node.end_lineno
                if end_line and start_line < len(self.source_lines):
                    return '\n'.join(self.source_lines[start_line:end_line])
        except:
            pass
        return ""
    
    def extract_type_hints(self, node) -> Optional[str]:
        """Extract type hint information."""
        if hasattr(node, 'annotation') and node.annotation:
            return ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation)
        return None
    
    def extract_default_value(self, default) -> str:
        """Extract default value from function argument."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(default)
            else:
                return str(default)
        except:
            return "..."
    
    def visit_ClassDef(self, node: ast.ClassDef):
        old_class = self.current_class
        self.current_class = node.name
        self.scope_stack.append(f"class:{node.name}")
        
        # Extract class information
        docstring = ast.get_docstring(node)
        base_classes = []
        for base in node.bases:
            base_name = self.get_name_from_node(base)
            if base_name:
                base_classes.append(base_name)
                self.used_symbols.add(base_name)
        
        # Extract methods
        methods = []
        class_variables = []
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self.extract_function_info(item, is_method=True)
                methods.append(method_info)
            elif isinstance(item, ast.Assign):
                # Class variables
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        var_info = {
                            "name": target.id,
                            "type": self.extract_type_hints(item) or self.infer_type_from_value(item.value),
                            "value_preview": self.get_value_preview(item.value),
                            "line": item.lineno
                        }
                        class_variables.append(var_info)
        
        class_info = {
            "name": node.name,
            "docstring": docstring,
            "base_classes": base_classes,
            "methods": methods,
            "class_variables": class_variables,
            "decorators": [self.get_name_from_node(d) for d in node.decorator_list],
            "line": node.lineno,
            "source_preview": self.get_source_segment(node)[:200] + "..."
        }
        
        self.classes.append(class_info)
        self.defined_symbols.add(node.name)
        
        # Visit children
        self.generic_visit(node)
        
        self.current_class = old_class
        self.scope_stack.pop()
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.visit_function(node, is_async=False)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_function(node, is_async=True)
    
    def visit_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool):
        old_function = self.current_function
        self.current_function = node.name
        self.scope_stack.append(f"function:{node.name}")
        
        # Only add to functions list if it's a top-level function (not a method)
        if not self.current_class:
            func_info = self.extract_function_info(node, is_method=False)
            self.functions.append(func_info)
            self.defined_symbols.add(node.name)
        
        # Visit children to find function calls, etc.
        self.generic_visit(node)
        
        self.current_function = old_function
        self.scope_stack.pop()
    
    def extract_function_info(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_method: bool) -> Dict[str, Any]:
        """Extract comprehensive function/method information."""
        docstring = ast.get_docstring(node)
        
        # Extract parameters
        parameters = []
        args = node.args
        
        # Regular arguments
        for i, arg in enumerate(args.args):
            param_info = {
                "name": arg.arg,
                "type": self.extract_type_hints(arg),
                "default": None,
                "kind": "positional"
            }
            
            # Check for default values
            defaults_offset = len(args.args) - len(args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                param_info["default"] = self.extract_default_value(args.defaults[default_idx])
            
            parameters.append(param_info)
        
        # *args
        if args.vararg:
            parameters.append({
                "name": args.vararg.arg,
                "type": self.extract_type_hints(args.vararg),
                "kind": "vararg"
            })
        
        # **kwargs
        if args.kwarg:
            parameters.append({
                "name": args.kwarg.arg,
                "type": self.extract_type_hints(args.kwarg),
                "kind": "kwarg"
            })
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            dec_name = self.get_name_from_node(decorator)
            if dec_name:
                decorators.append(dec_name)
                self.used_symbols.add(dec_name)
        
        # Extract calls made within this function
        function_calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_info = self.extract_call_info(child)
                if call_info:
                    function_calls.append(call_info)
        
        return {
            "name": node.name,
            "docstring": docstring,
            "parameters": parameters,
            "return_type": return_type,
            "decorators": decorators,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "is_method": is_method,
            "is_static": "staticmethod" in decorators,
            "is_class_method": "classmethod" in decorators,
            "is_property": "property" in decorators,
            "line": node.lineno,
            "calls_made": function_calls,
            "source_preview": self.get_source_segment(node)[:300] + "..."
        }
    
    def visit_Call(self, node: ast.Call):
        call_info = self.extract_call_info(node)
        if call_info:
            # Categorize the call
            if call_info["is_method_call"]:
                self.method_calls.append(call_info)
            elif call_info["is_class_instantiation"]:
                self.class_instantiations.append(call_info)
            else:
                self.function_calls.append(call_info)
        
        self.generic_visit(node)
    
    def extract_call_info(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        """Extract detailed information about a function/method call."""
        func_name = self.get_name_from_node(node.func)
        if not func_name:
            return None
        
        # Determine call type
        is_method_call = isinstance(node.func, ast.Attribute)
        is_class_instantiation = False
        
        # Check if it's a class instantiation (starts with uppercase)
        if func_name and func_name[0].isupper():
            is_class_instantiation = True
        
        # Extract arguments
        arguments = []
        
        # Positional arguments
        for arg in node.args:
            arg_info = {
                "type": "positional",
                "value": self.get_value_preview(arg),
                "inferred_type": self.infer_type_from_value(arg)
            }
            arguments.append(arg_info)
        
        # Keyword arguments
        for keyword in node.keywords:
            arg_info = {
                "type": "keyword",
                "name": keyword.arg,
                "value": self.get_value_preview(keyword.value),
                "inferred_type": self.infer_type_from_value(keyword.value)
            }
            arguments.append(arg_info)
        
        call_info = {
            "function_name": func_name,
            "is_method_call": is_method_call,
            "is_class_instantiation": is_class_instantiation,
            "arguments": arguments,
            "line": node.lineno,
            "context": {
                "current_class": self.current_class,
                "current_function": self.current_function,
                "scope": "::".join(self.scope_stack)
            }
        }
        
        # Add object for method calls
        if is_method_call and isinstance(node.func, ast.Attribute):
            object_name = self.get_name_from_node(node.func.value)
            call_info["object"] = object_name
            call_info["method"] = node.func.attr
            if object_name:
                self.used_symbols.add(object_name)
        
        self.used_symbols.add(func_name)
        return call_info
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            import_info = {
                "type": "import",
                "module": alias.name,
                "alias": alias.asname,
                "line": node.lineno
            }
            self.imports.append(import_info)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            import_info = {
                "type": "from_import",
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "line": node.lineno
            }
            self.imports.append(import_info)
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        # Only process top-level assignments (not in classes/functions)
        if not self.current_class and not self.current_function:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    
                    var_info = {
                        "name": name,
                        "type": self.extract_type_hints(node) or self.infer_type_from_value(node.value),
                        "value_preview": self.get_value_preview(node.value),
                        "line": node.lineno,
                        "is_constant": name.isupper()
                    }
                    
                    if name.isupper():
                        self.constants[name] = var_info
                    else:
                        self.variables[name] = var_info
                    
                    self.defined_symbols.add(name)
        
        self.generic_visit(node)
    
    def get_name_from_node(self, node) -> Optional[str]:
        """Extract name from various node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self.get_name_from_node(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        elif isinstance(node, ast.Call):
            return self.get_name_from_node(node.func)
        return None
    
    def get_value_preview(self, node) -> str:
        """Get a preview of the value from an AST node."""
        try:
            if hasattr(ast, 'unparse'):
                preview = ast.unparse(node)
                return preview[:100] + "..." if len(preview) > 100 else preview
            else:
                return str(node)[:100]
        except:
            return "<complex expression>"
    
    def infer_type_from_value(self, node) -> str:
        """Infer type from AST node."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "List"
        elif isinstance(node, ast.Dict):
            return "Dict"
        elif isinstance(node, ast.Set):
            return "Set"
        elif isinstance(node, ast.Tuple):
            return "Tuple"
        elif isinstance(node, ast.Call):
            func_name = self.get_name_from_node(node.func)
            return func_name or "unknown"
        elif isinstance(node, ast.Name):
            return "reference"
        return "unknown"

# ---------- CROSS-REFERENCE ANALYZER ----------

class CrossReferenceAnalyzer:
    def __init__(self, file_analyses: Dict[str, ComprehensiveAnalyzer]):
        self.file_analyses = file_analyses
        self.global_symbols = self._build_global_symbol_map()
        self.dependencies = self._analyze_dependencies()
        self.call_graph = self._build_call_graph()
        self.missing_references = self._find_missing_references()
    
    def _build_global_symbol_map(self) -> Dict[str, List[str]]:
        """Build a map of all symbols and where they're defined."""
        symbol_map = defaultdict(list)
        
        for file_path, analyzer in self.file_analyses.items():
            # Add all defined symbols
            for symbol in analyzer.defined_symbols:
                symbol_map[symbol].append(file_path)
        
        return dict(symbol_map)
    
    def _analyze_dependencies(self) -> Dict[str, Dict[str, List[str]]]:
        """Analyze what each file depends on from other files."""
        dependencies = defaultdict(lambda: defaultdict(list))
        
        for file_path, analyzer in self.file_analyses.items():
            for symbol in analyzer.used_symbols:
                # Find where this symbol is defined
                if symbol in self.global_symbols:
                    for def_file in self.global_symbols[symbol]:
                        if def_file != file_path:
                            dependencies[file_path][def_file].append(symbol)
        
        return dict(dependencies)
    
    def _build_call_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build a comprehensive call graph."""
        call_graph = defaultdict(list)
        
        for file_path, analyzer in self.file_analyses.items():
            # Process all calls
            all_calls = (analyzer.function_calls + 
                        analyzer.method_calls + 
                        analyzer.class_instantiations)
            
            for call in all_calls:
                call_info = {
                    "file": file_path,
                    "target": call["function_name"],
                    "type": "method" if call["is_method_call"] else "function",
                    "context": call["context"],
                    "line": call["line"],
                    "arguments": len(call["arguments"])
                }
                call_graph[call["function_name"]].append(call_info)
        
        return dict(call_graph)
    
    def _find_missing_references(self) -> List[Dict[str, Any]]:
        """Find potentially missing imports or undefined references."""
        missing = []
        
        for file_path, analyzer in self.file_analyses.items():
            for symbol in analyzer.used_symbols:
                # Check if symbol is defined anywhere
                if (symbol not in self.global_symbols and 
                    not self._is_builtin_or_imported(symbol, analyzer)):
                    missing.append({
                        "symbol": symbol,
                        "file": file_path,
                        "type": "undefined_reference"
                    })
        
        return missing
    
    def _is_builtin_or_imported(self, symbol: str, analyzer: ComprehensiveAnalyzer) -> bool:
        """Check if symbol is a builtin or imported."""
        # Check imports
        for imp in analyzer.imports:
            if imp["type"] == "import" and (symbol == imp["module"] or symbol == imp["alias"]):
                return True
            elif imp["type"] == "from_import" and (symbol == imp["name"] or symbol == imp["alias"]):
                return True
        
        # Check builtins
        import builtins
        return hasattr(builtins, symbol.split('.')[0])

# ---------- LLM ANALYSIS ----------

def analyze_functional_relationships(file_path: str, analyzer: ComprehensiveAnalyzer, 
                                   cross_ref: CrossReferenceAnalyzer) -> str:
    """Get LLM analysis of functional relationships and usage patterns."""
    
    # Prepare data for LLM
    analysis_data = {
        "file": file_path,
        "classes": analyzer.classes,
        "functions": analyzer.functions,
        "imports": analyzer.imports,
        "dependencies": cross_ref.dependencies.get(file_path, {}),
        "function_calls": analyzer.function_calls[:10],  # Limit for context
        "method_calls": analyzer.method_calls[:10],
    }
    
    prompt = f"""Analyze this Python module's functional relationships and provide comprehensive usage documentation:

{yaml.dump(analysis_data, default_flow_style=False)}

Provide analysis in this specific format:

## Module Purpose
Brief description of what this module accomplishes.

## Public API
For each public class/function, provide:
- **Usage pattern**: How to import and use it
- **Method signatures**: Exact calling conventions
- **Dependencies**: What other modules/classes it needs
- **Return types**: What it returns and how to use the results

## Cross-References
- **Imports from other modules**: What it uses and how
- **Exports to other modules**: What other modules use from this
- **Potential issues**: Missing imports, circular dependencies, etc.

## Integration Examples
Concrete examples of how this module integrates with others.

Focus on functional semantics - how to actually use the code, not implementation details."""

    messages = [{"role": "user", "content": prompt}]
    return cached_request(messages)

def generate_integration_guide(cross_ref: CrossReferenceAnalyzer) -> str:
    """Generate comprehensive integration guide."""
    
    prompt = f"""Based on this cross-reference analysis, create a comprehensive integration guide:

Global Symbols: {yaml.dump(dict(list(cross_ref.global_symbols.items())[:20]))}

Dependencies: {yaml.dump(dict(list(cross_ref.dependencies.items())[:10]))}

Call Graph: {yaml.dump(dict(list(cross_ref.call_graph.items())[:15]))}

Missing References: {yaml.dump(cross_ref.missing_references)}

Create a structured integration guide:

## Repository Architecture
High-level overview of how modules work together.

## Module Dependencies
Clear dependency graph and import patterns.

## API Usage Patterns
Common patterns for using the main APIs.

## Integration Points
Key interfaces between modules.

## Potential Issues
- Missing imports that could cause runtime errors
- Circular dependency risks
- Interface mismatches

## Quick Start Guide
Step-by-step guide for using the main functionality.

Focus on practical integration knowledge that would prevent bugs and enable correct usage."""

    messages = [{"role": "user", "content": prompt}]
    return cached_request(messages)

# ---------- MAIN PIPELINE ----------

def select_files(repo_root: Path, include_patterns: List[str] = None, 
                exclude_patterns: List[str] = None) -> List[Path]:
    """Select files to analyze based on patterns."""
    all_files = list(repo_root.rglob("*.py"))
    
    # Apply exclude patterns
    exclude_patterns = exclude_patterns or DEFAULT_IGNORE_PATTERNS
    filtered_files = [f for f in all_files if not should_ignore_file(f, exclude_patterns)]
    
    # Apply include patterns if specified
    if include_patterns:
        include_files = []
        for pattern in include_patterns:
            import fnmatch
            for f in filtered_files:
                rel_path = str(f.relative_to(repo_root))
                if (fnmatch.fnmatch(rel_path, pattern) or 
                    fnmatch.fnmatch(str(f), pattern) or
                    fnmatch.fnmatch(f.name, pattern)):
                    include_files.append(f)
        filtered_files = list(set(include_files))
    
    return sorted(filtered_files)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Repository Analyzer")
    parser.add_argument("--include", nargs="*", help="Include patterns (e.g., 'model/*.py')")
    parser.add_argument("--exclude", nargs="*", help="Additional exclude patterns")
    parser.add_argument("--output", help="Output file path", default=str(OUTPUT_FILE))
    
    args = parser.parse_args()
    
    selected_files = select_files(REPO_ROOT, args.include, args.exclude)
    print(f"Selected {len(selected_files)} files for analysis...")
    
    # Analyze each file comprehensively
    file_analyses = {}
    
    for pyfile in selected_files:
        print(f"Analyzing {pyfile.relative_to(REPO_ROOT)}...")
        
        try:
            code = pyfile.read_text(encoding="utf-8")
            analyzer = ComprehensiveAnalyzer(pyfile, code)
            analyzer.visit(ast.parse(code))
            
            rel_path = str(pyfile.relative_to(REPO_ROOT))
            file_analyses[rel_path] = analyzer
            
        except Exception as e:
            print(f"Error analyzing {pyfile}: {e}")
            continue
    
    # Cross-reference analysis
    print("Building cross-references...")
    cross_ref = CrossReferenceAnalyzer(file_analyses)
    
    # Generate functional analyses
    print("Generating functional analyses...")
    functional_analyses = {}
    for file_path, analyzer in file_analyses.items():
        try:
            functional_analyses[file_path] = analyze_functional_relationships(
                file_path, analyzer, cross_ref
            )
        except Exception as e:
            print(f"Error analyzing relationships for {file_path}: {e}")
            functional_analyses[file_path] = "Analysis failed"
    
    # Generate integration guide
    print("Generating integration guide...")
    integration_guide = generate_integration_guide(cross_ref)
    
    # Generate comprehensive report
    print("Generating final report...")
    report = generate_comprehensive_report(
        file_analyses, functional_analyses, cross_ref, integration_guide
    )
    
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"Comprehensive analysis written to {output_path}")

def generate_comprehensive_report(file_analyses, functional_analyses, 
                                cross_ref, integration_guide):
    """Generate the comprehensive markdown report."""
    
    md = "# Comprehensive Repository Analysis\n\n"
    md += f"Generated: {os.popen('date').read().strip()}\n"
    md += f"Files analyzed: {len(file_analyses)}\n\n"
    
    # Integration guide
    md += integration_guide + "\n\n"
    
    # Global symbols summary
    md += "## Global Symbol Map\n\n"
    for symbol, files in sorted(cross_ref.global_symbols.items())[:30]:
        md += f"- **`{symbol}`** defined in: {', '.join([f'`{f}`' for f in files])}\n"
    md += "\n"
    
    # Missing references
    if cross_ref.missing_references:
        md += "## Potential Issues\n\n"
        md += "### Missing References\n\n"
        for missing in cross_ref.missing_references:
            md += f"- `{missing['symbol']}` used in `{missing['file']}` but not defined\n"
        md += "\n"
    
    # Call graph highlights
    md += "## Most Called Functions/Methods\n\n"
    call_counts = {name: len(calls) for name, calls in cross_ref.call_graph.items()}
    top_calls = sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    for func_name, count in top_calls:
        md += f"- **`{func_name}`** ({count} calls)\n"
        example_calls = cross_ref.call_graph[func_name][:3]
        for call in example_calls:
            md += f"  - Called from `{call['file']}` line {call['line']} ({call['context']['scope']})\n"
    md += "\n"
    
    # Detailed file analysis
    md += "## Detailed Module Documentation\n\n"
    
    for file_path in sorted(file_analyses.keys()):
        analyzer = file_analyses[file_path]
        functional_analysis = functional_analyses.get(file_path, "")
        
        md += f"### `{file_path}`\n\n"
        
        # Functional analysis from LLM
        md += functional_analysis + "\n\n"
        
        # Detailed API documentation
        md += "#### Complete API Reference\n\n"
        
        # Classes
        for cls in analyzer.classes:
            md += f"##### Class: `{cls['name']}`\n\n"
            if cls['docstring']:
                md += f"**Description:** {cls['docstring'][:200]}...\n\n"
            
            if cls['base_classes']:
                md += f"**Inherits from:** {', '.join([f'`{b}`' for b in cls['base_classes']])}\n\n"
            
            md += "**Methods:**\n\n"
            for method in cls['methods']:
                md += f"- **`{method['name']}`**"
                if method['parameters']:
                    params = []
                    for param in method['parameters']:
                        param_str = param['name']
                        if param.get('type'):
                            param_str += f": {param['type']}"
                        if param.get('default'):
                            param_str += f" = {param['default']}"
                        params.append(param_str)
                    md += f"({', '.join(params)})"
                
                if method.get('return_type'):
                    md += f" -> {method['return_type']}"
                
                md += "\n"
                if method['docstring']:
                    md += f"  - {method['docstring'][:150]}...\n"
                
                # Show calls made by this method
                if method['calls_made']:
                    md += "  - **Calls:** "
                    calls = [call['function_name'] for call in method['calls_made'][:5]]
                    md += ", ".join([f"`{c}`" for c in calls])
                    if len(method['calls_made']) > 5:
                        md += f" (+{len(method['calls_made'])-5} more)"
                    md += "\n"
                md += "\n"
            
            # Class usage examples
            md += "**Usage Pattern:**\n"
            md += f"```python\n"
            md += f"from {file_path.replace('.py', '').replace('/', '.')} import {cls['name']}\n"
            if cls['base_classes']:
                md += f"# Inherits from: {', '.join(cls['base_classes'])}\n"
            md += f"instance = {cls['name']}("
            
            # Show constructor parameters if __init__ exists
            init_method = next((m for m in cls['methods'] if m['name'] == '__init__'), None)
            if init_method and init_method['parameters']:
                init_params = [p['name'] for p in init_method['parameters'] if p['name'] != 'self']
                if init_params:
                    md += ", ".join(init_params)
            md += ")\n```\n\n"
        
        # Functions
        if analyzer.functions:
            md += "##### Functions\n\n"
            for func in analyzer.functions:
                md += f"- **`{func['name']}`**"
                if func['parameters']:
                    params = []
                    for param in func['parameters']:
                        param_str = param['name']
                        if param.get('type'):
                            param_str += f": {param['type']}"
                        if param.get('default'):
                            param_str += f" = {param['default']}"
                        params.append(param_str)
                    md += f"({', '.join(params)})"
                
                if func.get('return_type'):
                    md += f" -> {func['return_type']}"
                
                if func['is_async']:
                    md += " [async]"
                
                md += "\n"
                if func['docstring']:
                    md += f"  - {func['docstring'][:150]}...\n"
                
                # Show calls made by this function
                if func['calls_made']:
                    md += "  - **Calls:** "
                    calls = [call['function_name'] for call in func['calls_made'][:5]]
                    md += ", ".join([f"`{c}`" for c in calls])
                    if len(func['calls_made']) > 5:
                        md += f" (+{len(func['calls_made'])-5} more)"
                    md += "\n"
                
                # Usage example
                md += f"  - **Usage:** `{func['name']}("
                if func['parameters']:
                    example_params = [p['name'] for p in func['parameters'][:3]]
                    md += ", ".join(example_params)
                    if len(func['parameters']) > 3:
                        md += ", ..."
                md += ")`\n\n"
        
        # Constants and Variables
        if analyzer.constants:
            md += "##### Constants\n\n"
            for name, const in analyzer.constants.items():
                md += f"- **`{name}`** = `{const['value_preview']}`"
                if const.get('type'):
                    md += f" ({const['type']})"
                md += f" (line {const['line']})\n"
            md += "\n"
        
        if analyzer.variables:
            md += "##### Module Variables\n\n"
            for name, var in analyzer.variables.items():
                md += f"- **`{name}`** = `{var['value_preview']}`"
                if var.get('type'):
                    md += f" ({var['type']})"
                md += f" (line {var['line']})\n"
            md += "\n"
        
        # Dependencies
        deps = cross_ref.dependencies.get(file_path, {})
        if deps:
            md += "##### Dependencies on Other Modules\n\n"
            for dep_file, symbols in deps.items():
                md += f"- **`{dep_file}`**: uses `{', '.join(symbols[:5])}`"
                if len(symbols) > 5:
                    md += f" (+{len(symbols)-5} more)"
                md += "\n"
            md += "\n"
        
        # Show what calls this module's functions
        md += "##### Called By\n\n"
        module_functions = {f['name'] for f in analyzer.functions}
        module_classes = {c['name'] for c in analyzer.classes}
        all_module_symbols = module_functions | module_classes
        
        callers = []
        for symbol in all_module_symbols:
            if symbol in cross_ref.call_graph:
                for call_info in cross_ref.call_graph[symbol]:
                    if call_info['file'] != file_path:  # External calls only
                        callers.append((symbol, call_info))
        
        if callers:
            caller_summary = {}
            for symbol, call_info in callers:
                key = (call_info['file'], symbol)
                if key not in caller_summary:
                    caller_summary[key] = []
                caller_summary[key].append(call_info)
            
            for (caller_file, symbol), calls in sorted(caller_summary.items()):
                md += f"- **`{symbol}`** called by `{caller_file}` ({len(calls)} times)\n"
                for call in calls[:2]:  # Show first 2 examples
                    md += f"  - Line {call['line']} in {call['context']['scope']}\n"
        else:
            md += "No external callers found.\n"
        md += "\n"
        
        # Import statements
        if analyzer.imports:
            md += "##### Imports\n\n"
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            
            for imp in analyzer.imports:
                if imp['type'] == 'import':
                    import_str = imp['module']
                    if imp['alias']:
                        import_str += f" as {imp['alias']}"
                elif imp['type'] == 'from_import':
                    import_str = f"from {imp['module']} import {imp['name']}"
                    if imp['alias']:
                        import_str += f" as {imp['alias']}"
                
                # Categorize imports
                if imp['module'].startswith('.'):
                    local_imports.append(import_str)
                elif imp['module'].split('.')[0] in {
                    'os', 'sys', 'json', 'yaml', 'ast', 'pathlib', 'typing', 
                    'collections', 'itertools', 'functools', 're', 'hashlib'
                }:
                    stdlib_imports.append(import_str)
                else:
                    third_party_imports.append(import_str)
            
            if stdlib_imports:
                md += "**Standard Library:**\n"
                for imp in stdlib_imports:
                    md += f"- `{imp}`\n"
                md += "\n"
            
            if third_party_imports:
                md += "**Third Party:**\n"
                for imp in third_party_imports:
                    md += f"- `{imp}`\n"
                md += "\n"
            
            if local_imports:
                md += "**Local:**\n"
                for imp in local_imports:
                    md += f"- `{imp}`\n"
                md += "\n"
        
        md += "---\n\n"
    
    return md

if __name__ == "__main__":
    main()