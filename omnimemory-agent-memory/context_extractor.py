"""
Context Extractor - Extracts important context from conversations

Extracts structured information from conversation messages including:
- File mentions
- Code snippets
- Error messages
- Tasks
- Dependencies
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContext:
    """Structured context extracted from conversation"""

    files_mentioned: List[str]
    code_snippets: List[Dict[str, str]]
    error_messages: List[str]
    tasks_identified: List[str]
    dependencies: List[str]
    decisions: List[str]
    urls: List[str]
    technical_terms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ContextExtractor:
    """
    Extracts important context from conversations.

    Extracts:
    - Mentioned files and paths
    - Code snippets (inline and blocks)
    - Error messages and stack traces
    - Decisions made
    - Tasks identified
    - Dependencies (libraries, packages)
    - URLs
    - Technical terms
    """

    def __init__(self):
        """Initialize context extraction patterns"""
        # File path patterns
        self.file_patterns = [
            r"`([^`]+\.[a-zA-Z0-9]+)`",  # Backtick-wrapped file names
            r'"([^"]+\.[a-zA-Z0-9]+)"',  # Quote-wrapped file names
            r"\'([^\']+\.[a-zA-Z0-9]+)\'",  # Single quote-wrapped
            r"\b([\w\-/]+/[\w\-/.]+\.[a-zA-Z0-9]+)\b",  # Path-like with extension
            r"\b([A-Za-z]:\\[\w\-\\/]+\.[a-zA-Z0-9]+)\b",  # Windows paths
        ]

        # Code block patterns
        self.code_block_pattern = r"```(\w+)?\n(.*?)```"

        # Inline code pattern
        self.inline_code_pattern = r"`([^`\n]{3,})`"

        # Error patterns
        self.error_patterns = [
            r"(Error|Exception|Traceback):\s*(.+?)(?:\n|$)",
            r"(\w+Error):\s*(.+?)(?:\n|$)",
            r"failed with:\s*(.+?)(?:\n|$)",
            r"error:\s*(.+?)(?:\n|$)",
        ]

        # Task patterns
        self.task_patterns = [
            r"\b(TODO|FIXME|NOTE|HACK):\s*(.+?)(?:\n|$)",
            r"\b(need to|should|must|have to)\s+(.+?)(?:\.|,|\n|$)",
            r"\b(let\'s|we\'ll|I\'ll)\s+(.+?)(?:\.|,|\n|$)",
        ]

        # Dependency patterns
        self.dependency_patterns = [
            r"\b(import|from|require|include)\s+([a-zA-Z0-9_\-.]+)",
            r"\b(pip install|npm install|yarn add)\s+([a-zA-Z0-9_\-]+)",
            r"\b(using|with|via)\s+([A-Z][a-zA-Z0-9]+)",
        ]

        # URL pattern
        self.url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

        # Decision patterns
        self.decision_patterns = [
            r"\b(decided to|chose to|going with|will use)\s+(.+?)(?:\.|,|\n|$)",
            r"\b(instead of|rather than)\s+(.+?)(?:\.|,|\n|$)",
            r"\b(better to|makes sense to)\s+(.+?)(?:\.|,|\n|$)",
        ]

        # Technical terms (common programming terms)
        self.technical_keywords = {
            "api",
            "database",
            "server",
            "client",
            "async",
            "await",
            "function",
            "class",
            "method",
            "variable",
            "parameter",
            "argument",
            "return",
            "callback",
            "promise",
            "query",
            "endpoint",
            "route",
            "middleware",
            "authentication",
            "authorization",
            "token",
            "jwt",
            "oauth",
            "session",
            "cookie",
            "cache",
            "redis",
            "postgres",
            "mongodb",
            "docker",
            "kubernetes",
            "container",
            "deployment",
            "testing",
            "unittest",
            "integration",
            "e2e",
        }

        logger.info("ContextExtractor initialized")

    def extract_context(self, message: str) -> Dict[str, Any]:
        """
        Extract all context from a message.

        Args:
            message: Message text to extract from

        Returns:
            Dictionary with extracted context
        """
        context = ExtractedContext(
            files_mentioned=[],
            code_snippets=[],
            error_messages=[],
            tasks_identified=[],
            dependencies=[],
            decisions=[],
            urls=[],
            technical_terms=[],
        )

        try:
            # Extract files
            context.files_mentioned = self._extract_files(message)

            # Extract code snippets
            context.code_snippets = self._extract_code(message)

            # Extract errors
            context.error_messages = self._extract_errors(message)

            # Extract tasks
            context.tasks_identified = self._extract_tasks(message)

            # Extract dependencies
            context.dependencies = self._extract_dependencies(message)

            # Extract decisions
            context.decisions = self._extract_decisions(message)

            # Extract URLs
            context.urls = self._extract_urls(message)

            # Extract technical terms
            context.technical_terms = self._extract_technical_terms(message)

            logger.debug(
                f"Extracted context: {len(context.files_mentioned)} files, "
                f"{len(context.code_snippets)} code snippets, "
                f"{len(context.error_messages)} errors"
            )

            return context.to_dict()

        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return context.to_dict()

    def _extract_files(self, message: str) -> List[str]:
        """Extract file paths from message"""
        files = set()

        for pattern in self.file_patterns:
            matches = re.findall(pattern, message)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if match and len(match) < 200:  # Reasonable file path length
                    files.add(match.strip())

        return sorted(list(files))

    def _extract_code(self, message: str) -> List[Dict[str, str]]:
        """Extract code snippets from message"""
        snippets = []

        # Extract code blocks
        code_blocks = re.finditer(self.code_block_pattern, message, re.DOTALL)
        for match in code_blocks:
            language = match.group(1) or "unknown"
            code = match.group(2).strip()
            if code:
                snippets.append(
                    {
                        "type": "block",
                        "language": language,
                        "code": code[:1000],  # Limit size
                    }
                )

        # Extract inline code (only if significant)
        inline_codes = re.findall(self.inline_code_pattern, message)
        for code in inline_codes:
            code = code.strip()
            if len(code) > 10:  # Only capture substantial inline code
                snippets.append(
                    {
                        "type": "inline",
                        "language": "unknown",
                        "code": code[:500],  # Limit size
                    }
                )

        return snippets

    def _extract_errors(self, message: str) -> List[str]:
        """Extract error messages from message"""
        errors = set()

        for pattern in self.error_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    error_text = " ".join(str(m) for m in match if m)
                else:
                    error_text = str(match)

                error_text = error_text.strip()
                if error_text and len(error_text) > 5:
                    errors.add(error_text[:500])  # Limit size

        return sorted(list(errors))

    def _extract_tasks(self, message: str) -> List[str]:
        """Extract tasks from message"""
        tasks = set()

        for pattern in self.task_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    task_text = match[1] if len(match) > 1 else match[0]
                else:
                    task_text = match

                task_text = str(task_text).strip()
                if task_text and len(task_text) > 5:
                    tasks.add(task_text[:200])  # Limit size

        return sorted(list(tasks))

    def _extract_dependencies(self, message: str) -> List[str]:
        """Extract dependencies from message"""
        dependencies = set()

        for pattern in self.dependency_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    dep = match[1] if len(match) > 1 else match[0]
                else:
                    dep = match

                dep = str(dep).strip()
                if dep and len(dep) > 2:
                    dependencies.add(dep)

        return sorted(list(dependencies))

    def _extract_decisions(self, message: str) -> List[str]:
        """Extract decisions from message"""
        decisions = set()

        for pattern in self.decision_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    decision = " ".join(str(m) for m in match if m)
                else:
                    decision = str(match)

                decision = decision.strip()
                if decision and len(decision) > 5:
                    decisions.add(decision[:300])  # Limit size

        return sorted(list(decisions))

    def _extract_urls(self, message: str) -> List[str]:
        """Extract URLs from message"""
        urls = re.findall(self.url_pattern, message)
        return sorted(list(set(urls)))

    def _extract_technical_terms(self, message: str) -> List[str]:
        """Extract technical terms from message"""
        message_lower = message.lower()
        terms = set()

        # Check for known technical keywords
        for keyword in self.technical_keywords:
            if re.search(r"\b" + keyword + r"\b", message_lower):
                terms.add(keyword)

        # Look for PascalCase words (often class/type names)
        pascal_case = re.findall(r"\b([A-Z][a-z]+[A-Z][a-zA-Z]*)\b", message)
        terms.update(pascal_case[:20])  # Limit to avoid noise

        # Look for UPPER_CASE words (often constants)
        upper_case = re.findall(r"\b([A-Z_]{3,})\b", message)
        terms.update([w for w in upper_case if len(w) < 30][:10])

        return sorted(list(terms))

    def extract_batch(self, messages: List[str]) -> List[Dict[str, Any]]:
        """
        Extract context from multiple messages.

        Args:
            messages: List of message texts

        Returns:
            List of extracted contexts
        """
        return [self.extract_context(msg) for msg in messages]

    def get_statistics(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics from a batch of extracted contexts.

        Args:
            contexts: List of context dictionaries

        Returns:
            Dictionary with statistics
        """
        if not contexts:
            return {
                "total": 0,
                "total_files": 0,
                "total_code_snippets": 0,
                "total_errors": 0,
                "total_tasks": 0,
                "total_dependencies": 0,
                "avg_files_per_message": 0.0,
                "avg_code_per_message": 0.0,
            }

        total_files = sum(len(ctx.get("files_mentioned", [])) for ctx in contexts)
        total_code = sum(len(ctx.get("code_snippets", [])) for ctx in contexts)
        total_errors = sum(len(ctx.get("error_messages", [])) for ctx in contexts)
        total_tasks = sum(len(ctx.get("tasks_identified", [])) for ctx in contexts)
        total_deps = sum(len(ctx.get("dependencies", [])) for ctx in contexts)

        return {
            "total": len(contexts),
            "total_files": total_files,
            "total_code_snippets": total_code,
            "total_errors": total_errors,
            "total_tasks": total_tasks,
            "total_dependencies": total_deps,
            "avg_files_per_message": total_files / len(contexts),
            "avg_code_per_message": total_code / len(contexts),
        }
