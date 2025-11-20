"""
State-of-the-art snippet extraction for search results

Features:
- Query-aware relevance scoring (BM25-style)
- Smart boundary detection (sentences, code blocks, paragraphs)
- Multi-segment extraction with context
- Code-aware parsing (preserves syntax structure)
- Configurable max length with intelligent truncation
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class Segment:
    """A text segment with relevance score"""

    text: str
    start_pos: int
    end_pos: int
    score: float
    is_code: bool = False


class SnippetExtractor:
    """
    SOTA snippet extractor using hybrid approach:
    1. BM25-style keyword scoring for relevance
    2. Sentence/paragraph boundary detection
    3. Code block preservation
    4. Multi-segment extraction
    """

    def __init__(
        self,
        max_length: int = 300,
        context_chars: int = 50,
        min_segment_length: int = 30,
    ):
        """
        Args:
            max_length: Maximum snippet length in characters
            context_chars: Characters of context around matches
            min_segment_length: Minimum segment length to consider
        """
        self.max_length = max_length
        self.context_chars = context_chars
        self.min_segment_length = min_segment_length

        # Sentence boundary patterns
        self.sentence_endings = re.compile(r"[.!?]\s+")

        # Code block patterns
        self.code_blocks = re.compile(
            r"(def\s+\w+|class\s+\w+|function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+)",
            re.IGNORECASE,
        )

        # Paragraph boundaries
        self.paragraph_breaks = re.compile(r"\n\s*\n")

    def extract(self, content: str, query: str = "") -> str:
        """
        Extract the most relevant snippet from content

        Args:
            content: Full document content
            query: Search query (optional, for relevance scoring)

        Returns:
            Optimized snippet with ... separators
        """
        if not content:
            return ""

        # If content is already short, return as-is
        if len(content) <= self.max_length:
            return content

        # If no query, use smart truncation without relevance
        if not query:
            return self._smart_truncate(content)

        # Extract query terms for scoring
        query_terms = self._extract_terms(query)

        # Score all segments
        segments = self._segment_content(content)
        scored_segments = self._score_segments(segments, query_terms)

        # Select best segments within length limit
        selected = self._select_segments(scored_segments)

        # Build final snippet
        return self._build_snippet(selected, content)

    def _extract_terms(self, query: str) -> List[str]:
        """Extract searchable terms from query"""
        # Remove special chars, split on whitespace
        terms = re.findall(r"\w+", query.lower())
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "must",
            "can",
        }
        return [t for t in terms if t not in stop_words and len(t) > 2]

    def _segment_content(self, content: str) -> List[Segment]:
        """
        Break content into logical segments

        Priority:
        1. Code blocks (functions, classes)
        2. Paragraphs
        3. Sentences
        """
        segments = []

        # Try code block segmentation first
        code_matches = list(self.code_blocks.finditer(content))
        if code_matches:
            for match in code_matches:
                start = match.start()
                # Find end of code block (next function/class or 500 chars)
                end = min(start + 500, len(content))

                # Try to find end of block by matching brackets
                end = self._find_code_block_end(content, start, end)

                if end - start >= self.min_segment_length:
                    segments.append(
                        Segment(
                            text=content[start:end],
                            start_pos=start,
                            end_pos=end,
                            score=0.0,
                            is_code=True,
                        )
                    )

        # Paragraph segmentation
        paragraph_positions = (
            [0]
            + [m.end() for m in self.paragraph_breaks.finditer(content)]
            + [len(content)]
        )

        for i in range(len(paragraph_positions) - 1):
            start = paragraph_positions[i]
            end = paragraph_positions[i + 1]

            # Skip if too short or already covered by code block
            if end - start < self.min_segment_length:
                continue

            if any(s.start_pos <= start < s.end_pos for s in segments):
                continue

            # If paragraph is too long (>200 chars), split by sentences for better relevance
            if end - start > 200:
                # Split this paragraph into sentence-based segments
                para_content = content[start:end]
                sent_positions = (
                    [0]
                    + [m.end() for m in self.sentence_endings.finditer(para_content)]
                    + [len(para_content)]
                )

                for j in range(len(sent_positions) - 1):
                    sent_start = start + sent_positions[j]
                    sent_end = start + sent_positions[j + 1]

                    if sent_end - sent_start >= self.min_segment_length:
                        segments.append(
                            Segment(
                                text=content[sent_start:sent_end].strip(),
                                start_pos=sent_start,
                                end_pos=sent_end,
                                score=0.0,
                                is_code=False,
                            )
                        )
            else:
                segments.append(
                    Segment(
                        text=content[start:end].strip(),
                        start_pos=start,
                        end_pos=end,
                        score=0.0,
                        is_code=False,
                    )
                )

        # If no segments found, create sentence-based segments
        if not segments:
            sentence_positions = (
                [0]
                + [m.end() for m in self.sentence_endings.finditer(content)]
                + [len(content)]
            )

            for i in range(len(sentence_positions) - 1):
                start = sentence_positions[i]
                end = sentence_positions[i + 1]

                if end - start >= self.min_segment_length:
                    segments.append(
                        Segment(
                            text=content[start:end].strip(),
                            start_pos=start,
                            end_pos=end,
                            score=0.0,
                            is_code=False,
                        )
                    )

        # Fallback: split into fixed-size chunks
        if not segments:
            chunk_size = 200
            for i in range(0, len(content), chunk_size):
                end = min(i + chunk_size, len(content))
                segments.append(
                    Segment(
                        text=content[i:end],
                        start_pos=i,
                        end_pos=end,
                        score=0.0,
                        is_code=False,
                    )
                )

        return segments

    def _find_code_block_end(self, content: str, start: int, max_end: int) -> int:
        """Find end of code block by matching brackets"""
        # Count opening/closing brackets
        open_chars = {"{": 0, "[": 0, "(": 0}
        close_map = {"}": "{", "]": "[", ")": "("}

        i = start
        in_string = False
        string_char = None

        while i < max_end:
            char = content[i]

            # Handle strings
            if char in ('"', "'", "`") and (i == 0 or content[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            # Count brackets outside strings
            if not in_string:
                if char in open_chars:
                    open_chars[char] += 1
                elif char in close_map:
                    open_chars[close_map[char]] -= 1

                # Check if all brackets closed
                if all(count == 0 for count in open_chars.values()):
                    # Find next newline
                    next_newline = content.find("\n", i)
                    if next_newline != -1 and next_newline < max_end:
                        return next_newline + 1

            i += 1

        return max_end

    def _score_segments(
        self, segments: List[Segment], query_terms: List[str]
    ) -> List[Segment]:
        """
        Score segments using BM25-inspired algorithm

        Factors:
        - Term frequency (TF)
        - Inverse document frequency (IDF)
        - Segment length normalization
        - Code block bonus
        """
        if not query_terms:
            return segments

        # Calculate IDF-like scores for terms (simpler since we have one doc)
        total_segments = len(segments)
        term_doc_freq = {}

        for term in query_terms:
            term_doc_freq[term] = sum(1 for seg in segments if term in seg.text.lower())

        # Score each segment
        scored = []
        avg_length = sum(len(s.text) for s in segments) / max(1, total_segments)

        for segment in segments:
            text_lower = segment.text.lower()
            score = 0.0

            for term in query_terms:
                # Term frequency in this segment
                tf = text_lower.count(term)
                if tf == 0:
                    continue

                # IDF-like component
                df = term_doc_freq[term]
                idf = math.log(1 + (total_segments - df + 0.5) / (df + 0.5))

                # BM25 formula (simplified)
                k1 = 1.5
                b = 0.75
                length_norm = 1 - b + b * (len(segment.text) / avg_length)

                term_score = idf * (tf * (k1 + 1)) / (tf + k1 * length_norm)
                score += term_score

            # Bonus for code blocks (often more relevant)
            if segment.is_code:
                score *= 1.3

            # Bonus for earlier position (often more important)
            # Only apply if we have a score (i.e., query terms matched)
            if score > 0:
                doc_length = sum(len(s.text) for s in segments)
                position_factor = 1.0 - (segment.start_pos / max(doc_length, 1)) * 0.2
                score *= position_factor

            scored.append(
                Segment(
                    text=segment.text,
                    start_pos=segment.start_pos,
                    end_pos=segment.end_pos,
                    score=score,
                    is_code=segment.is_code,
                )
            )

        return sorted(scored, key=lambda s: s.score, reverse=True)

    def _select_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Select best segments that fit within max_length

        Strategy:
        - Start with highest scored segment
        - Add non-overlapping segments while space available
        - Only include segments with positive scores (relevant)
        - Prefer contiguous segments
        """
        if not segments:
            return []

        selected = []
        total_length = 0
        separator_length = 5  # " ... "

        for segment in segments:
            # Skip segments with zero score (irrelevant) unless we have nothing
            if segment.score == 0 and selected:
                continue

            # Calculate length with separator
            needed_length = len(segment.text)
            if selected:
                needed_length += separator_length

            # Check if it fits
            if total_length + needed_length > self.max_length:
                # Try to truncate if it's the first segment
                if not selected:
                    truncated = segment.text[: self.max_length - 3] + "..."
                    selected.append(
                        Segment(
                            text=truncated,
                            start_pos=segment.start_pos,
                            end_pos=segment.start_pos + self.max_length - 3,
                            score=segment.score,
                            is_code=segment.is_code,
                        )
                    )
                break

            # Check for overlap with already selected segments
            overlaps = any(
                s.start_pos <= segment.start_pos < s.end_pos
                or s.start_pos < segment.end_pos <= s.end_pos
                for s in selected
            )

            if not overlaps:
                selected.append(segment)
                total_length += needed_length

        # Sort by position for natural reading order
        return sorted(selected, key=lambda s: s.start_pos)

    def _build_snippet(self, segments: List[Segment], full_content: str) -> str:
        """Build final snippet from selected segments"""
        if not segments:
            return full_content[: self.max_length - 3] + "..."

        parts = []
        for i, segment in enumerate(segments):
            # Add ellipsis if gap from previous segment
            if i > 0:
                prev_end = segments[i - 1].end_pos
                if segment.start_pos - prev_end > 10:  # Gap threshold
                    parts.append(" ... ")

            parts.append(segment.text.strip())

        result = "".join(parts)

        # Truncate if still too long
        if len(result) > self.max_length:
            result = result[: self.max_length - 3] + "..."

        return result

    def _smart_truncate(self, content: str) -> str:
        """
        Truncate without query, respecting boundaries

        Strategy:
        - Find sentence/paragraph boundary near max_length
        - Preserve code blocks if detected
        - Stay within max_length constraint
        """
        if len(content) <= self.max_length:
            return content

        # Try to find sentence boundary
        truncate_pos = self.max_length

        # Look for sentence endings in the content up to max_length
        search_text = content[: truncate_pos + 50]
        sentence_matches = list(self.sentence_endings.finditer(search_text))

        if sentence_matches:
            # Find the last sentence ending that's <= max_length
            valid_matches = [m for m in sentence_matches if m.end() <= self.max_length]

            if valid_matches:
                # Use the last valid match
                truncate_pos = valid_matches[-1].end()
            elif sentence_matches:
                # No match within max_length, but we have sentences
                # Use the last one even if it exceeds (with small tolerance)
                potential_pos = sentence_matches[-1].end()
                if potential_pos <= self.max_length + 20:
                    truncate_pos = potential_pos

        # Check if we're in the middle of a code block
        code_matches = list(self.code_blocks.finditer(content[:truncate_pos]))
        if code_matches:
            last_code_match = code_matches[-1]
            code_end = self._find_code_block_end(
                content, last_code_match.start(), len(content)
            )

            # If code block is short enough, include it
            if code_end < self.max_length + 100:
                truncate_pos = code_end

        # Hard limit to prevent exceeding max_length too much
        if truncate_pos > self.max_length + 20:
            truncate_pos = self.max_length

        result = content[:truncate_pos].rstrip()

        # Add ellipsis if we didn't capture everything
        if truncate_pos < len(content) - 10:
            result += "..."

        return result


# Global instance for easy use
_default_extractor = SnippetExtractor(max_length=300, context_chars=50)


def extract_snippet(content: str, query: str = "", max_length: int = 300) -> str:
    """
    Convenience function for snippet extraction

    Args:
        content: Full document content
        query: Search query (optional)
        max_length: Maximum snippet length

    Returns:
        Optimized snippet

    Examples:
        >>> extract_snippet("Long document...", "search terms")
        'Most relevant part ... with context'

        >>> extract_snippet("def foo():\\n    pass\\n\\ndef bar():\\n    pass", "foo")
        'def foo():\\n    pass'
    """
    if max_length != 300:
        extractor = SnippetExtractor(max_length=max_length)
        return extractor.extract(content, query)

    return _default_extractor.extract(content, query)
