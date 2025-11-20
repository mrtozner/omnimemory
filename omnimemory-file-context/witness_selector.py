"""
Witness Selector using MMR (Maximal Marginal Relevance) algorithm
Selects 3-5 most representative code snippets from a file
"""

import numpy as np
from typing import List, Dict
import sys
import asyncio
from pathlib import Path

# Add embeddings service to path
sys.path.append(str(Path(__file__).parent.parent / "omnimemory-embeddings" / "src"))

# Try importing MLX embedding service (optional - graceful degradation)
try:
    from mlx_embedding_service import MLXEmbeddingService

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    MLXEmbeddingService = None


class WitnessSelector:
    """Select representative snippets using MMR (Maximal Marginal Relevance)."""

    def __init__(self):
        """Initialize the witness selector with MLX embeddings."""
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX embedding service not available - install mlx or use alternative witness selector"
            )
        self.embedder = MLXEmbeddingService()
        self._initialized = False

    async def initialize(self):
        """Initialize the embedding service asynchronously."""
        if not self._initialized:
            await self.embedder.initialize()
            self._initialized = True

    async def select_witnesses(
        self, file_content: str, max_witnesses: int = 5, lambda_param: float = 0.7
    ) -> List[Dict]:
        """
        Select 3-5 most representative snippets using MMR.

        MMR balances:
        - Relevance: How well snippet represents the file
        - Diversity: How different from already selected snippets

        Formula: MMR = λ * Relevance(snippet, file) - (1-λ) * max(Similarity(snippet, selected))

        Args:
            file_content: Full file content
            max_witnesses: Max number of snippets (default 5)
            lambda_param: Balance between relevance and diversity (default 0.7)
                         Higher λ = more relevance, lower = more diversity

        Returns:
            List of dicts with keys: text, type, line, score
            Example:
            [
                {"text": "def authenticate_user(...):", "type": "function_signature", "line": 10, "score": 0.95},
                {"text": "class AuthManager:", "type": "class_declaration", "line": 5, "score": 0.88},
                ...
            ]
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Step 1: Extract candidate snippets
        candidates = self._extract_candidates(file_content)

        if not candidates:
            return []

        # If we have fewer candidates than max_witnesses, return all
        if len(candidates) <= max_witnesses:
            # Still embed to get scores
            file_emb = await self.embedder.embed_text(file_content)
            for candidate in candidates:
                cand_emb = await self.embedder.embed_text(candidate["text"])
                candidate["score"] = float(self._cosine_similarity(cand_emb, file_emb))
            return candidates

        # Step 2: Embed file and all candidates
        file_emb = await self.embedder.embed_text(file_content)
        candidate_texts = [c["text"] for c in candidates]
        candidate_embs = await self.embedder.embed_batch(candidate_texts)

        # Step 3: MMR selection
        selected = []
        selected_embs = []
        selected_indices = []  # Track indices in order

        # Calculate all relevances to file once
        relevances = [self._cosine_similarity(emb, file_emb) for emb in candidate_embs]

        # First: highest relevance to file
        first_idx = np.argmax(relevances)
        selected.append(candidates[first_idx])
        selected_embs.append(candidate_embs[first_idx])
        selected_indices.append(first_idx)

        # Rest: MMR score
        while len(selected) < max_witnesses:
            best_mmr = -1
            best_idx = None

            for idx in range(len(candidates)):
                if idx in selected_indices:
                    continue

                # Relevance to file
                relevance = relevances[idx]

                # Max similarity to already selected
                similarities = [
                    self._cosine_similarity(candidate_embs[idx], sel_emb)
                    for sel_emb in selected_embs
                ]
                max_similarity = max(similarities)

                # MMR score: balance relevance and diversity
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(candidates[best_idx])
            selected_embs.append(candidate_embs[best_idx])
            selected_indices.append(best_idx)

        # Add scores to selected witnesses
        for i, witness in enumerate(selected):
            witness["score"] = float(relevances[selected_indices[i]])

        return selected

    def _extract_candidates(self, content: str) -> List[Dict]:
        """
        Extract candidate snippets from code.

        Priority (in order):
        1. Function signatures
        2. Class declarations
        3. Import statements
        4. Type definitions
        5. Decorators
        """
        candidates = []
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines and comments-only lines
            if not stripped or stripped.startswith("#"):
                i += 1
                continue

            # Python decorators (include with next line)
            if stripped.startswith("@"):
                decorator = stripped
                i += 1
                # Get the decorated element
                while i < len(lines) and (
                    not lines[i].strip() or lines[i].strip().startswith("@")
                ):
                    if lines[i].strip().startswith("@"):
                        decorator += "\n" + lines[i].strip()
                    i += 1

                if i < len(lines):
                    next_line = lines[i].strip()
                    if (
                        next_line.startswith("def ")
                        or next_line.startswith("async def ")
                        or next_line.startswith("class ")
                    ):
                        candidates.append(
                            {
                                "text": decorator + "\n" + next_line,
                                "type": "decorated_definition",
                                "line": i - decorator.count("\n"),
                            }
                        )
                i += 1
                continue

            # Function signatures
            if stripped.startswith("def ") or stripped.startswith("async def "):
                # Include signature up to colon
                sig = stripped
                j = i + 1
                # Multi-line signatures
                while j < len(lines) and ":" not in sig:
                    sig += " " + lines[j].strip()
                    j += 1

                candidates.append(
                    {"text": sig, "type": "function_signature", "line": i + 1}
                )
                i = j
                continue

            # Class declarations
            elif stripped.startswith("class "):
                candidates.append(
                    {"text": stripped, "type": "class_declaration", "line": i + 1}
                )

            # Imports (group consecutive imports)
            elif stripped.startswith("import ") or stripped.startswith("from "):
                import_block = [stripped]
                j = i + 1
                while j < len(lines):
                    next_stripped = lines[j].strip()
                    if next_stripped.startswith("import ") or next_stripped.startswith(
                        "from "
                    ):
                        import_block.append(next_stripped)
                        j += 1
                    elif not next_stripped:  # Allow blank lines in import block
                        j += 1
                    else:
                        break

                # Only add as single import, not grouped
                candidates.append({"text": stripped, "type": "import", "line": i + 1})
                i = j - 1

            # TypeScript/JavaScript interfaces and types
            elif (
                stripped.startswith("interface ")
                or stripped.startswith("type ")
                or stripped.startswith("export interface ")
                or stripped.startswith("export type ")
            ):
                candidates.append(
                    {"text": stripped, "type": "type_definition", "line": i + 1}
                )

            # TypeScript/JavaScript classes
            elif stripped.startswith("export class ") or stripped.startswith(
                "export default class "
            ):
                candidates.append(
                    {"text": stripped, "type": "class_declaration", "line": i + 1}
                )

            # TypeScript/JavaScript functions
            elif (
                stripped.startswith("function ")
                or stripped.startswith("async function ")
                or stripped.startswith("export function ")
                or stripped.startswith("export async function ")
            ):
                sig = stripped
                j = i + 1
                # Multi-line signatures
                while j < len(lines) and (
                    "{" not in sig or sig.count("(") != sig.count(")")
                ):
                    sig += " " + lines[j].strip()
                    j += 1
                    if "{" in sig:
                        break

                candidates.append(
                    {
                        "text": sig.split("{")[0].strip(),  # Signature only, no body
                        "type": "function_signature",
                        "line": i + 1,
                    }
                )
                i = j - 1

            i += 1

        return candidates

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
