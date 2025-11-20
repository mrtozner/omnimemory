"""
Procedural Memory Engine for OmniMemory

Learns workflow patterns from command sequences and predicts next actions
using MLX embeddings and NetworkX graphs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import networkx as nx
import numpy as np
import asyncio
import pickle
import hashlib
import re
import httpx


@dataclass
class WorkflowPattern:
    """Represents a learned workflow pattern"""

    pattern_id: str
    command_sequence: List[str]
    embeddings: List[np.ndarray]
    transitions: List[np.ndarray]
    success_count: int = 0
    failure_count: int = 0
    avg_duration_ms: float = 0.0

    @property
    def confidence(self) -> float:
        """Calculate confidence based on success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class Prediction:
    """Represents a predicted next action"""

    next_command: str
    confidence: float
    reason: str
    similar_patterns: List[str]
    auto_suggestions: List[str]


class ProceduralMemoryEngine:
    """
    Learn and predict developer workflows using causal patterns
    """

    def __init__(self, embedding_service_url: str = "http://localhost:8000"):
        self.embedding_url = embedding_service_url
        self.workflow_graph = nx.DiGraph()
        self.patterns = {}  # pattern_id -> WorkflowPattern
        self.pattern_index = {}  # embedding -> pattern_ids
        self.causal_chains = defaultdict(list)

    async def learn_workflow(
        self, session_commands: List[Dict], session_outcome: str = "success"
    ) -> str:
        """
        Learn from a session of commands
        """
        # Extract command sequence
        commands = [cmd["command"] for cmd in session_commands]

        # Skip if too short
        if len(commands) < 3:
            return None

        # Create pattern ID
        pattern_id = self._generate_pattern_id(commands)

        # Get embeddings
        embeddings = await self._get_sequence_embeddings(commands)

        # Create or update pattern
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            if session_outcome == "success":
                pattern.success_count += 1
            else:
                pattern.failure_count += 1
        else:
            pattern = WorkflowPattern(
                pattern_id=pattern_id,
                command_sequence=commands,
                embeddings=embeddings["command_embeddings"],
                transitions=embeddings["transition_embeddings"],
                success_count=1 if session_outcome == "success" else 0,
                failure_count=1 if session_outcome == "failure" else 0,
            )
            self.patterns[pattern_id] = pattern

        # Update graph
        self._update_workflow_graph(commands, session_outcome)

        # Update causal chains
        self._update_causal_chains(commands, session_outcome)

        return pattern_id

    async def predict_next_action(
        self, current_context: List[str], top_k: int = 3
    ) -> List[Prediction]:
        """
        Predict next likely actions based on current context
        """
        if len(current_context) < 2:
            return []

        # Get embeddings for current context
        context_embeddings = await self._get_sequence_embeddings(current_context)

        # Find similar patterns
        similar_patterns = self._find_similar_patterns(
            context_embeddings["sequence_embedding"]
        )

        # Generate predictions
        predictions = []

        for pattern_id, similarity in similar_patterns[:top_k]:
            pattern = self.patterns[pattern_id]

            # Find where we are in this pattern
            position = self._find_position_in_pattern(
                current_context, pattern.command_sequence
            )

            if position >= 0 and position < len(pattern.command_sequence) - 1:
                next_cmd = pattern.command_sequence[position + 1]

                # Get all commands that follow this context in the graph
                graph_suggestions = self._get_graph_suggestions(current_context[-1])

                prediction = Prediction(
                    next_command=next_cmd,
                    confidence=pattern.confidence * similarity,
                    reason=f"Based on pattern with {pattern.success_count} successes",
                    similar_patterns=[pattern_id],
                    auto_suggestions=graph_suggestions[:3],
                )
                predictions.append(prediction)

        # Sort by confidence
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        return predictions

    def _find_similar_patterns(
        self, embedding: np.ndarray, threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find patterns similar to given embedding
        """
        similarities = []

        for pattern_id, pattern in self.patterns.items():
            # Calculate similarity with pattern's sequence embedding
            pattern_embedding = np.mean(pattern.embeddings, axis=0)
            similarity = self._cosine_similarity(embedding, pattern_embedding)

            if similarity >= threshold:
                similarities.append((pattern_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def _find_position_in_pattern(self, context: List[str], pattern: List[str]) -> int:
        """
        Find where the current context matches in the pattern
        """
        context_str = " -> ".join(context[-3:])  # Last 3 commands

        for i in range(len(pattern) - len(context) + 1):
            pattern_slice = " -> ".join(pattern[i : i + len(context)])
            if self._fuzzy_match(context_str, pattern_slice):
                return i + len(context) - 1

        return -1

    def _update_workflow_graph(self, commands: List[str], outcome: str):
        """
        Update the workflow graph with command transitions
        """
        for i in range(len(commands) - 1):
            from_cmd = self._normalize_command(commands[i])
            to_cmd = self._normalize_command(commands[i + 1])

            # Add edge or update weight
            if self.workflow_graph.has_edge(from_cmd, to_cmd):
                self.workflow_graph[from_cmd][to_cmd]["weight"] += 1
                if outcome == "success":
                    self.workflow_graph[from_cmd][to_cmd]["success"] += 1
            else:
                self.workflow_graph.add_edge(
                    from_cmd, to_cmd, weight=1, success=1 if outcome == "success" else 0
                )

    def _update_causal_chains(self, commands: List[str], outcome: str):
        """
        Update causal chains tracking command relationships
        """
        for i in range(len(commands) - 1):
            from_cmd = self._normalize_command(commands[i])
            to_cmd = self._normalize_command(commands[i + 1])

            # Check if this transition already exists
            existing = None
            for chain in self.causal_chains[from_cmd]:
                if chain["next"] == to_cmd and chain["outcome"] == outcome:
                    existing = chain
                    break

            if existing:
                existing["count"] += 1
            else:
                self.causal_chains[from_cmd].append(
                    {"next": to_cmd, "outcome": outcome, "count": 1}
                )

    def _get_graph_suggestions(self, command: str) -> List[str]:
        """
        Get likely next commands from graph
        """
        normalized = self._normalize_command(command)

        if normalized not in self.workflow_graph:
            return []

        # Get all outgoing edges with weights
        edges = []
        for successor in self.workflow_graph.successors(normalized):
            edge_data = self.workflow_graph[normalized][successor]
            success_rate = edge_data["success"] / edge_data["weight"]
            edges.append((successor, edge_data["weight"], success_rate))

        # Sort by success rate * frequency
        edges.sort(key=lambda x: x[2] * x[1], reverse=True)

        return [edge[0] for edge in edges]

    def _normalize_command(self, command: str) -> str:
        """
        Normalize command for graph storage
        """
        # Remove specific arguments but keep command structure
        # Order matters: URLs first, then files, then numbers

        # Replace URLs (do this first before file pattern matches)
        normalized = re.sub(r"https?://[^\s]+", "<URL>", command)
        # Replace file paths with placeholders
        normalized = re.sub(r"[/\w]+\.\w+", "<FILE>", normalized)
        # Replace numbers
        normalized = re.sub(r"\b\d+\b", "<NUM>", normalized)

        return normalized.strip()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _fuzzy_match(self, a: str, b: str, threshold: float = 0.8) -> bool:
        """Fuzzy string matching for commands"""
        # Simple implementation - can use fuzzywuzzy for better matching
        a_parts = set(a.split())
        b_parts = set(b.split())

        intersection = len(a_parts & b_parts)
        union = len(a_parts | b_parts)

        if union == 0:
            return False

        return (intersection / union) >= threshold

    async def _get_sequence_embeddings(self, commands: List[str]) -> Dict:
        """Get embeddings from MLX service"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.embedding_url}/embed/command-sequence",
                    json={"commands": commands},
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                # Return mock embeddings if service unavailable
                print(f"Warning: Embedding service unavailable: {e}")
                return {
                    "command_embeddings": [
                        np.random.randn(768).tolist() for _ in commands
                    ],
                    "transition_embeddings": [
                        np.random.randn(768).tolist() for _ in range(len(commands) - 1)
                    ],
                    "sequence_embedding": np.random.randn(768).tolist(),
                }

    def _generate_pattern_id(self, commands: List[str]) -> str:
        """Generate unique ID for pattern"""
        normalized = [self._normalize_command(cmd) for cmd in commands]
        pattern_str = " -> ".join(normalized)
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]

    def save(self, filepath: str):
        """Save the procedural memory to disk"""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "patterns": self.patterns,
                    "graph": self.workflow_graph,
                    "causal_chains": dict(self.causal_chains),
                },
                f,
            )

    def load(self, filepath: str):
        """Load procedural memory from disk"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.patterns = data["patterns"]
            self.workflow_graph = data["graph"]
            self.causal_chains = defaultdict(list, data["causal_chains"])
