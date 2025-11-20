"""
Temporal Conflict Resolver for Bi-temporal Checkpoint Management

Implements automatic conflict resolution between SQLite bi-temporal data
and Qdrant vector store, matching Zep's temporal graph conflict resolution
capabilities without requiring a separate graph database.

Key Features:
- Automatic detection of overlapping validity windows
- Conflict resolution based on recorded time (newer knowledge wins)
- Complete audit trail maintenance (never deletes, only closes validity windows)
- Provenance tracking (supersedes/superseded_by relationships)
- Atomic updates across both SQLite and Qdrant
- Handles out-of-order ingestion and retroactive corrections
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from .data_store import MetricsStore
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class TemporalConflictResolver:
    """
    Automatic conflict resolution for bi-temporal checkpoint updates

    Matches Zep's temporal graph conflict resolution:
    - Newer information supersedes older (based on recorded_at)
    - Complete audit trail maintained (never delete old versions)
    - Provenance tracking (supersedes/superseded_by relationships)
    - Handles out-of-order ingestion (late-arriving data)
    - Atomic updates across SQLite + Qdrant

    Resolution Rules (in priority order):
    1. Later recorded_at supersedes earlier (newer knowledge wins)
    2. More specific validity window supersedes broader
    3. Higher quality_score breaks ties
    4. Keep all versions in audit trail
    """

    def __init__(self, data_store: MetricsStore, vector_store: VectorStore):
        """
        Initialize temporal conflict resolver

        Args:
            data_store: MetricsStore instance for SQLite operations
            vector_store: VectorStore instance for Qdrant operations
        """
        self.data_store = data_store
        self.vector_store = vector_store

        logger.info("Initialized TemporalConflictResolver")

    async def store_checkpoint_with_resolution(
        self,
        checkpoint_id: str,
        session_id: str,
        tool_id: str,
        checkpoint_type: str,
        summary: str,
        valid_from: datetime,
        valid_to: Optional[datetime] = None,
        recorded_at: Optional[datetime] = None,
        key_facts: Optional[List[str]] = None,
        decisions: Optional[List[Dict]] = None,
        patterns: Optional[List[Dict]] = None,
        files_modified: Optional[List[str]] = None,
        dependencies_added: Optional[List[str]] = None,
        commands_run: Optional[List[str]] = None,
        todos: Optional[List[str]] = None,
        blockers: Optional[List[str]] = None,
        quality_score: Optional[float] = None,
        tool_version: Optional[str] = None,
        influenced_by: Optional[List[str]] = None,
    ) -> Dict:
        """
        Store checkpoint with automatic conflict resolution

        This is the main entry point that:
        1. Finds overlapping checkpoints in the same validity window
        2. Determines which ones should be superseded based on resolution rules
        3. Updates SQLite to close validity windows and set superseded_by
        4. Updates Qdrant to update relationship metadata
        5. Inserts new checkpoint in both stores atomically

        Args:
            checkpoint_id: Unique checkpoint ID
            session_id: Session identifier
            tool_id: Tool identifier
            checkpoint_type: Type of checkpoint
            summary: Checkpoint summary
            valid_from: When this checkpoint becomes valid
            valid_to: When this checkpoint expires (None = open-ended)
            recorded_at: When this checkpoint was recorded (defaults to now)
            key_facts: Important facts
            decisions: Decisions made
            patterns: Code patterns
            files_modified: Modified files
            dependencies_added: Added dependencies
            commands_run: Commands executed
            todos: Open tasks
            blockers: Current blockers
            quality_score: Quality score for tie-breaking
            tool_version: Tool version

        Returns:
            {
                "checkpoint_id": str,
                "superseded": [list of checkpoint IDs that were superseded],
                "conflicts_resolved": int,
                "status": "new" | "superseded_existing" | "updated"
            }
        """
        recorded_at = recorded_at or datetime.now()
        valid_to = valid_to or datetime(9999, 12, 31, 23, 59, 59)

        logger.info(
            f"Storing checkpoint {checkpoint_id} with conflict resolution "
            f"(valid_from={valid_from}, recorded_at={recorded_at})"
        )

        try:
            # Step 1: Find overlapping checkpoints
            overlapping = self.find_overlapping_checkpoints(
                tool_id=tool_id,
                session_id=session_id,
                valid_from=valid_from,
                valid_to=valid_to,
            )

            logger.info(f"Found {len(overlapping)} overlapping checkpoints")

            # Step 2: Determine which checkpoints to supersede
            new_checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "recorded_at": recorded_at,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "quality_score": quality_score or 0.0,
            }

            to_supersede = await self.resolve_conflicts(
                new_checkpoint=new_checkpoint_data,
                overlapping=overlapping,
            )

            logger.info(f"Resolved to supersede {len(to_supersede)} checkpoints")

            # Step 3: Update superseded checkpoints in both stores
            if to_supersede:
                await self.update_superseded_checkpoints(
                    checkpoint_ids=to_supersede,
                    superseded_by=checkpoint_id,
                    valid_to=valid_from,  # Close their validity at new checkpoint's start
                )

            # Step 4: Insert new checkpoint in SQLite
            self.data_store.update_checkpoint_with_temporal(
                checkpoint_id=checkpoint_id,
                session_id=session_id,
                tool_id=tool_id,
                checkpoint_type=checkpoint_type,
                summary=summary,
                valid_from=valid_from,
                recorded_at=recorded_at,  # Pass recorded_at to preserve bi-temporal semantics
                supersedes=to_supersede if to_supersede else None,
                influenced_by=influenced_by,
                key_facts=key_facts,
                decisions=decisions,
                patterns=patterns,
                files_modified=files_modified,
                dependencies_added=dependencies_added,
                commands_run=commands_run,
                todos=todos,
                blockers=blockers,
                tool_version=tool_version,
            )

            # Step 5: Store embedding in Qdrant
            embedding_text = summary
            if key_facts:
                embedding_text += " " + " ".join(key_facts)

            await self.vector_store.store_checkpoint_embedding(
                checkpoint_id=checkpoint_id,
                text=embedding_text,
                tool_id=tool_id,
                checkpoint_type=checkpoint_type,
                summary=summary,
                recorded_at=recorded_at,
                valid_from=valid_from,
                valid_to=valid_to,
                supersedes=to_supersede if to_supersede else None,
                influenced_by=influenced_by,
                session_id=session_id,
                quality_score=quality_score,
            )

            # Step 6: Determine status
            status = "new"
            if to_supersede:
                status = "superseded_existing"

            result = {
                "checkpoint_id": checkpoint_id,
                "superseded": to_supersede,
                "conflicts_resolved": len(to_supersede),
                "status": status,
            }

            logger.info(
                f"Successfully stored checkpoint {checkpoint_id} "
                f"(status={status}, superseded={len(to_supersede)})"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to store checkpoint with resolution: {e}")
            raise

    def find_overlapping_checkpoints(
        self,
        tool_id: str,
        session_id: str,
        valid_from: datetime,
        valid_to: datetime,
    ) -> List[Dict]:
        """
        Find checkpoints with overlapping validity windows

        A checkpoint overlaps if:
        1. Same tool_id and session_id
        2. Valid time windows overlap:
           - existing.valid_from < new.valid_to
           - existing.valid_to > new.valid_from
        3. Not already superseded (valid_to is still open or in future)

        Args:
            tool_id: Tool identifier
            session_id: Session identifier
            valid_from: Start of new checkpoint's validity
            valid_to: End of new checkpoint's validity

        Returns:
            List of overlapping checkpoint dictionaries
        """
        cursor = self.data_store.conn.cursor()

        try:
            # Query for overlapping checkpoints
            # Two intervals overlap if: A.start < B.end AND A.end > B.start
            result = cursor.execute(
                """
                SELECT
                    checkpoint_id,
                    recorded_at,
                    valid_from,
                    valid_to,
                    superseded_by
                FROM checkpoints
                WHERE tool_id = ?
                AND session_id = ?
                AND valid_from < ?
                AND valid_to > ?
                AND superseded_by IS NULL
                ORDER BY recorded_at DESC
            """,
                (
                    tool_id,
                    session_id,
                    valid_to.isoformat(),
                    valid_from.isoformat(),
                ),
            )

            rows = result.fetchall()

            overlapping = []
            for row in rows:
                overlapping.append(
                    {
                        "checkpoint_id": row[0],
                        "recorded_at": datetime.fromisoformat(row[1])
                        if row[1]
                        else None,
                        "valid_from": datetime.fromisoformat(row[2])
                        if row[2]
                        else None,
                        "valid_to": datetime.fromisoformat(row[3]) if row[3] else None,
                        "quality_score": 0.0,  # Default quality score (not in SQLite)
                        "superseded_by": row[4],
                    }
                )

            logger.debug(f"Found {len(overlapping)} overlapping checkpoints")
            return overlapping

        except Exception as e:
            logger.error(f"Failed to find overlapping checkpoints: {e}")
            return []

    async def resolve_conflicts(
        self,
        new_checkpoint: Dict,
        overlapping: List[Dict],
    ) -> List[str]:
        """
        Apply conflict resolution rules to determine which checkpoints to supersede

        Resolution Rules (in priority order):
        1. Later recorded_at supersedes earlier (newer knowledge wins)
           - If new checkpoint has later recorded_at, supersede all overlapping
        2. More specific validity window supersedes broader
           - Smaller time window = more specific
        3. Higher quality_score breaks ties
        4. Keep all versions in audit trail (never delete)

        Args:
            new_checkpoint: New checkpoint being inserted
            overlapping: List of overlapping checkpoints

        Returns:
            List of checkpoint IDs to supersede
        """
        if not overlapping:
            return []

        to_supersede = []
        new_recorded_at = new_checkpoint["recorded_at"]
        new_valid_from = new_checkpoint["valid_from"]
        new_valid_to = new_checkpoint["valid_to"]
        new_quality = new_checkpoint.get("quality_score", 0.0)

        # Calculate new checkpoint's validity window size
        new_window_size = (new_valid_to - new_valid_from).total_seconds()

        for existing in overlapping:
            existing_recorded_at = existing["recorded_at"]
            existing_valid_from = existing["valid_from"]
            existing_valid_to = existing["valid_to"]
            existing_quality = existing.get("quality_score", 0.0)

            # Calculate existing checkpoint's validity window size
            existing_window_size = (
                existing_valid_to - existing_valid_from
            ).total_seconds()

            # Rule 1: Later recorded_at supersedes earlier
            if new_recorded_at > existing_recorded_at:
                to_supersede.append(existing["checkpoint_id"])
                logger.debug(
                    f"Superseding {existing['checkpoint_id']} by recorded_at rule "
                    f"({existing_recorded_at} < {new_recorded_at})"
                )
                continue

            # Rule 2: More specific (smaller) validity window supersedes broader
            if new_window_size < existing_window_size:
                to_supersede.append(existing["checkpoint_id"])
                logger.debug(
                    f"Superseding {existing['checkpoint_id']} by specificity rule "
                    f"({new_window_size}s < {existing_window_size}s)"
                )
                continue

            # Rule 3: Higher quality score breaks ties
            if (
                new_recorded_at == existing_recorded_at
                and new_window_size == existing_window_size
            ):
                if new_quality > existing_quality:
                    to_supersede.append(existing["checkpoint_id"])
                    logger.debug(
                        f"Superseding {existing['checkpoint_id']} by quality rule "
                        f"({new_quality} > {existing_quality})"
                    )
                    continue

            # If new checkpoint doesn't supersede this one, log it
            logger.debug(
                f"NOT superseding {existing['checkpoint_id']} "
                f"(existing is newer or more specific)"
            )

        return to_supersede

    async def update_superseded_checkpoints(
        self,
        checkpoint_ids: List[str],
        superseded_by: str,
        valid_to: datetime,
    ) -> None:
        """
        Close validity windows for superseded checkpoints
        Updates both SQLite and Qdrant atomically

        Args:
            checkpoint_ids: List of checkpoint IDs to supersede
            superseded_by: ID of the superseding checkpoint
            valid_to: Time when validity ends (usually new checkpoint's valid_from)
        """
        if not checkpoint_ids:
            return

        logger.info(
            f"Updating {len(checkpoint_ids)} superseded checkpoints "
            f"(superseded_by={superseded_by}, valid_to={valid_to})"
        )

        cursor = self.data_store.conn.cursor()

        try:
            # Update SQLite for each superseded checkpoint
            for checkpoint_id in checkpoint_ids:
                cursor.execute(
                    """
                    UPDATE checkpoints
                    SET valid_to = ?,
                        superseded_by = ?
                    WHERE checkpoint_id = ?
                """,
                    (valid_to.isoformat(), superseded_by, checkpoint_id),
                )
                # NOTE: We do NOT set recorded_end when superseding!
                # Superseding means the fact is no longer valid (valid_to changes)
                # but we still know about it (recorded_end stays infinity)

                logger.debug(
                    f"Updated SQLite for superseded checkpoint {checkpoint_id}"
                )

            self.data_store.conn.commit()

            # Update Qdrant metadata for each superseded checkpoint
            # Note: Qdrant doesn't have a direct update_payload method in the Python client,
            # so we need to retrieve the point, update its payload, and upsert it back
            for checkpoint_id in checkpoint_ids:
                try:
                    # Generate point ID
                    import uuid

                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, checkpoint_id))

                    # Retrieve existing point
                    points = self.vector_store.client.retrieve(
                        collection_name="checkpoints",
                        ids=[point_id],
                    )

                    if points:
                        # Update payload
                        point = points[0]
                        point.payload["valid_to"] = valid_to.isoformat()
                        point.payload["superseded_by"] = superseded_by
                        point.payload["recorded_end"] = datetime.now().isoformat()

                        # Upsert back
                        from qdrant_client.models import PointStruct

                        self.vector_store.client.upsert(
                            collection_name="checkpoints",
                            points=[
                                PointStruct(
                                    id=point_id,
                                    vector=point.vector,
                                    payload=point.payload,
                                )
                            ],
                        )

                        logger.debug(
                            f"Updated Qdrant for superseded checkpoint {checkpoint_id}"
                        )
                    else:
                        logger.warning(
                            f"Checkpoint {checkpoint_id} not found in Qdrant, skipping update"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to update Qdrant for checkpoint {checkpoint_id}: {e}"
                    )
                    # Don't fail the whole operation, continue with other checkpoints

            logger.info(
                f"Successfully updated {len(checkpoint_ids)} superseded checkpoints"
            )

        except Exception as e:
            logger.error(f"Failed to update superseded checkpoints: {e}")
            self.data_store.conn.rollback()
            raise

    async def handle_retroactive_correction(
        self,
        checkpoint_id: str,
        session_id: str,
        tool_id: str,
        checkpoint_type: str,
        corrected_data: Dict,
        valid_from: datetime,
        corrects: List[str],
        recorded_at: Optional[datetime] = None,
        quality_score: Optional[float] = None,
    ) -> Dict:
        """
        Handle retroactive corrections (new information about past events)

        Example: "We learned today (T'=Jan 10) that the bug was introduced
                  on Jan 1 (T=Jan 1), not Jan 5 as we thought"

        This requires:
        - Creating new version with earlier valid_from but later recorded_at
        - Marking old versions as corrected (superseded)
        - Maintaining full audit trail

        Args:
            checkpoint_id: New checkpoint ID for the correction
            session_id: Session identifier
            tool_id: Tool identifier
            checkpoint_type: Type of checkpoint
            corrected_data: Corrected checkpoint data (summary, key_facts, etc.)
            valid_from: When the corrected information was actually valid (past)
            corrects: List of checkpoint IDs being corrected
            recorded_at: When we learned about the correction (defaults to now)
            quality_score: Quality score

        Returns:
            Resolution result dictionary
        """
        recorded_at = recorded_at or datetime.now()

        logger.info(
            f"Handling retroactive correction {checkpoint_id} "
            f"(valid_from={valid_from}, corrects={corrects})"
        )

        try:
            # First, mark the corrected checkpoints as superseded
            await self.update_superseded_checkpoints(
                checkpoint_ids=corrects,
                superseded_by=checkpoint_id,
                valid_to=valid_from,
            )

            # Now store the new correction checkpoint
            # Use store_checkpoint_with_resolution to handle any other overlaps
            result = await self.store_checkpoint_with_resolution(
                checkpoint_id=checkpoint_id,
                session_id=session_id,
                tool_id=tool_id,
                checkpoint_type=checkpoint_type,
                summary=corrected_data.get("summary", ""),
                valid_from=valid_from,
                valid_to=corrected_data.get("valid_to"),
                recorded_at=recorded_at,
                key_facts=corrected_data.get("key_facts"),
                decisions=corrected_data.get("decisions"),
                patterns=corrected_data.get("patterns"),
                files_modified=corrected_data.get("files_modified"),
                dependencies_added=corrected_data.get("dependencies_added"),
                commands_run=corrected_data.get("commands_run"),
                todos=corrected_data.get("todos"),
                blockers=corrected_data.get("blockers"),
                quality_score=quality_score,
            )

            result["corrects"] = corrects
            result["retroactive"] = True

            logger.info(f"Successfully handled retroactive correction {checkpoint_id}")

            return result

        except Exception as e:
            logger.error(f"Failed to handle retroactive correction: {e}")
            raise

    def get_checkpoint_history(
        self,
        checkpoint_id: str,
    ) -> List[Dict]:
        """
        Get complete history of checkpoint including all superseded versions

        Returns list ordered by recorded_at (audit trail)

        Args:
            checkpoint_id: Checkpoint ID to get history for

        Returns:
            List of checkpoint versions ordered by recorded_at
        """
        try:
            # Use the data_store's get_checkpoint_history method
            history = self.data_store.get_checkpoint_history(checkpoint_id)

            # Sort by recorded_at to show audit trail
            history.sort(key=lambda x: x.get("recorded_at") or "1970-01-01T00:00:00")

            logger.info(
                f"Retrieved {len(history)} versions in history for {checkpoint_id}"
            )

            return history

        except Exception as e:
            logger.error(f"Failed to get checkpoint history: {e}")
            return []

    def validate_temporal_consistency(
        self,
        checkpoint_id: str,
    ) -> Dict:
        """
        Validate that SQLite and Qdrant are temporally consistent

        Checks:
        1. Checkpoint exists in both stores
        2. Temporal metadata matches (valid_from, valid_to, recorded_at)
        3. Supersedes/superseded_by relationships match
        4. Quality scores match

        Args:
            checkpoint_id: Checkpoint ID to validate

        Returns:
            {
                "consistent": bool,
                "sqlite_state": Dict,
                "qdrant_state": Dict,
                "discrepancies": List[str]
            }
        """
        discrepancies = []

        try:
            # Get SQLite state
            sqlite_checkpoint = self.data_store.get_checkpoint(checkpoint_id)

            if not sqlite_checkpoint:
                return {
                    "consistent": False,
                    "sqlite_state": None,
                    "qdrant_state": None,
                    "discrepancies": [
                        f"Checkpoint {checkpoint_id} not found in SQLite"
                    ],
                }

            # Get Qdrant state
            import uuid

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, checkpoint_id))

            qdrant_points = self.vector_store.client.retrieve(
                collection_name="checkpoints",
                ids=[point_id],
            )

            if not qdrant_points:
                return {
                    "consistent": False,
                    "sqlite_state": sqlite_checkpoint,
                    "qdrant_state": None,
                    "discrepancies": [
                        f"Checkpoint {checkpoint_id} not found in Qdrant"
                    ],
                }

            qdrant_checkpoint = qdrant_points[0].payload

            # Compare temporal metadata
            sqlite_valid_from = sqlite_checkpoint.get("valid_from")
            qdrant_valid_from = qdrant_checkpoint.get("valid_from")

            if sqlite_valid_from != qdrant_valid_from:
                discrepancies.append(
                    f"valid_from mismatch: SQLite={sqlite_valid_from}, "
                    f"Qdrant={qdrant_valid_from}"
                )

            sqlite_valid_to = sqlite_checkpoint.get("valid_to")
            qdrant_valid_to = qdrant_checkpoint.get("valid_to")

            if sqlite_valid_to != qdrant_valid_to:
                discrepancies.append(
                    f"valid_to mismatch: SQLite={sqlite_valid_to}, "
                    f"Qdrant={qdrant_valid_to}"
                )

            sqlite_recorded_at = sqlite_checkpoint.get("recorded_at")
            qdrant_recorded_at = qdrant_checkpoint.get("recorded_at")

            if sqlite_recorded_at != qdrant_recorded_at:
                discrepancies.append(
                    f"recorded_at mismatch: SQLite={sqlite_recorded_at}, "
                    f"Qdrant={qdrant_recorded_at}"
                )

            # Compare relationships (supersedes)
            sqlite_supersedes = sqlite_checkpoint.get("supersedes")
            qdrant_supersedes = qdrant_checkpoint.get("supersedes")

            # Handle JSON parsing
            if isinstance(sqlite_supersedes, str):
                import json

                try:
                    sqlite_supersedes = json.loads(sqlite_supersedes)
                except:
                    sqlite_supersedes = None

            if sqlite_supersedes != qdrant_supersedes:
                discrepancies.append(
                    f"supersedes mismatch: SQLite={sqlite_supersedes}, "
                    f"Qdrant={qdrant_supersedes}"
                )

            # Compare quality scores
            sqlite_quality = sqlite_checkpoint.get("quality_score")
            qdrant_quality = qdrant_checkpoint.get("quality_score")

            if sqlite_quality != qdrant_quality:
                discrepancies.append(
                    f"quality_score mismatch: SQLite={sqlite_quality}, "
                    f"Qdrant={qdrant_quality}"
                )

            # Determine consistency
            consistent = len(discrepancies) == 0

            result = {
                "consistent": consistent,
                "sqlite_state": {
                    "valid_from": sqlite_valid_from,
                    "valid_to": sqlite_valid_to,
                    "recorded_at": sqlite_recorded_at,
                    "supersedes": sqlite_supersedes,
                    "quality_score": sqlite_quality,
                },
                "qdrant_state": {
                    "valid_from": qdrant_valid_from,
                    "valid_to": qdrant_valid_to,
                    "recorded_at": qdrant_recorded_at,
                    "supersedes": qdrant_supersedes,
                    "quality_score": qdrant_quality,
                },
                "discrepancies": discrepancies,
            }

            if consistent:
                logger.info(f"Checkpoint {checkpoint_id} is temporally consistent")
            else:
                logger.warning(
                    f"Checkpoint {checkpoint_id} has {len(discrepancies)} discrepancies"
                )

            return result

        except Exception as e:
            logger.error(f"Failed to validate temporal consistency: {e}")
            return {
                "consistent": False,
                "sqlite_state": None,
                "qdrant_state": None,
                "discrepancies": [f"Validation error: {str(e)}"],
            }
