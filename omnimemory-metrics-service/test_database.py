#!/usr/bin/env python3
"""
Test script for database models and operations
"""

import uuid
from datetime import datetime
from src.database import (
    get_db_context,
    ToolSession,
    ToolOperation,
    init_db,
    get_db_info,
)


def test_database():
    """Test database operations"""
    print("Testing database operations...")
    print()

    # Initialize database
    print("1. Initializing database...")
    init_db()
    print("   ✅ Database initialized")
    print()

    # Display database info
    print("2. Database configuration:")
    db_info = get_db_info()
    for key, value in db_info.items():
        print(f"   {key}: {value}")
    print()

    # Create a test session
    print("3. Creating test session...")
    with get_db_context() as db:
        session = ToolSession(
            session_id=uuid.uuid4(),
            tool_id="claude-code",
            tool_version="1.0.0",
            started_at=datetime.utcnow(),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        print(f"   ✅ Created session: {session.session_id}")
        print()

        # Create test operations
        print("4. Creating test operations...")
        operations = [
            ToolOperation(
                id=uuid.uuid4(),
                session_id=session.session_id,
                tool_name="read",
                operation_mode="full",
                parameters={"compress": False},
                file_path="/path/to/file1.py",
                tokens_original=5000,
                tokens_actual=5000,
                tokens_prevented=0,
                response_time_ms=150.5,
                tool_id="claude-code",
            ),
            ToolOperation(
                id=uuid.uuid4(),
                session_id=session.session_id,
                tool_name="read",
                operation_mode="overview",
                parameters={"compress": True},
                file_path="/path/to/file2.py",
                tokens_original=5000,
                tokens_actual=500,
                tokens_prevented=4500,
                response_time_ms=120.3,
                tool_id="claude-code",
            ),
            ToolOperation(
                id=uuid.uuid4(),
                session_id=session.session_id,
                tool_name="search",
                operation_mode="semantic",
                parameters={"query": "authentication", "limit": 5},
                file_path=None,
                tokens_original=50000,
                tokens_actual=5000,
                tokens_prevented=45000,
                response_time_ms=250.7,
                tool_id="claude-code",
            ),
        ]

        for op in operations:
            db.add(op)

        db.commit()
        print(f"   ✅ Created {len(operations)} operations")
        print()

        # Query operations
        print("5. Querying operations...")
        results = db.query(ToolOperation).filter_by(session_id=session.session_id).all()
        print(f"   Found {len(results)} operations:")
        for op in results:
            print(
                f"   - {op.tool_name}/{op.operation_mode}: {op.tokens_prevented} tokens prevented"
            )
        print()

        # Calculate total tokens prevented
        total_prevented = sum(op.tokens_prevented for op in results)
        print(f"6. Total tokens prevented: {total_prevented}")
        print(f"   Cost saved: ${total_prevented * 0.000015:.4f}")
        print()

        # Test relationship
        print("7. Testing session -> operations relationship...")
        session_with_ops = (
            db.query(ToolSession).filter_by(session_id=session.session_id).first()
        )
        print(f"   Session has {len(session_with_ops.operations)} operations")
        print()

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_database()
