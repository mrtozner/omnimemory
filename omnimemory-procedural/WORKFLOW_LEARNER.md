# Autonomous Workflow Learning Service

The Workflow Learner is a background service that automatically learns workflow patterns from user activity without requiring any manual intervention or MCP tool calls.

## Overview

Similar to the checkpoint monitor, the workflow learner runs autonomously in the background, monitoring the memory daemon database for user activity and automatically learning patterns.

### Key Features

- **Autonomous Operation**: Runs in the background, no manual intervention required
- **Session Detection**: Automatically groups events into logical workflow sessions
- **Pattern Learning**: Learns patterns when sessions become idle (5 minutes)
- **Event Monitoring**: Polls memory daemon database every 30 seconds
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Error Resilience**: Continues running despite errors

## Architecture

### Data Flow

```
Memory Daemon Database
       ↓
Workflow Learner (polls every 30s)
       ↓
Session Detection & Grouping
       ↓
Pattern Learning (on session idle)
       ↓
Procedural Memory API
       ↓
Pattern Storage
```

### Event Types Monitored

The workflow learner monitors these event types from the memory daemon:

| Event Type | Command | Description |
|------------|---------|-------------|
| `file_open` | `open_file` | File opened in editor |
| `file_save` | `save_file` | File saved |
| `file_create` | `create_file` | New file created |
| `file_delete` | `delete_file` | File deleted |
| `file_modify` | `modify_file` | File modified |
| `process_start` | `execute_command` | Command executed |
| `process_end` | `command_completed` | Command finished |
| `editor_open` | `open_editor` | Editor opened |
| `editor_close` | `close_editor` | Editor closed |
| `code_edit` | `edit_code` | Code edited |
| `test_run` | `run_tests` | Tests executed |
| `build_start` | `build_project` | Build started |
| `git_commit` | `commit_changes` | Git commit |
| `git_push` | `push_changes` | Git push |
| `git_pull` | `pull_changes` | Git pull |

### Session Management

**Session Creation**:
- New session created for each unique `session_id` from memory daemon
- Sessions track commands, timestamps, and last activity

**Session Completion**:
- Sessions become idle after 5 minutes of no activity
- Idle sessions are processed for pattern learning
- Sessions with <3 commands are skipped

**Pattern Learning**:
- Calls procedural API `/learn` endpoint
- Sends command sequence with timestamps
- Infers outcome (success/failure) from command patterns

## Configuration

Default configuration in `workflow_learner.py`:

```python
SESSION_IDLE_MINUTES = 5  # Session idle threshold
POLL_INTERVAL_SECONDS = 30  # Database poll interval
MIN_COMMANDS_FOR_LEARNING = 3  # Minimum commands to learn
PROCEDURAL_API_URL = "http://localhost:8002"  # Procedural API
MEMORY_DAEMON_DB_PATH = "~/.memory-daemon/storage/memory_data.db"
```

## Usage

### Start via Launcher

The workflow learner is integrated into the OmniMemory launcher:

```bash
# Start all services (includes workflow learner)
./omnimemory_launcher.sh start

# Check status
./omnimemory_launcher.sh status

# View logs
tail -f ~/.omnimemory/logs/workflow-learner.log

# Stop services
./omnimemory_launcher.sh stop
```

### Start Standalone

For testing or development:

```bash
cd omnimemory-procedural/src
python3 workflow_learner.py
```

### View Logs

Logs are written to:
- Console (stdout)
- `~/.omnimemory/workflow_learner.log`

Log levels:
- **INFO**: Service status, sessions processed, patterns learned
- **DEBUG**: Event details, session updates
- **WARNING**: API issues, database not found
- **ERROR**: Processing errors, API failures

## Monitoring

### Statistics

The workflow learner tracks these statistics:

```python
{
    "running": bool,              # Service running status
    "active_sessions": int,        # Number of active sessions
    "stats": {
        "events_processed": int,   # Total events processed
        "sessions_created": int,   # Total sessions created
        "patterns_learned": int,   # Total patterns learned
        "errors": int,            # Total errors encountered
    }
}
```

### Log Examples

**Service Start**:
```
2025-11-08 04:37:01 - INFO - ============================================================
2025-11-08 04:37:01 - INFO - OmniMemory Autonomous Workflow Learner
2025-11-08 04:37:01 - INFO - ============================================================
2025-11-08 04:37:01 - INFO - Memory daemon DB: ~/.memory-daemon/storage/memory_data.db
2025-11-08 04:37:01 - INFO - Procedural API: http://localhost:8002
2025-11-08 04:37:01 - INFO - Session idle threshold: 5 minutes
2025-11-08 04:37:01 - INFO - Poll interval: 30 seconds
```

**Pattern Learning**:
```
2025-11-08 05:00:00 - INFO - Learning from session abc123: 5 commands, outcome=success
2025-11-08 05:00:01 - INFO - Successfully learned pattern xyz789 from session abc123 (5 commands)
```

**Error Handling**:
```
2025-11-08 05:05:00 - ERROR - Failed to connect to procedural API: Connection refused
2025-11-08 05:05:00 - INFO - Will retry on next poll cycle
```

## Integration Points

### Memory Daemon Database

**Schema Used**:
```sql
SELECT rowid, timestamp, event_type, session_id
FROM events
WHERE rowid > ?
ORDER BY rowid ASC
```

**Notes**:
- Events may be encrypted; workflow learner uses metadata only
- Uses `rowid` to track processing position
- Queries 1000 events per poll cycle maximum

### Procedural Memory API

**POST /learn**:
```json
{
  "session_commands": [
    {
      "command": "open_file",
      "timestamp": 1699456000.0,
      "context": {"event_type": "file_open"}
    },
    {
      "command": "edit_code",
      "timestamp": 1699456010.0,
      "context": {"event_type": "code_edit"}
    },
    {
      "command": "save_file",
      "timestamp": 1699456020.0,
      "context": {"event_type": "file_save"}
    }
  ],
  "session_outcome": "success"
}
```

**Response**:
```json
{
  "pattern_id": "abc123def456",
  "message": "Successfully learned pattern abc123def456"
}
```

## Outcome Inference

The workflow learner infers session outcomes based on command patterns:

**Success Indicators**:
- `commit_changes` - Code committed
- `push_changes` - Changes pushed
- `save_file` - Work saved
- `command_completed` - Command finished successfully

**Failure Indicators**:
- `delete_file` - Work deleted (potential failure)

**Default**: Sessions are assumed successful unless failure indicators are present.

## Testing

Run the test suite:

```bash
cd omnimemory-procedural
python3 test_workflow_learner.py
```

**Tests Include**:
1. Initialization - Verify service starts correctly
2. Event Type Mapping - Verify event types map to commands
3. Session Management - Verify sessions are created and tracked
4. Outcome Inference - Verify outcomes are inferred correctly
5. Statistics - Verify stats are tracked correctly

All tests should pass:
```
============================================================
Test Results
============================================================
Passed: 5/5
✓ All tests passed!
```

## Troubleshooting

### Service Not Learning Patterns

**Check**:
1. Memory daemon is running: `ps aux | grep memory_daemon`
2. Memory daemon database exists: `ls -la ~/.memory-daemon/storage/memory_data.db`
3. Procedural API is running: `curl http://localhost:8002/health`
4. Workflow learner logs: `tail -f ~/.omnimemory/logs/workflow-learner.log`

### Database Not Found

**Solution**:
- Wait for memory daemon to create database
- Workflow learner will wait and retry automatically
- Check memory daemon logs: `tail -f ~/.omnimemory/logs/daemon.log`

### Procedural API Unreachable

**Solution**:
- Start procedural service: `./omnimemory_launcher.sh start`
- Verify API health: `curl http://localhost:8002/health`
- Check procedural logs: `tail -f ~/.omnimemory/logs/procedural.log`

### Sessions Too Short

**Check**:
- Minimum 3 commands required per session
- Increase activity or decrease `MIN_COMMANDS_FOR_LEARNING`
- Check session activity in debug logs

## Performance

**Resource Usage**:
- Memory: ~20-50 MB (lightweight)
- CPU: Minimal (polls every 30 seconds)
- Disk: Logs rotate automatically
- Network: Minimal (API calls only when learning)

**Scalability**:
- Handles 1000s of events per poll cycle
- Sessions tracked in memory (lightweight)
- Database queries are indexed (fast)

## Future Enhancements

Potential improvements:

1. **Event Decryption**: Decrypt event data for richer command details
2. **Smart Outcome Inference**: Use ML to infer outcomes more accurately
3. **Pattern Quality Scoring**: Rate learned patterns by quality
4. **Real-time Learning**: Learn immediately instead of waiting for idle
5. **Custom Event Mapping**: User-configurable event type mappings
6. **Session Clustering**: Group similar sessions together
7. **Anomaly Detection**: Detect unusual workflow patterns

## Dependencies

- Python 3.8+
- `asyncio` - Asynchronous operations
- `sqlite3` - Database queries
- `httpx` - HTTP client for API calls
- `pathlib` - Path operations
- Memory daemon database (SQLite)
- Procedural memory API (port 8002)

## License

Part of the OmniMemory project.
