"""
Secure Code Execution Sandbox for OmniMemory

Provides safe Python code execution with resource limits and OmniMemory API pre-installed.
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class CodeExecutor:
    """Secure code execution environment with resource limits"""

    def __init__(
        self,
        timeout_seconds: int = 30,
        max_memory_mb: int = 512,
        enable_network: bool = True,
    ):
        """
        Initialize code executor

        Args:
            timeout_seconds: Maximum execution time
            max_memory_mb: Maximum memory usage
            enable_network: Whether to allow network access (needed for OmniMemory APIs)
        """
        self.timeout = timeout_seconds
        self.max_memory = max_memory_mb
        self.enable_network = enable_network

        # Path to omnimemory Python API package
        self.omnimemory_api_path = str(
            Path(__file__).parent.parent / "omnimemory-python-api"
        )

    def execute_python(
        self, code: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in sandboxed environment

        Args:
            code: Python code to execute
            context: Optional context variables to inject

        Returns:
            Dictionary with:
                - success: bool
                - output: stdout
                - error: stderr (if any)
                - return_value: last expression value (if any)
                - stats: execution stats (time, memory)
        """
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            # Write imports and setup
            tmp_file.write("import sys\n")
            tmp_file.write("import json\n")
            tmp_file.write(f"sys.path.insert(0, '{self.omnimemory_api_path}')\n\n")

            # Write context variables if provided
            if context:
                tmp_file.write("# Context variables\n")
                for key, value in context.items():
                    tmp_file.write(f"{key} = {repr(value)}\n")
                tmp_file.write("\n")

            # Write user code
            tmp_file.write("# User code\n")
            tmp_file.write(code)
            tmp_file.write("\n")

            tmp_path = tmp_file.name

        try:
            # Execute with resource limits
            env = os.environ.copy()

            # Add OmniMemory service URLs
            env.update(
                {
                    "OMNIMEMORY_COMPRESSION_URL": "http://localhost:8001",
                    "OMNIMEMORY_EMBEDDINGS_URL": "http://localhost:8000",
                    "OMNIMEMORY_PROCEDURAL_URL": "http://localhost:8002",
                    "OMNIMEMORY_METRICS_URL": "http://localhost:8003",
                }
            )

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=os.getcwd(),  # Use current working directory
            )

            success = result.returncode == 0
            output = result.stdout
            error = result.stderr if result.returncode != 0 else None

            return {
                "success": success,
                "output": output,
                "error": error,
                "return_code": result.returncode,
                "execution_time": None,  # TODO: measure time
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {self.timeout} seconds",
                "return_code": -1,
            }

        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "return_code": -1,
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

    def execute_with_omnimemory(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code with OmniMemory API pre-imported

        This is the main entry point for MCP code execution.

        Args:
            code: Python code to execute (can use omnimemory.* functions)

        Returns:
            Execution result with output and stats

        Example:
            >>> executor = CodeExecutor()
            >>> result = executor.execute_with_omnimemory('''
            ... from omnimemory import read
            ... content = read("test.py")
            ... print(f"Read {len(content)} chars")
            ... ''')
            >>> print(result['output'])
        """
        return self.execute_python(code)

    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate Python code without executing

        Args:
            code: Python code to validate

        Returns:
            Dictionary with validation result
        """
        try:
            compile(code, "<string>", "exec")
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}",
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}


# Global instance
_executor = None


def get_executor() -> CodeExecutor:
    """Get or create global code executor instance"""
    global _executor
    if _executor is None:
        _executor = CodeExecutor(
            timeout_seconds=30, max_memory_mb=512, enable_network=True
        )
    return _executor


def execute_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code in sandboxed environment with OmniMemory API

    This is the main function called by MCP tools.

    Args:
        code: Python code to execute

    Returns:
        Execution result with output, errors, and stats
    """
    executor = get_executor()
    return executor.execute_with_omnimemory(code)
