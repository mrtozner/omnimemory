"""Python 3.8 compatibility patch for asyncio.to_thread"""
import asyncio
import functools

if not hasattr(asyncio, "to_thread"):
    # Add to_thread for Python 3.8 compatibility
    async def to_thread(func, *args, **kwargs):
        """Run function in thread pool executor (Python 3.8 backport)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

    asyncio.to_thread = to_thread
