"""Quick smoke test for witness selector"""
import asyncio
from witness_selector import WitnessSelector


async def main():
    s = WitnessSelector()
    await s.initialize()
    w = await s.select_witnesses(
        "def foo():\n    pass\n\nclass Bar:\n    pass\n\ndef baz():\n    pass",
        max_witnesses=3,
    )
    print(f"✓ Quick test: {len(w)} witnesses selected")
    for witness in w:
        print(f"  - {witness['type']}: {witness['text']}")


asyncio.run(main())
