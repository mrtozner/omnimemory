"""
Example usage of WitnessSelector
Demonstrates how to select representative code snippets from files
"""

import asyncio
from witness_selector import WitnessSelector
from pathlib import Path


async def analyze_file(file_path: str):
    """Analyze a single file and display witnesses."""
    selector = WitnessSelector()
    await selector.initialize()

    # Read file content
    content = Path(file_path).read_text()

    print(f"\n{'='*70}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*70}\n")

    # Select witnesses
    witnesses = await selector.select_witnesses(
        content, max_witnesses=5, lambda_param=0.7  # 70% relevance, 30% diversity
    )

    if not witnesses:
        print("No witnesses found (file may be empty or only comments)")
        return

    print(f"Selected {len(witnesses)} witnesses:\n")

    for i, w in enumerate(witnesses, 1):
        print(f"{i}. {w['type'].upper().replace('_', ' ')}")
        print(f"   Line {w['line']} | Relevance Score: {w['score']:.3f}")
        print(f"   {w['text'][:80]}...")
        print()

    return witnesses


async def compare_lambda_params(file_path: str):
    """
    Compare different lambda parameters to show the effect on diversity.

    Lambda controls the balance:
    - High λ (0.9): More relevance, less diversity
    - Medium λ (0.7): Balanced (default)
    - Low λ (0.5): More diversity, less relevance
    """
    selector = WitnessSelector()
    await selector.initialize()

    content = Path(file_path).read_text()

    print(f"\n{'='*70}")
    print(f"Lambda Parameter Comparison: {file_path}")
    print(f"{'='*70}\n")

    for lambda_val in [0.9, 0.7, 0.5]:
        witnesses = await selector.select_witnesses(
            content, max_witnesses=5, lambda_param=lambda_val
        )

        print(
            f"λ = {lambda_val} ({'High Relevance' if lambda_val >= 0.8 else 'High Diversity' if lambda_val <= 0.6 else 'Balanced'})"
        )
        print("-" * 70)

        for w in witnesses:
            print(f"  {w['type']:20s} | Score: {w['score']:.3f} | {w['text'][:40]}...")

        print()


async def batch_analyze(directory: str):
    """Analyze multiple files in a directory."""
    selector = WitnessSelector()
    await selector.initialize()

    print(f"\n{'='*70}")
    print(f"Batch Analysis: {directory}")
    print(f"{'='*70}\n")

    # Find Python files
    files = list(Path(directory).glob("*.py"))

    for file_path in files[:5]:  # Limit to 5 files for demo
        try:
            content = file_path.read_text()
            witnesses = await selector.select_witnesses(content, max_witnesses=3)

            if witnesses:
                print(f"\n{file_path.name}:")
                for w in witnesses:
                    print(f"  {w['type']:20s} | {w['text'][:50]}...")
        except Exception as e:
            print(f"\n{file_path.name}: Error - {e}")


async def main():
    """Run example demonstrations."""
    print("\n" + "=" * 70)
    print("Witness Selector - Usage Examples")
    print("=" * 70)

    # Example 1: Analyze the witness selector itself
    await analyze_file("witness_selector.py")

    # Example 2: Compare lambda parameters
    await compare_lambda_params("witness_selector.py")

    # Example 3: Analyze test file
    await analyze_file("test_witness_selector.py")

    # Example 4: Batch analysis (if in a directory with Python files)
    # await batch_analyze(".")

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
