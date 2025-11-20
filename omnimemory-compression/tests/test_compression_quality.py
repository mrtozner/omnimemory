#!/usr/bin/env python3
"""
Compression Quality Validation Tests

Tests if compressed content preserves enough context to be useful for real tasks:
1. Question-answering accuracy
2. Key fact retention
3. Code understanding tasks
4. Task completion with compressed context
"""

import asyncio
import httpx
import json


class CompressionQualityValidator:
    """Validates compression quality through real-world tasks"""

    def __init__(self, compression_service_url="http://localhost:8001"):
        self.compression_url = compression_service_url

    async def test_question_answering(
        self, original_text: str, questions: list
    ) -> dict:
        """
        Test if compressed version can answer questions correctly

        Args:
            original_text: Full text
            questions: List of {"question": str, "answer": str} dicts

        Returns:
            {"accuracy": float, "results": [...]}
        """
        # Compress the text
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.compression_url}/compress",
                json={
                    "context": original_text,
                    "tool_id": "test",
                    "session_id": "quality-test",
                },
            )
            compression_data = response.json()

        compressed_text = compression_data["compressed_text"]

        print(f"\n{'='*80}")
        print(f"QUESTION-ANSWERING QUALITY TEST")
        print(f"{'='*80}")
        print(f"Original tokens:    {compression_data['original_tokens']}")
        print(f"Compressed tokens:  {compression_data['compressed_tokens']}")
        print(f"Compression ratio:  {compression_data['compression_ratio']:.1%}")
        print(f"Quality score:      {compression_data['quality_score']:.1%}")
        print()

        # Test each question
        results = []
        for i, qa in enumerate(questions, 1):
            question = qa["question"]
            expected_keywords = qa.get("keywords", [])

            # Check if compressed text contains answer keywords
            found_keywords = [
                kw for kw in expected_keywords if kw.lower() in compressed_text.lower()
            ]
            contains_answer = len(found_keywords) > 0

            results.append(
                {
                    "question": question,
                    "expected_keywords": expected_keywords,
                    "found_keywords": found_keywords,
                    "passed": contains_answer,
                }
            )

            status = "✅ PASS" if contains_answer else "❌ FAIL"
            print(f"{i}. {question}")
            print(f"   Expected keywords: {expected_keywords}")
            print(f"   Found keywords: {found_keywords}")
            print(f"   {status}")
            print()

        accuracy = sum(1 for r in results if r["passed"]) / len(results)

        print(f"{'='*80}")
        print(
            f"OVERALL ACCURACY: {accuracy:.1%} ({sum(1 for r in results if r['passed'])}/{len(results)} passed)"
        )
        print(f"{'='*80}\n")

        return {
            "accuracy": accuracy,
            "results": results,
            "compression_ratio": compression_data["compression_ratio"],
            "quality_score": compression_data["quality_score"],
        }

    async def test_key_fact_retention(
        self, original_text: str, key_facts: list
    ) -> dict:
        """
        Test if key facts are preserved in compressed version

        Args:
            original_text: Full text
            key_facts: List of essential facts/concepts that must be retained

        Returns:
            {"retention_rate": float, "results": [...]}
        """
        # Compress the text
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.compression_url}/compress",
                json={
                    "context": original_text,
                    "tool_id": "test",
                    "session_id": "quality-test",
                },
            )
            compression_data = response.json()

        compressed_text = compression_data["compressed_text"]

        print(f"\n{'='*80}")
        print(f"KEY FACT RETENTION TEST")
        print(f"{'='*80}")

        results = []
        for i, fact in enumerate(key_facts, 1):
            # Check if fact is in compressed text
            retained = fact.lower() in compressed_text.lower()
            results.append({"fact": fact, "retained": retained})

            status = "✅ RETAINED" if retained else "❌ LOST"
            print(f"{i}. {fact}: {status}")

        retention_rate = sum(1 for r in results if r["retained"]) / len(results)

        print(f"\nRETENTION RATE: {retention_rate:.1%}\n")

        return {
            "retention_rate": retention_rate,
            "results": results,
            "compression_ratio": compression_data["compression_ratio"],
            "quality_score": compression_data["quality_score"],
        }

    async def test_code_understanding(self, code: str, questions: list) -> dict:
        """
        Test if compressed code preserves enough context for understanding

        Args:
            code: Source code to compress
            questions: Questions about the code structure/functionality

        Returns:
            {"accuracy": float, "results": [...]}
        """
        return await self.test_question_answering(code, questions)

    def print_comparison(self, original: str, compressed: str):
        """Print side-by-side comparison of original vs compressed"""
        print(f"\n{'='*80}")
        print(f"CONTENT COMPARISON")
        print(f"{'='*80}")
        print(f"\nORIGINAL ({len(original)} chars):")
        print("-" * 80)
        print(original[:500])
        if len(original) > 500:
            print(f"... ({len(original) - 500} more chars)")
        print()

        print(f"COMPRESSED ({len(compressed)} chars):")
        print("-" * 80)
        print(compressed[:500])
        if len(compressed) > 500:
            print(f"... ({len(compressed) - 500} more chars)")
        print()


async def main():
    """Run quality validation tests"""
    validator = CompressionQualityValidator()

    # Test 1: Python code comprehension
    python_code = """
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
import time


class WebScraper:
    '''A simple web scraper for extracting data from websites'''

    def __init__(self, base_url: str):
        '''Initialize the scraper with a base URL'''
        self.base_url = base_url
        self.session = requests.Session()
        self.visited = set()

    def fetch_page(self, url: str) -> str:
        '''Fetch a page from the website'''
        response = self.session.get(url)
        response.raise_for_status()
        return response.text

    def parse_data(self, html: str, selectors: Dict[str, str]) -> Dict:
        '''Parse data from HTML using CSS selectors'''
        soup = BeautifulSoup(html, 'html.parser')
        data = {}

        for key, selector in selectors.items():
            elements = soup.select(selector)
            data[key] = [elem.text.strip() for elem in elements]

        return data

    def scrape_data(self, selectors: Dict[str, str]) -> Dict:
        '''Main method to scrape data from the base URL'''
        html = self.fetch_page(self.base_url)
        return self.parse_data(html, selectors)
"""

    code_questions = [
        {
            "question": "What library is used for parsing HTML?",
            "keywords": ["BeautifulSoup", "bs4"],
        },
        {
            "question": "What does the fetch_page method do?",
            "keywords": ["fetch", "page", "requests"],
        },
        {
            "question": "What data structure stores visited pages?",
            "keywords": ["visited", "set"],
        },
        {
            "question": "What is the main scraping method?",
            "keywords": ["scrape_data"],
        },
    ]

    print("\n" + "=" * 80)
    print("COMPRESSION QUALITY VALIDATION SUITE")
    print("=" * 80)

    result1 = await validator.test_code_understanding(python_code, code_questions)

    # Test 2: Key fact retention
    documentation = """
The VisionDrop compression algorithm uses semantic chunking and embedding-based
importance scoring to achieve 94.4% token reduction while maintaining 91% quality.

Key features:
- Smart chunking that respects code and command boundaries
- Query-aware filtering using cosine similarity
- Token-aware adaptive thresholding for accurate compression ratios
- Quality assessment via centroid comparison
- Integration with enterprise tokenization via OmniTokenizer
- Three-tier caching for performance optimization

The algorithm works by:
1. Splitting text into semantic chunks
2. Computing embeddings for all chunks
3. Calculating importance scores (query-aware or self-attention)
4. Selecting chunks based on token-weighted threshold
5. Reconstructing compressed text from retained chunks
"""

    key_facts = [
        "94.4% token reduction",
        "semantic chunking",
        "query-aware filtering",
        "token-aware adaptive thresholding",
        "OmniTokenizer",
    ]

    result2 = await validator.test_key_fact_retention(documentation, key_facts)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Code Understanding Test:    {result1['accuracy']:.1%} accuracy")
    print(f"Key Fact Retention Test:    {result2['retention_rate']:.1%} retention")
    print()
    print("INTERPRETATION:")
    if result1["accuracy"] >= 0.75 and result2["retention_rate"] >= 0.8:
        print("✅ EXCELLENT - Compression preserves context for real-world use")
    elif result1["accuracy"] >= 0.5 and result2["retention_rate"] >= 0.6:
        print("⚠️  ACCEPTABLE - Some context loss, may affect complex tasks")
    else:
        print("❌ POOR - Significant context loss, not suitable for production")
    print()


if __name__ == "__main__":
    asyncio.run(main())
