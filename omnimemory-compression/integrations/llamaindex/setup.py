"""
Setup script for OmniMemory LlamaIndex integration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="omnimemory-llamaindex",
    version="0.1.0",
    description="OmniMemory integration for LlamaIndex",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OmniMemory Team",
    author_email="support@omnimemory.ai",
    url="https://github.com/omnimemory/omnimemory-compression",
    packages=find_packages(),
    install_requires=[
        "llama-index-core>=0.10.0",
        "omnimemory>=0.1.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="compression llm context tokens ai llamaindex",
)
