"""
Setup script for LeanAgent.
"""

from setuptools import setup, find_packages

setup(
    name="leanagent",
    version="0.1.0",
    description="LeanAgent - Lifelong Learning for Formal Theorem Proving",
    author="LeanAgent Team",
    packages=find_packages(),
    python_requires=">=3.8,<4.0",
    install_requires=[
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "leanagent=leanagent.cli:main",
        ],
    },
) 