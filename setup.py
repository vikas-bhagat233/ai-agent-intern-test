from setuptools import setup, find_packages

setup(
    name="aster-row-support-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.12.0",
        "chromadb>=0.4.24",
        "langchain>=0.1.16",
        "langchain-community>=0.0.38",
        "python-dotenv>=1.0.1",
        "click>=8.1.7",
        "pytest>=8.1.1",
        "pytest-cov>=5.0.0",
        "tiktoken>=0.6.0",
        "pydantic>=2.6.4",
        "rich>=13.7.1",
        "python-frontmatter>=1.1.0"
    ],
    entry_points={
        "console_scripts": [
            "aster-support=src.cli:main",
        ],
    },
    python_requires=">=3.9",
)