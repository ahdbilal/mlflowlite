from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlflowlite",
    version="0.1.0",
    author="mlflowlite",
    description="Easy LLM observability with automatic MLflow tracing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "mlflow>=2.10.0",
        "litellm>=1.30.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "openai>=1.12.0",
        "anthropic>=0.18.0",
        "google-generativeai>=0.3.0",
        "dspy-ai>=2.4.0",
        "requests>=2.31.0",
        "tenacity>=8.2.0",
        "tiktoken>=0.5.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "mypy>=1.0.0"],
    },
)

