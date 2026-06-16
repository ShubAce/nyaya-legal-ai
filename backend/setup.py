from setuptools import setup, find_packages

setup(
    name="nyaya-legal-ai",
    version="1.0.0",
    description="Finetuned LLM system for Indian legal judgment parsing",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.46.0",
        "torch>=2.1.0",
        "peft>=0.11.0",
        "trl>=0.12.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.30.0",
        "datasets>=2.19.0",
        "langchain>=0.2.0",
        "langgraph>=0.1.0",
        "chromadb>=0.5.0",
        "sentence-transformers>=3.0.0",
        "fastapi>=0.111.0",
        "google-genai",
        "gradio>=4.36.0",
        "wandb>=0.17.0",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
    ],
)
