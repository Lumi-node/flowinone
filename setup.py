from setuptools import setup, find_packages

setup(
    name="unified_vision",
    version="0.1.0",
    description="Unified visual representation system for multimodal generation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "transformers",
        "pillow",
        "numpy",
        "tqdm",
        "accelerate",
        "datasets",
        "sentencepiece",
        "clip-by-openai",
        "lpips"
    ],
    entry_points={
        "console_scripts": [
            "unified-vision-train=unified_vision.cli:train_cli",
            "unified-vision-generate=unified_vision.cli:generate_cli"
        ]
    }
)