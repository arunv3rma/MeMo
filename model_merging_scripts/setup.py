from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="model_merge",
    version="0.1.0",
    author="<anonymous>",
    author_email="<anonymous>",
    description="A tool for merging and averaging AI model weights in safetensors format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<repo-url>",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "safetensors",
        "torch",
        "tqdm",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "model_merge=model_merge.main:main",
        ],
    },
)
