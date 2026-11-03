from setuptools import setup, find_packages

setup(
    name="claris",
    version="0.1.0",
    description="Framework CLARIS: Arquitetura Híbrida de Explicabilidade e Monitoramento de Drift",
    author="Vivian Corrêa Silva",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if not line.startswith("ragas") and not line.startswith("pytest") and not line.startswith("datasets")
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "ragas>=0.1.0", "datasets>=2.14.0"]
    }
)