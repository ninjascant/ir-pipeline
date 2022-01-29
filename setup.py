from setuptools import setup, find_packages

setup(
    name="ir-pipeline",
    package_dir={"": "src"},
    packages=find_packages('src'),
    install_requires=[
        "click ==^8.0.3",
        "torch==^1.10.1",
        "transformers==^4.15.0"
        "datasets==^1.17.0"
        "faiss-cpu==^1.7.2"
        "tqdm==^4.62.3"
        "loguru==^0.5.3"
    ],
    entry_points={
        'console_scripts': [
            'irpipeline=irpipeline.cli:run',
        ],
    }
)
