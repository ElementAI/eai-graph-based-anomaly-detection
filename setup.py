from setuptools import setup

with open("README.md", "r") as file_long_description:
    long_description = file_long_description.read()

setup(
    name='eai-graph-tools',
    version='0.0.1',
    packages=['eai_graph_tools'],
    data_files=[('.', ['LICENSE'])],
    description='Graph embeddings tools and experiments.',
    long_description=long_description,
    url="https://github.com/elementai/eai-graph-based-anomaly-detection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2 License",
        "Operating System :: OS Independent"
    ]
)
