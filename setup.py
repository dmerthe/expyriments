import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="empyric-dmerthe",
    version="0.1",
    author="Daniel Merthe",
    author_email="dmerthe@gmail.com",
    description="A package for experiment automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dmerthe/empyric",
    packages=setuptools.find_packages(),
    package_data={'': ['*.yaml']},
    data_files=[
        ('tests', [
            os.path.join(
                'examples',
                'Henon Map Experiment',
                'henon_runcard_example.yaml'
            )
        ])
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib>=3.6',
        'pandas',
        'pykwalify',
        'ruamel.yaml',
        'pytest'
    ],
    entry_points={'console_scripts': ['empyric = empyric:execute', ]}
)
