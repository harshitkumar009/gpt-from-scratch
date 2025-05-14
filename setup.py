import os
import re
from setuptools import setup, find_packages

def get_version():
    with open(os.path.join("gpt","__init__.py"), "r") as f:
        match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', f.read())
        return match.group(1)

setup(
    name='gpt-model',
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "jupyterlab>=4.0",
        "numpy",
        "tiktoken>=0.5.1",
        "matplotlib>=3.7.1",
        "tqdm>=4.66.1",
        "pandas>=2.2.1",
        "psutil>=5.9.5",
        "transformers"
    ],
    author='Harshith Kumar',
    author_email='chiluveruharshit@gmail.com',
    description='A collection of utility functions for gpt projects',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/harshitkumar009/gpt-from-scratch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
