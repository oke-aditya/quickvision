# Adopted from PyTorch Lightning Bolts.
# !/usr/bin/env python

import os

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/

PATH_ROOT = os.path.dirname(__file__)

# import vision  # noqa: E402


def load_requirements(path_dir=PATH_ROOT, file_name='requirements.txt', comment_char='#'):
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if comment_char in ln:  # filer all comments
            ln = ln[:ln.index(comment_char)].strip()
        if ln.startswith('http'):  # skip directly installed dependencies
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def load_long_description():
    # url = os.path.join("https://github.com/Quick-AI/quickvision", 'raw', , 'docs')
    text = open('README.md', encoding='utf-8').read()
    # replace relative repository path to absolute link to the release
    # text = text.replace('](docs', f']({url}')
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace('.svg', '.png')
    return text


# https://packaging.python.org/discussions/install-requires-vs-requirements /
setup(
    name='quickvision',
    version="0.1.0",
    description="Computer Vision models and training",
    author="Aditya Oke",
    author_email="okeaditya315@gmail.com",
    # url=pl_bolts.__homepage__,
    download_url="https://github.com/Quick-AI/quickvision",
    license="apache2",
    packages=find_packages(exclude=['tests', 'docs']),

    long_description=load_long_description(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,

    keywords=['Deep Learning', 'PyTorch'],
    python_requires='>=3.6',
    setup_requires=[],
    install_requires=load_requirements(),

    project_urls={
        "Bug Tracker": "https://github.com/Quick-AI/quickvision/issues",
        "Documentation": "https://quick-ai.github.io/quickvision/",
        "Source Code": "https://github.com/Quick-AI/quickvision",
    },

    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
