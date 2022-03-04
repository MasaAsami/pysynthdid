import os
import sys

from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))

install_requires = [
    'pandas',
    'matplotlib',
    'numpy',
    'tqdm',
    'sklearn',
    'scipy',
    'toolz',
    'bayesian-optimization >= 1.1.0'
]
tests_require = [
    'pytest',
    'pytest-cov',
    'mock',
    'tox'
]
setup_requires = [
    'flake8',
    'isort'
]
extras_require = {
    'docs': [
        'ipython',
        'jupyter'
    ]
}

packages = ['synthdid']

_version = {}
_version_path = os.path.join(here, 'synthdid', '__version__.py')

with open(_version_path, 'r') as f:
    exec(f.read(), _version)

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name='pysynthdid',
    version=_version['__version__'],
    author='MasaAsami',
    author_email='m.asami.moj@gmail.com',
    url='https://github.com/MasaAsami/pysynthdid',
    description= "Python version of Synthetic difference in differences",
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    extras_require=extras_require,
    license='Apache License 2.0',
    keywords='causal-inference',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
    ],
    project_urls={
        'Source': 'https://github.com/MasaAsami/pysynthdid'
    },
    python_requires='>=3',
    test_suite='tests'