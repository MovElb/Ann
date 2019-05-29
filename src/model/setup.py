from setuptools import setup, find_packages

setup(
    name='bertynet',
    version='0.0.1',
    python_requires='>=3.6',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['bertynet = bertynet.app:main'],
    },
    namespace_packages=['bertynet'],
)
