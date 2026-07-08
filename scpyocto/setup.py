from setuptools import setup, find_packages

setup(
    name='scpyocto',
    version='0.1.0',
    description='PyOcto associator as a SeisComP module',
    package_dir={'': 'lib'},
    packages=find_packages(where='lib'),
    install_requires=['pyocto', 'pandas', 'pyproj'],
)
