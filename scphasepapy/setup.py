from setuptools import setup, find_packages

setup(
    name='scphasepapy',
    version='0.1.0',
    description='SeisComP PhasePaPy associator module',
    packages=find_packages(where='lib'),
    package_dir={'': 'lib'},
    install_requires=[
        'PyYAML',
        'watchdog',
        'obspy',
        'scipy',
        'numpy',
        'SQLAlchemy',
    ],
    python_requires='>=3.7',
)
