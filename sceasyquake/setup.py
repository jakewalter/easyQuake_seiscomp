from setuptools import setup, find_packages

setup(
    name='sceasyquake',
    version='0.1.0',
    description='SeisComP ML-based phase picker module (easyQuake / PhaseNet)',
    packages=find_packages(where='lib'),
    package_dir={'': 'lib'},
    # The bin/ scripts are installed directly by install.sh into
    # $SEISCOMP_ROOT/bin/ with the correct seiscomp-python shebang.
    # They are also usable as standalone scripts via:
    #   python3 bin/sceasyquake-stream.py   (standalone mode)
    install_requires=[
        'PyYAML',
        'watchdog',
        'obspy',
        'scipy',
        'numpy',
    ],
    python_requires='>=3.7',
)
