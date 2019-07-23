from distutils.core import setup

setup(
    name='ZakPy',
    version='0.1dev',
    packages=['zakpy',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=[
        "pandas",
        "sklearn",
        "numpy",
        "matplotlib",
        "boto3"
    ],
)