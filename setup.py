from setuptools import setup, find_packages


PACKAGENAME = "gordon"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author=["Matt Becker", "Andrew Hearin"],
    author_email=["", "ahearin@anl.gov"],
    description="jax+numba demo",
    long_description="jax+numba demo",
    install_requires=["numpy", "numba", "jax"],
    packages=find_packages(),
    url="https://github.com/ArgonneCPAC/gordon",
)
