from setuptools import setup

setup(
    name = "npgamma",
    version = "0.1",
    author = "Simon Biggs",
    author_email = "mail@simonbiggs.net",
    description = "Perform a gamma evaluation comparing two arbitrarily sized dose grids",
    keywords = "radiotherapy, gamma evaluation, distance to agreement",
    url = "https://github.com/SimonBiggs/npgamma/",
    dowload_url = "https://github.com/SimonBiggs/npgamma/tarball/0.1",
    packages = ["npgamma"],
    license='AGPL3+',
    long_description=open("README.md").read(),
    classifiers = [],
)
