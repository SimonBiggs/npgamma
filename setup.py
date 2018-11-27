from setuptools import setup

setup(
    name = "npgamma",
    version = "0.8.0",
    author = "Simon Biggs",
    author_email = "me@simonbiggs.net",
    description = "`npgamma` is deprecated. Please use `gamma` module within the `pymedphys` package instead.",
    long_description = """
`npgamma` is deprecated. It has been superceded by `pymedphys.gamma`.

To install `pymedphys` see https://pymedphys.com/en/latest/getting-started/installation.html

On usage of the gamma function within `pymedphys` see
https://pymedphys.com/en/latest/user/gamma.html
    """,
    keywords = ["radiotherapy", "gamma evaluation", "gamma index", "distance to agreement", "medical physics"],
    url = "https://github.com/SimonBiggs/npgamma/",
    packages = ["npgamma"],
    license='AGPL3+',
    classifiers = [],
)
