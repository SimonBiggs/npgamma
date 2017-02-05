from setuptools import setup

setup(
    name = "npgamma",
    version = "0.7.0",
    author = "Simon Biggs",
    author_email = "mail@simonbiggs.net",
    description = "Perform a gamma evaluation comparing two arbitrary dose grids in 1, 2, or 3 dimensions.",
    long_description = """Using numpy and scipy to find the gamma index. This gamma index is often used in Medical Physics. Notebooks demonstrating the usage of this this module can be found at the following links:

  * http://nbviewer.ipython.org/github/SimonBiggs/npgamma/blob/master/Module%20usage%202D.ipynb
  * http://nbviewer.ipython.org/github/SimonBiggs/npgamma/blob/master/Module%20usage%203D.ipynb
  
Notebooks stepping through the internals of the code demonstrating how it works are given at the following links:

  * http://nbviewer.ipython.org/github/SimonBiggs/npgamma/blob/master/Method%20explanation%202D.ipynb
  * http://nbviewer.ipython.org/github/SimonBiggs/npgamma/blob/master/Method%20explanation%203D.ipynb


This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.""",
    keywords = ["radiotherapy", "gamma evaluation", "gamma index", "distance to agreement", "medical physics"],
    url = "https://github.com/SimonBiggs/npgamma/",
    packages = ["npgamma"],
    license='AGPL3+',
    classifiers = [],
)
