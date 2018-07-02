# Numpy Gamma Index

## Important note

Before deciding to use this package please read the following paper:

> [Evaluating IMRT and VMAT dose accuracy: Practical examples of failure
> to detect systematic errors when applying a commonly used metric
> and action levels](http://download.xuebalib.com/xuebalib.com.42814.pdf)

## Algorithim implementation derived inspiration from the following paper

 > Wendling, M. , Zijp, L. J., McDermott, L. N., Smit, E. J., Sonke, J. , Mijnheer, B. J. and van Herk, M. (2007), *A fast algorithm for gamma evaluation in 3D*. Med. Phys., 34: 1647-1654. [doi:10.1118/1.2721657](https://doi.org/10.1118/1.2721657)

## Installation

Install with:

    pip install npgamma

## Usage examples

Using numpy and scipy to find the gamma index. This gamma index is often used in Medical Physics. Notebooks demonstrating the usage of this this module can be found at the following links:

  * http://nbviewer.ipython.org/github/SimonBiggs/npgamma/blob/master/Module%20usage%202D.ipynb
  * http://nbviewer.ipython.org/github/SimonBiggs/npgamma/blob/master/Module%20usage%203D.ipynb
  
## Independent implementation

For an independent implementation which can be used to validate npgamma see:

> <https://github.com/rickardcronholm/gammaValidation>

## License agreement

Copyright (C) 2015 Simon Biggs

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version (the "AGPL-3.0+").

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License and the additional terms for more
details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

ADDITIONAL TERMS are also included as allowed by Section 7 of the GNU
Affrero General Public License. These aditional terms are Sections 1, 5,
6, 7, 8, and 9 from the Apache License, Version 2.0 (the "Apache-2.0")
where all references to the definition "License" are instead defined to
mean the AGPL-3.0+.

You should have received a copy of the Apache-2.0 along with this
program. If not, see <http://www.apache.org/licenses/LICENSE-2.0>.
