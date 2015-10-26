import os
from glob import glob

notebook_paths = glob('*.ipynb')

for path in notebook_paths:
    os.system('jupyter nbconvert --to html "' + path + '"')