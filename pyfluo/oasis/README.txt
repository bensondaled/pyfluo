This is from OASIS: https://github.com/j-friedrich/OASIS

When installing on a new machine, run these commands to compile the C code (or ./compile):
python setup.py build_ext --inplace
python setup.py clean --all
cp pyfluo/oasis/oasis.cpython-37m-darwin.so .
