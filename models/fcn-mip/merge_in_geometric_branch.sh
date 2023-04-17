git fetch fcn geometric
git reset --hard
git read-tree --prefix=geometric/ fcn/geometric:networks
rm -rf geometric
git checkout -- geometric
git mv -f geometric/* networks/geometric
