package=$1
sed -i.bak 's/import networks/import networks.'$package'/g' networks/$package/*.py
sed -i.bak 's/from networks/from networks.'$package'/g' networks/$package/*.py
