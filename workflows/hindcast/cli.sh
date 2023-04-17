model=sfno_73ch

submit () {
for year in 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015
do
	export year model
	envsubst '$year $model' < template | sbatch
done
}

list () {
	ls /lustre/fsw/sw_climate_fno/nbrenowitz/hindcast/$model/*.nc
}

$@
