This directory contains scripts related to generating idealized datasets for testing / better understanding the (Attention/Spherical) Fourier Neural Operator models. The following sections briefly document how the datasets were created using these scripts.

# Source dataset

We used a single netcdf file from the FV3GFS 10-year v4 dataset. While this is not a stable long-term solution, this is the path to the data file:

```sh
INPUT_FILE=/net/nfs/climate/gideond/data/fv3gfs-fcn-v4-2010-2019/2019010100.nc
```

# Overfitting Case

The classical ML workflow to test whether your model has sufficient capacity is to see whether it is able to overfit the n = 1 case. If it can’t, there’s little chance that it has capacity for a full dataset. Create a dataset {(x_1, x_2)}, a single input matched with a single output and study the training loss.

We also began to study use the spherical harmonic (SH) roundtrip as a regridding/smoothing operation. In terms of the code, these only differ by the `--smooth` flag.

## Unsmoothed

```sh
python extract_times.py -i $INPUT_FILE -o /net/nfs.cirrascale/climate/gideond/data/idealizations/train-pair/train-pair.nc -x 0 -x 1
python extract_times.py -i $INPUT_FILE -o /net/nfs.cirrascale/climate/gideond/data/idealizations/valid-pair/valid-pair.nc -x 42 -x 43
```

These files were then uploaded to beaker as

```sh
beaker dataset create -n fv3-train-pair /net/nfs.cirrascale/climate/gideond/data/idealizations/train-pair/train-pair.nc
beaker dataset create -n fv3-valid-pair /net/nfs.cirrascale/climate/gideond/data/idealizations/valid-pair/valid-pair.nc
```


## Smoothed

```sh
python extract_times.py -i $INPUT_FILE -o /net/nfs.cirrascale/climate/gideond/data/idealizations/train-pair/train-pair-smooth.nc -x 0 -x 1 --smooth
python extract_times.py -i $INPUT_FILE -o /net/nfs.cirrascale/climate/gideond/data/idealizations/valid-pair/valid-pair-smooth.nc -x 42 -x 43 --smooth
```

These files were then uploaded to beaker as

```sh
beaker dataset create -n fv3-train-pair /net/nfs.cirrascale/climate/gideond/data/idealizations/train-pair/train-pair-smooth.nc
beaker dataset create -n fv3-valid-pair /net/nfs.cirrascale/climate/gideond/data/idealizations/valid-pair/valid-pair-smooth.nc
```

# Cyclic Case

Here we want to test the relationship between training and unrolling beyond the training dataset. Construct a dataset which is slightly larger than the overfitting case: [x_1, x_2, x_1]. Now the model should learn to go from x_1 to x_2 and back again. What happens when we unroll a trained model for many steps? This system is largely local and periodic. In this idealization, we “control” for advection, i.e. there is no advection in this system.

Since we found that smoothing made such an important difference, we only studied the smooth case.

Note that in this case, we want to study longer unrolls. Plus, we can use the standard dataloaders and thus larger datasets.

```sh
TRAIN_PATH=/net/nfs.cirrascale/climate/gideond/data/idealizations/train-cyclic-010/train-cyclic-010.nc
VALID_PATH=/net/nfs.cirrascale/climate/gideond/data/idealizations/valid-cyclic-010/valid-cyclic-010.nc
REPEAT_PATH=/net/nfs.cirrascale/climate/gideond/data/idealizations/valid-cyclic-010/valid-cyclic-010-repeat-50.nc
```


```sh
python extract_times.py -i $INPUT_FILE -o $TRAIN_PATH -x 0 -x 1 -x 0 --smooth
python extract_times.py -i $INPUT_FILE -o $VALID_PATH -x 42 -x 43 -x 42 --smooth
python extract_times.py -i $INPUT_FILE -o $REPEAT_PATH -x 0 -x 1 -r 50 -z 0 --smooth
```

```sh
beaker dataset create -n fv3-train-cyclic-010 $TRAIN_PATH
beaker dataset create -n fv3-valid-cyclic-010 $VALID_PATH
beaker dataset create -n fv3-valid-cyclic-010-repeat-50 $REPEAT_PATH
```