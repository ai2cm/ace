#!/bin/bash

set -e -x

function get6MonthRunPath {
    ls tests/Output.$1.Globe.*/*.nc | head -n 1
}

function check6MonthRun {
    [ -f tests/Output.$1.Globe.*/*.nc ]
}

function create6MonthRun {
	python3 bin/inference_ensemble.py --fcn_model $1 6month.json
}

function delete6MonthRun {
	rm -r tests/Output.$1.Globe.*
}

function usage {
    echo "eval_model.sh <model>"
}

if [[ $# != 1 ]]
then
    usage
    exit 1
fi

model="$1"

# run 6 month simulation
cat << EOF > 6month.json
{
	"ensemble_members": 1,
	"noise_amplitude": 0.0,
	"simulation_length": 720,
	"weather_event": {
		"properties": {
			"name": "Globe",
			"start_time": "2018-01-01 00:00:00"
		},
		"domains": [
			{
			"name": "global",
			"type": "Window",
			"diagnostics": [
				{
				"type": "raw",
				"channels": [
					"tcwv",
					"t2m",
					"u10m",
					"v10m"
				]
				}
			]
			}
		]
	},
	"output_dir": "./tests/",
	"output_frequency": 1,
	"seed": 12345,
	"use_cuda_graphs": false,
	"ensemble_batch_size": 2,
	"autocast_fp16": false,
	"perturbation_strategy": "gaussian",
	"noise_reddening": 2.0
}
EOF

if ! check6MonthRun $model
then
	create6MonthRun $model
fi

dataPath=$(get6MonthRunPath $model)
output=report/6month/$model/
mkdir -p "$output"
python plots/plot_long_time_moisture_bias.py "$dataPath" "$output"

