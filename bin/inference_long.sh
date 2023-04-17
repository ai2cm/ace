#!/bin/bash

if [[ $# != 2 ]]
then
	exit 1
fi

model="$1"
output="$2"


wd=.tmp.inference_long.$model
mkdir -p $wd
config=$wd/config.json

export model wd

envsubst << EOF > $config
{
  "ensemble_members": 1,
  "noise_amplitude": 0,
  "simulation_length": 240,
  "weather_event": {
    "properties": {
      "name": "ECWMFValid",
      "start_time": "2018-01-02T00:00:00"
    },
    "domains": [
      {
        "name": "global",
        "type": "Window",
        "diagnostics": [
          {
            "type": "raw",
            "function": "",
            "channels": [
              "tcwv"
            ]
          }
        ]
      }
    ]
  },
  "output_path": "$wd/run",
  "output_frequency": 4,
  "fcn_model": "$model",
  "seed": 12345,
  "use_cuda_graphs": false,
  "ensemble_batch_size": 1,
  "autocast_fp16": false,
  "perturbation_strategy": "gaussian",
  "noise_reddening": 2
}
EOF

mpirun --allow-run-as-root -n 1 python bin/inference_ensemble.py $config
mv $wd/run/*.nc $output
