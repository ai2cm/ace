#!/bin/bash
#
# Fields: z,t,u and v at 850, 500 and 200hPa, one per file
# Initial times: 6 June 2018 to 4 June 2019 00z, with forecasts produced every 3rd day
# Sampling: Output every 6 hour, on ~0.25 degree grid, 10 day forecast
#
# channels in 73 channel model: ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl",
# "tcwv", "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600",
# "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250",
# "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", "z50",
# "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700",
# "z850", "z925", "z1000", "t50", "t100", "t150", "t200", "t250", "t300",
# "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "r50", "r100",
# "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850",
# "r925", "r1000"]

set -e -x

if [[ $# != 1 ]]
then
    2>&1 echo "evaluate_ecwmf.sh <time>"
    exit 1
fi

model=sfno_73ch
time="$1"

wd=.tmp/$time
output=ecwmf

mkdir -p $output
mkdir -p $wd

config=$wd/config.json
simulation=$wd/inference
export time model simulation
envsubst << EOF > $config
{
        "ensemble_members": 1,
        "noise_amplitude": 0.00,
        "simulation_length": 40,
        "weather_event": {
            "properties": {
                "name": "ECWMFValid",
                "start_time": "$time"
            },
            "domains": [{
                "name": "global",
                "type": "Window",
                "diagnostics": [
                    {
                        "type": "raw",
                        "function": "",
                        "channels": [
                            "u200",
                            "v200",
                            "t200",
                            "z200",
                            "u500",
                            "v500",
                            "t500",
                            "z500",
                            "u850",
                            "v850",
                            "t850",
                            "z850"
                        ]
                    }
                ]
            }]
        },
        "output_path": "$simulation",
        "output_frequency": 1,
        "fcn_model": "$model",
        "seed": 12345,
        "use_cuda_graphs": false,
        "ensemble_batch_size": 2,
        "autocast_fp16": false,
        "perturbation_strategy": "gaussian",
        "noise_reddening": 2.0
}
EOF

python bin/inference_ensemble.py $config
# squeeze the "ensemble" dim
ncwa -O -G : -g global -a ensemble $simulation/*.nc $output/$time.nc