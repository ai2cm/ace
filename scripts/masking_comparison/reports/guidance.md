Identify the following for -bestinf runs, focus on 2014-2024 inference values of annual/air_temperature_7 metrics. Ensure in-sample values don't degrade too. The idea is to show masking improves out-of-sample 2014-2024 generalization by matching target values better than without masking.

Questions:
- Why does VarMasking8 not produce a palateaud val loss like VarMasking3?
- Why are there marginal improvements on long_46year/annual/air_temperature_7 in VarMasking8 compared to earlier projects?
- What is the role of GMR on/off, why does VarMasking8 not benefit from masking on GMR off, only GMR on?
- What is the optimal masking config for improving out-of-sample 2014-2024 annula/air_temperature_7 inference runs over a baseline?
- What are the differnces in all the project configs?
- Does bernoulli or uniform masking perform better?
