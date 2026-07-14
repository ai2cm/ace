# Generate "next N-month Nino3.4" label datasets aligned to a host ocean dataset.
# Run in the repo's default `fme` conda environment.
#
# Override paths on the command line, e.g.:
#   make ufs_nino_labels HOST=gs://bucket/host.zarr OUTPUT=gs://bucket/host-nino-leads.zarr

N_LEADS ?= 12
FIRST_LEAD ?= 1
# Trailing running-mean window for the Nino3.4 index (1 = raw monthly anomaly).
RUNNING_MEAN ?= 1

# --- UFS replay ocean (5-day cadence, real calendar) ---
UFS_HOST ?= gs://vcm-ml-intermediate/2026-06-29-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr
UFS_OUTPUT ?= gs://vcm-ml-intermediate/2026-06-29-ufs-replay-ocean-1deg-19level-5day-1994-2023-nino-leads.zarr
# Reference climatology period (WMO-style 30yr is not available; use the
# training span). Override as needed.
UFS_CLIM_START ?= 1994-01
UFS_CLIM_STOP ?= 2015-12

# --- CM4 1pctCO2 ocean (5-day cadence, model calendar) ---
CM4_HOST ?= gs://vcm-ml-intermediate/2025-10-16-cm4-1pctCO2-140yr-ocean-no-smoothing.zarr
CM4_OUTPUT ?= gs://vcm-ml-intermediate/2026-07-14-cm4-1pctCO2-140yr-ocean-no-smoothing-nino-leads.zarr

ufs_nino_labels:
	python compute_nino_lead_labels.py \
		--host-dataset $(UFS_HOST) \
		--output-zarr $(UFS_OUTPUT) \
		--sst-var sst \
		--n-leads $(N_LEADS) \
		--first-lead $(FIRST_LEAD) \
		--running-mean-months $(RUNNING_MEAN) \
		--clim-start $(UFS_CLIM_START) \
		--clim-stop $(UFS_CLIM_STOP)

cm4_nino_labels:
	python compute_nino_lead_labels.py \
		--host-dataset $(CM4_HOST) \
		--output-zarr $(CM4_OUTPUT) \
		--sst-var sst \
		--n-leads $(N_LEADS) \
		--first-lead $(FIRST_LEAD) \
		--running-mean-months $(RUNNING_MEAN) \
		--linear-detrend

.PHONY: ufs_nino_labels cm4_nino_labels
