#!/bin/bash
# Shared library for coupled/uncoupled training experiment initialization
# Provides common functions for parsing experiment configuration yaml files,
# creating experiment directories in configs/experiments, and populating config
# templates.

set -e

# Cache repository root for performance (avoid repeated git rev-parse calls)
if [[ -z "${REPO_ROOT:-}" ]]; then
    REPO_ROOT=$(git rev-parse --show-toplevel)
    export REPO_ROOT
fi

# Get dataset value from datasets.yaml using yq
# Args:
#   $1 - DATASET_GROUP (e.g., cm4_piControl, cm4_1pctCO2)
#   $2 - DATASET_KEY (e.g., ocean, coupled_ocean, atmosphere)
# Outputs: Dataset filename to stdout
# Example: OCEAN_DATASET=$(get_dataset_value "cm4_piControl" "ocean")
get_dataset_value() {
    local DATASET_GROUP="$1"
    local DATASET_KEY="$2"

    yq ".${DATASET_GROUP}.datasets.${DATASET_KEY}" "$REPO_ROOT/job_runner/datasets.yaml"
}

# Create experiment directory structure
# Args:
#   $1 - NAME (e.g., my_exper)
#   $2 - TEMPLATE (e.g., coupled, uncoupled_ocean, uncoupled_atmos)
# Creates: configs/experiments/<name>/<template>/
# Outputs: Full path to created directory
create_experiment_directory() {
    local NAME="$1"
    local TEMPLATE="$2"

    local EXPERIMENT_DIR="$REPO_ROOT/configs/experiments/${NAME}/${TEMPLATE}"

    if [[ -d "$EXPERIMENT_DIR" ]]; then
        echo "Error: Experiment directory already exists: $EXPERIMENT_DIR" >&2
        return 1
    fi

    mkdir -p "$EXPERIMENT_DIR"
    echo "$EXPERIMENT_DIR"
}

# Create input txt files with proper headers
# Args:
#   $1 - EXPERIMENT_DIR (full path to experiment directory)
#   $2 - TEMPLATE_TYPE (e.g., ocean, atmos, coupled)
create_input_txt_files() {
    local EXPERIMENT_DIR="$1"
    local TEMPLATE_TYPE="$2"

    # Determine which files to create based on template type
    case "$TEMPLATE_TYPE" in
        ocean|atmos)
            # Create training.txt with header
            echo "group|tag|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override_args" \
                > "$EXPERIMENT_DIR/training.txt"

            # Create finetuning.txt with header (note: 'tag' is second field)
            echo "group|tag|wandb_project|wandb_id|ckpt_type|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override|results_dataset" \
                > "$EXPERIMENT_DIR/finetuning.txt"

            # Create experiments.txt with header
            echo "group|tag|experiment_id|status|checkpoint|priority|preemptible|override|results_dataset|workspace" \
                > "$EXPERIMENT_DIR/experiments.txt"

            # Create resuming.txt with header
            echo "group|tag|wandb_project|wandb_id|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override|results_dataset" \
                > "$EXPERIMENT_DIR/resuming.txt"
            ;;

        coupled)
            # Create pretraining.txt with header (note: 'tag' is second field)
            echo "group|tag|ocean_project|ocean_wandb_id|ocean_ckpt_type|atmos_project|atmos_wandb_id|atmos_ckpt_type|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override|ocean_results_dataset|atmos_results_dataset" \
                > "$EXPERIMENT_DIR/pretraining.txt"

            # Create finetuning.txt with header (note: 'tag' is second field)
            echo "group|tag|wandb_project|wandb_id|ckpt_type|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override|results_dataset" \
                > "$EXPERIMENT_DIR/finetuning.txt"

            # Create experiments.txt with header
            echo "group|tag|experiment_id|status|checkpoint|priority|preemptible|override|results_dataset|workspace" \
                > "$EXPERIMENT_DIR/experiments.txt"

            # Create resuming.txt with header
            echo "group|tag|wandb_project|wandb_id|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override|results_dataset|ocean_results_dataset|atmos_results_dataset" \
                > "$EXPERIMENT_DIR/resuming.txt"
            ;;

        *)
            echo "Warning: Unknown template type '$TEMPLATE_TYPE', creating generic files" >&2
            echo "group|tag|experiment_id|status|checkpoint|priority|preemptible|override|results_dataset|workspace" \
                > "$EXPERIMENT_DIR/experiments.txt"
            ;;
    esac
}

# Populate ocean config from experiment metadata
# Args:
#   $1 - TEMPLATE_PATH (source template file)
#   $2 - OUTPUT_PATH (destination config file)
#   $3 - EXPERIMENT_NAME (experiment file name without .yaml extension)
populate_ocean_config_from_experiment() {
    local TEMPLATE_PATH="$1"
    local OUTPUT_PATH="$2"
    local EXPERIMENT_NAME="$3"
    local DATASETS_YAML="$REPO_ROOT/job_runner/datasets.yaml"
    local EXPERS_FILE="$REPO_ROOT/job_runner/expers/${EXPERIMENT_NAME}.yaml"

    cp "$TEMPLATE_PATH" "$OUTPUT_PATH"

    # Batch read all needed values from experiment file in one yq call
    local EXPER_JSON=$(yq eval -o json '{
        "train_group": .datasets.train,
        "validation_group": (.datasets.validation // .datasets.train),
        "inference_group": (.datasets.inference // .datasets.validation // .datasets.train),
        "train_subset": .train.subset,
        "validation_subset": .validation.subset,
        "ocean_steps": .inference.ocean.n_forward_steps,
        "ocean_memory": .inference.ocean.forward_steps_in_memory,
        "times": .inference.times,
        "logging_entity": .logging.entity
    }' "$EXPERS_FILE")

    local TRAIN_GROUP=$(echo "$EXPER_JSON" | jq -r '.train_group')
    if [[ -z "$TRAIN_GROUP" || "$TRAIN_GROUP" == "null" ]]; then
        echo "Error: Experiment '${EXPERIMENT_NAME}' missing datasets.train" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    local VALIDATION_GROUP=$(echo "$EXPER_JSON" | jq -r '.validation_group')
    local INFERENCE_GROUP=$(echo "$EXPER_JSON" | jq -r '.inference_group')

    # Batch read all dataset paths in one yq call
    local DATASETS_JSON=$(yq eval -o json '{
        "train_coupled": .'"${TRAIN_GROUP}"'.coupled_ocean,
        "train_ocean": .'"${TRAIN_GROUP}"'.ocean,
        "val_coupled": .'"${VALIDATION_GROUP}"'.coupled_ocean,
        "val_ocean": .'"${VALIDATION_GROUP}"'.ocean,
        "inf_coupled": .'"${INFERENCE_GROUP}"'.coupled_ocean,
        "inf_ocean": .'"${INFERENCE_GROUP}"'.ocean
    }' "$DATASETS_YAML")

    local TRAIN_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.train_coupled')
    local TRAIN_OCEAN=$(echo "$DATASETS_JSON" | jq -r '.train_ocean')
    if [[ "$TRAIN_COUPLED" == "null" || "$TRAIN_OCEAN" == "null" ]]; then
        echo "Error: Dataset group '${TRAIN_GROUP}' missing ocean entries" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    local VAL_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.val_coupled')
    local VAL_OCEAN=$(echo "$DATASETS_JSON" | jq -r '.val_ocean')
    if [[ "$VAL_COUPLED" == "null" || "$VAL_OCEAN" == "null" ]]; then
        echo "Error: Dataset group '${VALIDATION_GROUP}' missing ocean entries" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    local INF_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.inf_coupled')
    local INF_OCEAN=$(echo "$DATASETS_JSON" | jq -r '.inf_ocean')
    if [[ "$INF_COUPLED" == "null" || "$INF_OCEAN" == "null" ]]; then
        echo "Error: Dataset group '${INFERENCE_GROUP}' missing ocean entries" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    # Extract optional values
    local TRAIN_SUBSET=$(echo "$EXPER_JSON" | jq -r '.train_subset')
    local VAL_SUBSET=$(echo "$EXPER_JSON" | jq -r '.validation_subset')
    local OCEAN_STEPS=$(echo "$EXPER_JSON" | jq -r '.ocean_steps')
    local OCEAN_MEMORY=$(echo "$EXPER_JSON" | jq -r '.ocean_memory')
    local TIMES=$(echo "$EXPER_JSON" | jq -r '.times')
    local LOGGING_ENTITY=$(echo "$EXPER_JSON" | jq -r '.logging_entity')

    # Build single batched yq expression for all modifications
    local YQ_EXPR=".train_loader.dataset.merge[0].file_pattern = \"$TRAIN_COUPLED\" |
        .train_loader.dataset.merge[1].file_pattern = \"$TRAIN_OCEAN\" |
        .validation_loader.dataset.merge[0].file_pattern = \"$VAL_COUPLED\" |
        .validation_loader.dataset.merge[1].file_pattern = \"$VAL_OCEAN\" |
        .inference.loader.dataset.merge[0].file_pattern = \"$INF_COUPLED\" |
        .inference.loader.dataset.merge[1].file_pattern = \"$INF_OCEAN\""

    # Handle train subset
    if [[ "$TRAIN_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .train_loader.dataset.merge[].subset = load(\"$EXPERS_FILE\").train.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.train_loader.dataset.merge[].subset)"
    fi

    # Handle validation subset
    if [[ "$VAL_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .validation_loader.dataset.merge[].subset = load(\"$EXPERS_FILE\").validation.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.validation_loader.dataset.merge[].subset)"
    fi

    # Handle ocean steps
    if [[ -n "$OCEAN_STEPS" && "$OCEAN_STEPS" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.n_forward_steps = $OCEAN_STEPS | .inference.n_forward_steps = $OCEAN_STEPS"
    fi

    # Handle ocean memory
    if [[ -n "$OCEAN_MEMORY" && "$OCEAN_MEMORY" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.forward_steps_in_memory = $OCEAN_MEMORY"
    fi

    # Handle times
    if [[ "$TIMES" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.loader.start_indices.times = load(\"$EXPERS_FILE\").inference.times"
    else
        YQ_EXPR="$YQ_EXPR | del(.inference.loader.start_indices.times)"
    fi

    # Handle logging
    if [[ -n "$LOGGING_ENTITY" && "$LOGGING_ENTITY" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .logging.entity = \"$LOGGING_ENTITY\""
    fi

    # Apply all modifications in a single yq call
    yq -i "$YQ_EXPR" "$OUTPUT_PATH"
}

# Populate atmosphere config from experiment metadata
populate_atmos_config_from_experiment() {
    local TEMPLATE_PATH="$1"
    local OUTPUT_PATH="$2"
    local EXPERIMENT_NAME="$3"
    local DATASETS_YAML="$REPO_ROOT/job_runner/datasets.yaml"
    local EXPERS_FILE="$REPO_ROOT/job_runner/expers/${EXPERIMENT_NAME}.yaml"

    cp "$TEMPLATE_PATH" "$OUTPUT_PATH"

    # Batch read all needed values from experiment file in one yq call
    local EXPER_JSON=$(yq eval -o json '{
        "train_group": .datasets.train,
        "validation_group": (.datasets.validation // .datasets.train),
        "inference_group": (.datasets.inference // .datasets.validation // .datasets.train),
        "train_subset": .train.subset,
        "validation_subset": .validation.subset,
        "atmos_steps": .inference.atmos.n_forward_steps,
        "atmos_memory": .inference.atmos.forward_steps_in_memory,
        "times": .inference.times,
        "logging_entity": .logging.entity
    }' "$EXPERS_FILE")

    local TRAIN_GROUP=$(echo "$EXPER_JSON" | jq -r '.train_group')
    if [[ -z "$TRAIN_GROUP" || "$TRAIN_GROUP" == "null" ]]; then
        echo "Error: Experiment '${EXPERIMENT_NAME}' missing datasets.train" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    local VALIDATION_GROUP=$(echo "$EXPER_JSON" | jq -r '.validation_group')
    local INFERENCE_GROUP=$(echo "$EXPER_JSON" | jq -r '.inference_group')

    # Batch read all dataset paths in one yq call (handle both atmos and atmosphere naming)
    local DATASETS_JSON=$(yq eval -o json '{
        "train_coupled": .'"${TRAIN_GROUP}"'.coupled_sea_ice,
        "train_atmos": .'"${TRAIN_GROUP}"'.atmos,
        "val_coupled": .'"${VALIDATION_GROUP}"'.coupled_sea_ice,
        "val_atmos": .'"${VALIDATION_GROUP}"'.atmos,
        "inf_coupled": .'"${INFERENCE_GROUP}"'.coupled_sea_ice,
        "inf_atmos": .'"${INFERENCE_GROUP}"'.atmos
    }' "$DATASETS_YAML")

    local TRAIN_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.train_coupled')
    local TRAIN_ATMOS=$(echo "$DATASETS_JSON" | jq -r '.train_atmos')
    if [[ "$TRAIN_COUPLED" == "null" || "$TRAIN_ATMOS" == "null" ]]; then
        echo "Error: Dataset group '${TRAIN_GROUP}' missing atmosphere entries" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    local VAL_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.val_coupled')
    local VAL_ATMOS=$(echo "$DATASETS_JSON" | jq -r '.val_atmos')
    if [[ "$VAL_COUPLED" == "null" || "$VAL_ATMOS" == "null" ]]; then
        echo "Error: Dataset group '${VALIDATION_GROUP}' missing atmosphere entries" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    local INF_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.inf_coupled')
    local INF_ATMOS=$(echo "$DATASETS_JSON" | jq -r '.inf_atmos')
    if [[ "$INF_COUPLED" == "null" || "$INF_ATMOS" == "null" ]]; then
        echo "Error: Dataset group '${INFERENCE_GROUP}' missing atmosphere entries" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    # Extract optional values
    local TRAIN_SUBSET=$(echo "$EXPER_JSON" | jq -r '.train_subset')
    local VAL_SUBSET=$(echo "$EXPER_JSON" | jq -r '.validation_subset')
    local ATMOS_STEPS=$(echo "$EXPER_JSON" | jq -r '.atmos_steps')
    local ATMOS_MEMORY=$(echo "$EXPER_JSON" | jq -r '.atmos_memory')
    local TIMES=$(echo "$EXPER_JSON" | jq -r '.times')
    local LOGGING_ENTITY=$(echo "$EXPER_JSON" | jq -r '.logging_entity')

    # Build single batched yq expression for all modifications
    local YQ_EXPR=".train_loader.dataset.merge[0].file_pattern = \"$TRAIN_COUPLED\" |
        .train_loader.dataset.merge[1].file_pattern = \"$TRAIN_ATMOS\" |
        .validation_loader.dataset.merge[0].file_pattern = \"$VAL_COUPLED\" |
        .validation_loader.dataset.merge[1].file_pattern = \"$VAL_ATMOS\" |
        .inference.loader.dataset.merge[0].file_pattern = \"$INF_COUPLED\" |
        .inference.loader.dataset.merge[1].file_pattern = \"$INF_ATMOS\""

    # Handle train subset
    if [[ "$TRAIN_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .train_loader.dataset.merge[].subset = load(\"$EXPERS_FILE\").train.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.train_loader.dataset.merge[].subset)"
    fi

    # Handle validation subset
    if [[ "$VAL_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .validation_loader.dataset.merge[].subset = load(\"$EXPERS_FILE\").validation.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.validation_loader.dataset.merge[].subset)"
    fi

    # Handle atmos steps
    if [[ -n "$ATMOS_STEPS" && "$ATMOS_STEPS" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.n_forward_steps = $ATMOS_STEPS | .inference.n_forward_steps = $ATMOS_STEPS"
    fi

    # Handle atmos memory
    if [[ -n "$ATMOS_MEMORY" && "$ATMOS_MEMORY" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.forward_steps_in_memory = $ATMOS_MEMORY"
    fi

    # Handle times
    if [[ "$TIMES" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.loader.start_indices.times = load(\"$EXPERS_FILE\").inference.times"
    else
        YQ_EXPR="$YQ_EXPR | del(.inference.loader.start_indices.times)"
    fi

    # Handle logging
    if [[ -n "$LOGGING_ENTITY" && "$LOGGING_ENTITY" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .logging.entity = \"$LOGGING_ENTITY\""
    fi

    # Apply all modifications in a single yq call
    yq -i "$YQ_EXPR" "$OUTPUT_PATH"
}

# Populate coupled config from experiment metadata
populate_coupled_config_from_experiment() {
    local TEMPLATE_PATH="$1"
    local OUTPUT_PATH="$2"
    local EXPERIMENT_NAME="$3"
    local DATASETS_YAML="$REPO_ROOT/job_runner/datasets.yaml"
    local EXPERS_FILE="$REPO_ROOT/job_runner/expers/${EXPERIMENT_NAME}.yaml"

    cp "$TEMPLATE_PATH" "$OUTPUT_PATH"

    # Batch read all needed values from experiment file in one yq call
    local EXPER_JSON=$(yq eval -o json '{
        "train_group": .datasets.train,
        "validation_group": (.datasets.validation // .datasets.train),
        "inference_group": (.datasets.inference // .datasets.validation // .datasets.train),
        "train_subset": .train.subset,
        "validation_subset": .validation.subset,
        "coupled_subset": .inference.coupled.subset,
        "coupled_steps": .inference.coupled.n_coupled_steps,
        "coupled_memory": .inference.coupled.coupled_steps_in_memory,
        "times": .inference.times,
        "logging_entity": .logging.entity
    }' "$EXPERS_FILE")

    local TRAIN_GROUP=$(echo "$EXPER_JSON" | jq -r '.train_group')
    if [[ -z "$TRAIN_GROUP" || "$TRAIN_GROUP" == "null" ]]; then
        echo "Error: Experiment '${EXPERIMENT_NAME}' missing datasets.train" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    local VALIDATION_GROUP=$(echo "$EXPER_JSON" | jq -r '.validation_group')
    local INFERENCE_GROUP=$(echo "$EXPER_JSON" | jq -r '.inference_group')

    # Batch read all dataset paths in one yq call
    local DATASETS_JSON=$(yq eval -o json '{
        "train_coupled_o": .'"${TRAIN_GROUP}"'.coupled_ocean,
        "train_o": .'"${TRAIN_GROUP}"'.ocean,
        "train_coupled_a": .'"${TRAIN_GROUP}"'.coupled_atmos,
        "train_a": .'"${TRAIN_GROUP}"'.atmos,
        "val_coupled_o": .'"${VALIDATION_GROUP}"'.coupled_ocean,
        "val_o": .'"${VALIDATION_GROUP}"'.ocean,
        "val_coupled_a": .'"${VALIDATION_GROUP}"'.coupled_atmos,
        "val_a": .'"${VALIDATION_GROUP}"'.atmos,
        "inf_coupled_o": .'"${INFERENCE_GROUP}"'.coupled_ocean,
        "inf_o": .'"${INFERENCE_GROUP}"'.ocean,
        "inf_coupled_a": .'"${INFERENCE_GROUP}"'.coupled_atmos,
        "inf_a": .'"${INFERENCE_GROUP}"'.atmos
    }' "$DATASETS_YAML")

    local TRAIN_COUPLED_O=$(echo "$DATASETS_JSON" | jq -r '.train_coupled_o')
    local TRAIN_O=$(echo "$DATASETS_JSON" | jq -r '.train_o')
    local TRAIN_COUPLED_A=$(echo "$DATASETS_JSON" | jq -r '.train_coupled_a')
    local TRAIN_A=$(echo "$DATASETS_JSON" | jq -r '.train_a')
    if [[ "$TRAIN_COUPLED_O" == "null" || "$TRAIN_O" == "null" || "$TRAIN_COUPLED_A" == "null" || "$TRAIN_A" == "null" ]]; then
        echo "Error: Dataset group '${TRAIN_GROUP}' missing coupled entries" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    local VAL_COUPLED_O=$(echo "$DATASETS_JSON" | jq -r '.val_coupled_o')
    local VAL_O=$(echo "$DATASETS_JSON" | jq -r '.val_o')
    local VAL_COUPLED_A=$(echo "$DATASETS_JSON" | jq -r '.val_coupled_a')
    local VAL_A=$(echo "$DATASETS_JSON" | jq -r '.val_a')
    if [[ "$VAL_COUPLED_O" == "null" || "$VAL_O" == "null" || "$VAL_COUPLED_A" == "null" || "$VAL_A" == "null" ]]; then
        echo "Error: Dataset group '${VALIDATION_GROUP}' missing coupled entries" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    local INF_COUPLED_O=$(echo "$DATASETS_JSON" | jq -r '.inf_coupled_o')
    local INF_O=$(echo "$DATASETS_JSON" | jq -r '.inf_o')
    local INF_COUPLED_A=$(echo "$DATASETS_JSON" | jq -r '.inf_coupled_a')
    local INF_A=$(echo "$DATASETS_JSON" | jq -r '.inf_a')
    if [[ "$INF_COUPLED_O" == "null" || "$INF_O" == "null" || "$INF_COUPLED_A" == "null" || "$INF_A" == "null" ]]; then
        echo "Error: Dataset group '${INFERENCE_GROUP}' missing coupled entries" >&2
        rm -f "$OUTPUT_PATH"
        return 1
    fi

    # Extract optional values
    local TRAIN_SUBSET=$(echo "$EXPER_JSON" | jq -r '.train_subset')
    local VAL_SUBSET=$(echo "$EXPER_JSON" | jq -r '.validation_subset')
    local COUPLED_SUBSET=$(echo "$EXPER_JSON" | jq -r '.coupled_subset')
    local COUPLED_STEPS=$(echo "$EXPER_JSON" | jq -r '.coupled_steps')
    local COUPLED_MEMORY=$(echo "$EXPER_JSON" | jq -r '.coupled_memory')
    local TIMES=$(echo "$EXPER_JSON" | jq -r '.times')
    local LOGGING_ENTITY=$(echo "$EXPER_JSON" | jq -r '.logging_entity')

    # Build single batched yq expression for all modifications
    local YQ_EXPR=".train_loader.dataset[0].ocean.merge[0].file_pattern = \"$TRAIN_COUPLED_O\" |
        .train_loader.dataset[0].ocean.merge[1].file_pattern = \"$TRAIN_O\" |
        .train_loader.dataset[0].atmosphere.merge[0].file_pattern = \"$TRAIN_COUPLED_A\" |
        .train_loader.dataset[0].atmosphere.merge[1].file_pattern = \"$TRAIN_A\" |
        .validation_loader.dataset[0].ocean.merge[0].file_pattern = \"$VAL_COUPLED_O\" |
        .validation_loader.dataset[0].ocean.merge[1].file_pattern = \"$VAL_O\" |
        .validation_loader.dataset[0].atmosphere.merge[0].file_pattern = \"$VAL_COUPLED_A\" |
        .validation_loader.dataset[0].atmosphere.merge[1].file_pattern = \"$VAL_A\" |
        .inference.loader.dataset.ocean.merge[0].file_pattern = \"$INF_COUPLED_O\" |
        .inference.loader.dataset.ocean.merge[1].file_pattern = \"$INF_O\" |
        .inference.loader.dataset.atmosphere.merge[0].file_pattern = \"$INF_COUPLED_A\" |
        .inference.loader.dataset.atmosphere.merge[1].file_pattern = \"$INF_A\""

    # Handle train subset
    if [[ "$TRAIN_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .train_loader.dataset[0].ocean.merge[].subset = load(\"$EXPERS_FILE\").train.subset |
            .train_loader.dataset[0].atmosphere.merge[].subset = load(\"$EXPERS_FILE\").train.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.train_loader.dataset[0].ocean.merge[].subset) |
            del(.train_loader.dataset[0].atmosphere.merge[].subset)"
    fi

    # Handle validation subset
    if [[ "$VAL_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .validation_loader.dataset[0].ocean.merge[].subset = load(\"$EXPERS_FILE\").validation.subset |
            .validation_loader.dataset[0].atmosphere.merge[].subset = load(\"$EXPERS_FILE\").validation.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.validation_loader.dataset[0].ocean.merge[].subset) |
            del(.validation_loader.dataset[0].atmosphere.merge[].subset)"
    fi

    # Handle coupled inference subset
    if [[ "$COUPLED_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.loader.dataset.ocean.merge[].subset = load(\"$EXPERS_FILE\").inference.coupled.subset |
            .inference.loader.dataset.atmosphere.merge[].subset = load(\"$EXPERS_FILE\").inference.coupled.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.inference.loader.dataset.ocean.merge[].subset) |
            del(.inference.loader.dataset.atmosphere.merge[].subset)"
    fi

    # Handle coupled steps
    if [[ -n "$COUPLED_STEPS" && "$COUPLED_STEPS" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .n_coupled_steps = $COUPLED_STEPS | .inference.n_coupled_steps = $COUPLED_STEPS"
    fi

    # Handle coupled memory
    if [[ -n "$COUPLED_MEMORY" && "$COUPLED_MEMORY" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .coupled_steps_in_memory = $COUPLED_MEMORY | .inference.coupled_steps_in_memory = $COUPLED_MEMORY"
    fi

    # Handle times
    if [[ "$TIMES" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.loader.start_indices.times = load(\"$EXPERS_FILE\").inference.times"
    else
        YQ_EXPR="$YQ_EXPR | del(.inference.loader.start_indices.times)"
    fi

    # Handle logging
    if [[ -n "$LOGGING_ENTITY" && "$LOGGING_ENTITY" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .logging.entity = \"$LOGGING_ENTITY\""
    fi

    # Apply all modifications in a single yq call
    yq -i "$YQ_EXPR" "$OUTPUT_PATH"
}

# Populate finetuning config from experiment metadata (ocean/atmos)
# Args:
#   $1 - TEMPLATE_PATH (source template file)
#   $2 - OUTPUT_PATH (destination config file)
#   $3 - EXPERIMENT_NAME (experiment file name without .yaml extension)
#   $4 - TEMPLATE_TYPE (ocean or atmos)
populate_finetune_config_from_experiment() {
    local TEMPLATE_PATH="$1"
    local OUTPUT_PATH="$2"
    local EXPERIMENT_NAME="$3"
    local TEMPLATE_TYPE="$4"
    local DATASETS_YAML="$REPO_ROOT/job_runner/datasets.yaml"
    local EXPERS_FILE="$REPO_ROOT/job_runner/expers/${EXPERIMENT_NAME}.yaml"

    cp "$TEMPLATE_PATH" "$OUTPUT_PATH"

    # Batch read all needed values from experiment file in one yq call
    local EXPER_JSON=$(yq eval -o json '{
        "train_group": .datasets.train,
        "validation_group": (.datasets.validation // .datasets.train),
        "inference_group": (.datasets.inference // .datasets.validation // .datasets.train),
        "train_subset": .train.subset,
        "validation_subset": .validation.subset,
        "n_forward_steps": .inference.'"${TEMPLATE_TYPE}"'.n_forward_steps,
        "forward_steps_in_memory": .inference.'"${TEMPLATE_TYPE}"'.forward_steps_in_memory,
        "times": .inference.times,
        "logging_entity": .logging.entity
    }' "$EXPERS_FILE")

    local TRAIN_GROUP=$(echo "$EXPER_JSON" | jq -r '.train_group')
    local VALIDATION_GROUP=$(echo "$EXPER_JSON" | jq -r '.validation_group')
    local INFERENCE_GROUP=$(echo "$EXPER_JSON" | jq -r '.inference_group')

    # Batch read all dataset paths in one yq call
    if [[ "$TEMPLATE_TYPE" == "ocean" ]]; then
        local DATASETS_JSON=$(yq eval -o json '{
            "train_coupled": .'"${TRAIN_GROUP}"'.coupled_ocean,
            "train_zarr": .'"${TRAIN_GROUP}"'.ocean,
            "val_coupled": .'"${VALIDATION_GROUP}"'.coupled_ocean,
            "val_zarr": .'"${VALIDATION_GROUP}"'.ocean,
            "inf_coupled": .'"${INFERENCE_GROUP}"'.coupled_ocean,
            "inf_zarr": .'"${INFERENCE_GROUP}"'.ocean
        }' "$DATASETS_YAML")
    else  # atmos
        local DATASETS_JSON=$(yq eval -o json '{
            "train_coupled": .'"${TRAIN_GROUP}"'.coupled_sea_ice,
            "train_zarr": .'"${TRAIN_GROUP}"'.atmos,
            "val_coupled": .'"${VALIDATION_GROUP}"'.coupled_sea_ice,
            "val_zarr": .'"${VALIDATION_GROUP}"'.atmos,
            "inf_coupled": .'"${INFERENCE_GROUP}"'.coupled_sea_ice,
            "inf_zarr": .'"${INFERENCE_GROUP}"'.atmos
        }' "$DATASETS_YAML")
    fi

    local TRAIN_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.train_coupled')
    local TRAIN_ZARR=$(echo "$DATASETS_JSON" | jq -r '.train_zarr')
    local VAL_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.val_coupled')
    local VAL_ZARR=$(echo "$DATASETS_JSON" | jq -r '.val_zarr')
    local INF_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.inf_coupled')
    local INF_ZARR=$(echo "$DATASETS_JSON" | jq -r '.inf_zarr')

    # Extract optional values
    local TRAIN_SUBSET=$(echo "$EXPER_JSON" | jq -r '.train_subset')
    local VAL_SUBSET=$(echo "$EXPER_JSON" | jq -r '.validation_subset')
    local N_FORWARD_STEPS=$(echo "$EXPER_JSON" | jq -r '.n_forward_steps')
    local FORWARD_STEPS_IN_MEMORY=$(echo "$EXPER_JSON" | jq -r '.forward_steps_in_memory')
    local TIMES=$(echo "$EXPER_JSON" | jq -r '.times')
    local LOGGING_ENTITY=$(echo "$EXPER_JSON" | jq -r '.logging_entity')

    # Build single batched yq expression for all modifications
    local YQ_EXPR=""

    # Handle dataset patterns
    if [[ "$INF_COUPLED" != "null" ]]; then
        YQ_EXPR=".inference.loader.dataset.merge[0].file_pattern = \"$INF_COUPLED\""
    fi
    if [[ "$INF_ZARR" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .inference.loader.dataset.merge[1].file_pattern = \"$INF_ZARR\""
    fi

    if [[ "$TRAIN_COUPLED" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .train_loader.dataset.merge[0].file_pattern = \"$TRAIN_COUPLED\""
    fi
    if [[ "$TRAIN_ZARR" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .train_loader.dataset.merge[1].file_pattern = \"$TRAIN_ZARR\""
    fi

    if [[ "$VAL_COUPLED" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .validation_loader.dataset.merge[0].file_pattern = \"$VAL_COUPLED\""
    fi
    if [[ "$VAL_ZARR" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .validation_loader.dataset.merge[1].file_pattern = \"$VAL_ZARR\""
    fi

    # Handle forward steps
    if [[ -n "$N_FORWARD_STEPS" && "$N_FORWARD_STEPS" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .inference.n_forward_steps = $N_FORWARD_STEPS | .inference.n_forward_steps = $N_FORWARD_STEPS"
    fi

    # Handle forward steps in memory
    if [[ -n "$FORWARD_STEPS_IN_MEMORY" && "$FORWARD_STEPS_IN_MEMORY" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .inference.forward_steps_in_memory = $FORWARD_STEPS_IN_MEMORY"
    fi

    # Handle train subset
    if [[ "$TRAIN_SUBSET" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .train_loader.dataset.merge[].subset = load(\"$EXPERS_FILE\").train.subset"
    else
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR del(.train_loader.dataset.merge[].subset)"
    fi

    # Handle validation subset
    if [[ "$VAL_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .validation_loader.dataset.merge[].subset = load(\"$EXPERS_FILE\").validation.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.validation_loader.dataset.merge[].subset)"
    fi

    # Handle times
    if [[ "$TIMES" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.loader.start_indices.times = load(\"$EXPERS_FILE\").inference.times"
    fi

    # Handle logging
    if [[ -n "$LOGGING_ENTITY" && "$LOGGING_ENTITY" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .logging.entity = \"$LOGGING_ENTITY\""
    fi

    # Apply all modifications in a single yq call
    yq -i "$YQ_EXPR" "$OUTPUT_PATH"
}

# Populate coupled finetuning config from experiment metadata
# Args:
#   $1 - TEMPLATE_PATH (source template file)
#   $2 - OUTPUT_PATH (destination config file)
#   $3 - EXPERIMENT_NAME (experiment file name without .yaml extension)
populate_finetune_coupled_config_from_experiment() {
    local TEMPLATE_PATH="$1"
    local OUTPUT_PATH="$2"
    local EXPERIMENT_NAME="$3"
    local DATASETS_YAML="$REPO_ROOT/job_runner/datasets.yaml"
    local EXPERS_FILE="$REPO_ROOT/job_runner/expers/${EXPERIMENT_NAME}.yaml"

    cp "$TEMPLATE_PATH" "$OUTPUT_PATH"

    # Batch read all needed values from experiment file in one yq call
    local EXPER_JSON=$(yq eval -o json '{
        "train_group": .datasets.train,
        "validation_group": (.datasets.validation // .datasets.train),
        "inference_group": (.datasets.inference // .datasets.validation // .datasets.train),
        "train_subset": .train.subset,
        "validation_subset": .validation.subset,
        "n_coupled_steps": .inference.coupled.n_coupled_steps,
        "coupled_steps_in_memory": .inference.coupled.coupled_steps_in_memory,
        "times": .inference.times,
        "logging_entity": .logging.entity
    }' "$EXPERS_FILE")

    local TRAIN_GROUP=$(echo "$EXPER_JSON" | jq -r '.train_group')
    local VALIDATION_GROUP=$(echo "$EXPER_JSON" | jq -r '.validation_group')
    local INFERENCE_GROUP=$(echo "$EXPER_JSON" | jq -r '.inference_group')

    # Batch read all dataset paths in one yq call
    local DATASETS_JSON=$(yq eval -o json '{
        "train_coupled_o": .'"${TRAIN_GROUP}"'.coupled_ocean,
        "train_o": .'"${TRAIN_GROUP}"'.ocean,
        "train_coupled_a": .'"${TRAIN_GROUP}"'.coupled_atmos,
        "train_a": .'"${TRAIN_GROUP}"'.atmos,
        "val_coupled_o": .'"${VALIDATION_GROUP}"'.coupled_ocean,
        "val_o": .'"${VALIDATION_GROUP}"'.ocean,
        "val_coupled_a": .'"${VALIDATION_GROUP}"'.coupled_atmos,
        "val_a": .'"${VALIDATION_GROUP}"'.atmos,
        "inf_coupled_o": .'"${INFERENCE_GROUP}"'.coupled_ocean,
        "inf_o": .'"${INFERENCE_GROUP}"'.ocean,
        "inf_coupled_a": .'"${INFERENCE_GROUP}"'.coupled_atmos,
        "inf_a": .'"${INFERENCE_GROUP}"'.atmos
    }' "$DATASETS_YAML")

    local TRAIN_COUPLED_O=$(echo "$DATASETS_JSON" | jq -r '.train_coupled_o')
    local TRAIN_O=$(echo "$DATASETS_JSON" | jq -r '.train_o')
    local TRAIN_COUPLED_A=$(echo "$DATASETS_JSON" | jq -r '.train_coupled_a')
    local TRAIN_A=$(echo "$DATASETS_JSON" | jq -r '.train_a')
    local VAL_COUPLED_O=$(echo "$DATASETS_JSON" | jq -r '.val_coupled_o')
    local VAL_O=$(echo "$DATASETS_JSON" | jq -r '.val_o')
    local VAL_COUPLED_A=$(echo "$DATASETS_JSON" | jq -r '.val_coupled_a')
    local VAL_A=$(echo "$DATASETS_JSON" | jq -r '.val_a')
    local INF_COUPLED_O=$(echo "$DATASETS_JSON" | jq -r '.inf_coupled_o')
    local INF_O=$(echo "$DATASETS_JSON" | jq -r '.inf_o')
    local INF_COUPLED_A=$(echo "$DATASETS_JSON" | jq -r '.inf_coupled_a')
    local INF_A=$(echo "$DATASETS_JSON" | jq -r '.inf_a')

    # Extract optional values
    local TRAIN_SUBSET=$(echo "$EXPER_JSON" | jq -r '.train_subset')
    local VAL_SUBSET=$(echo "$EXPER_JSON" | jq -r '.validation_subset')
    local N_COUPLED_STEPS=$(echo "$EXPER_JSON" | jq -r '.n_coupled_steps')
    local COUPLED_STEPS_IN_MEMORY=$(echo "$EXPER_JSON" | jq -r '.coupled_steps_in_memory')
    local TIMES=$(echo "$EXPER_JSON" | jq -r '.times')
    local LOGGING_ENTITY=$(echo "$EXPER_JSON" | jq -r '.logging_entity')

    # Build single batched yq expression for all modifications
    local YQ_EXPR=""

    # Handle inference datasets
    if [[ "$INF_COUPLED_O" != "null" ]]; then
        YQ_EXPR=".inference.loader.dataset.ocean.merge[0].file_pattern = \"$INF_COUPLED_O\""
    fi
    if [[ "$INF_O" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .inference.loader.dataset.ocean.merge[1].file_pattern = \"$INF_O\""
    fi
    if [[ "$INF_COUPLED_A" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .inference.loader.dataset.atmosphere.merge[0].file_pattern = \"$INF_COUPLED_A\""
    fi
    if [[ "$INF_A" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .inference.loader.dataset.atmosphere.merge[1].file_pattern = \"$INF_A\""
    fi

    # Handle training datasets
    if [[ "$TRAIN_COUPLED_O" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .train_loader.dataset[0].ocean.merge[0].file_pattern = \"$TRAIN_COUPLED_O\""
    fi
    if [[ "$TRAIN_O" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .train_loader.dataset[0].ocean.merge[1].file_pattern = \"$TRAIN_O\""
    fi
    if [[ "$TRAIN_COUPLED_A" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .train_loader.dataset[0].atmosphere.merge[0].file_pattern = \"$TRAIN_COUPLED_A\""
    fi
    if [[ "$TRAIN_A" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .train_loader.dataset[0].atmosphere.merge[1].file_pattern = \"$TRAIN_A\""
    fi

    # Handle validation datasets
    if [[ "$VAL_COUPLED_O" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .validation_loader.dataset[0].ocean.merge[0].file_pattern = \"$VAL_COUPLED_O\""
    fi
    if [[ "$VAL_O" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .validation_loader.dataset[0].ocean.merge[1].file_pattern = \"$VAL_O\""
    fi
    if [[ "$VAL_COUPLED_A" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .validation_loader.dataset[0].atmosphere.merge[0].file_pattern = \"$VAL_COUPLED_A\""
    fi
    if [[ "$VAL_A" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .validation_loader.dataset[0].atmosphere.merge[1].file_pattern = \"$VAL_A\""
    fi

    # Handle coupled steps
    if [[ -n "$N_COUPLED_STEPS" && "$N_COUPLED_STEPS" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .inference.n_coupled_steps = $N_COUPLED_STEPS"
    fi

    # Handle coupled memory
    if [[ -n "$COUPLED_STEPS_IN_MEMORY" && "$COUPLED_STEPS_IN_MEMORY" != "null" ]]; then
        if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
        YQ_EXPR="$YQ_EXPR .inference.coupled_steps_in_memory = $COUPLED_STEPS_IN_MEMORY"
    fi

    # Handle train subset
    if [[ "$TRAIN_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .train_loader.dataset[0].ocean.merge[].subset = load(\"$EXPERS_FILE\").train.subset |
            .train_loader.dataset[0].atmosphere.merge[].subset = load(\"$EXPERS_FILE\").train.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.train_loader.dataset[0].ocean.merge[].subset) |
            del(.train_loader.dataset[0].atmosphere.merge[].subset)"
    fi

    # Handle validation subset
    if [[ "$VAL_SUBSET" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .validation_loader.dataset[0].ocean.merge[].subset = load(\"$EXPERS_FILE\").validation.subset |
            .validation_loader.dataset[0].atmosphere.merge[].subset = load(\"$EXPERS_FILE\").validation.subset"
    else
        YQ_EXPR="$YQ_EXPR | del(.validation_loader.dataset[0].ocean.merge[].subset) |
            del(.validation_loader.dataset[0].atmosphere.merge[].subset)"
    fi

    # Handle times
    if [[ "$TIMES" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .inference.loader.start_indices.times = load(\"$EXPERS_FILE\").inference.times"
    fi

    # Handle logging
    if [[ -n "$LOGGING_ENTITY" && "$LOGGING_ENTITY" != "null" ]]; then
        YQ_EXPR="$YQ_EXPR | .logging.entity = \"$LOGGING_ENTITY\""
    fi

    # Apply all modifications in a single yq call
    yq -i "$YQ_EXPR" "$OUTPUT_PATH"
}

# Populate evaluation configs from experiment metadata (ocean/atmos)
# Args:
#   $1 - EXPERIMENT_DIR (directory where configs will be created)
#   $2 - EXPERIMENT_NAME (experiment file name without .yaml extension)
#   $3 - TEMPLATE_TYPE (ocean or atmos)
populate_eval_configs_from_experiment() {
    local EXPERIMENT_DIR="$1"
    local EXPERIMENT_NAME="$2"
    local TEMPLATE_TYPE="$3"
    local DATASETS_YAML="$REPO_ROOT/job_runner/datasets.yaml"
    local EXPERS_FILE="$REPO_ROOT/job_runner/expers/${EXPERIMENT_NAME}.yaml"
    local TEMPLATE_PATH="$REPO_ROOT/job_runner/config_templates/eval_uncoupled.yaml"

    # Pre-load all common data in one batch
    local COMMON_JSON=$(yq eval -o json '{
        "eval_keys": [keys[] | select(test("^eval_"))],
        "inference_group": (.datasets.inference // .datasets.validation // .datasets.train),
        "eval_group": (.datasets.eval // .datasets.inference // .datasets.validation // .datasets.train),
        "logging_entity": .logging.entity
    }' "$EXPERS_FILE")

    local EVAL_KEYS=$(echo "$COMMON_JSON" | jq -r '.eval_keys[]')
    if [[ -z "$EVAL_KEYS" ]]; then
        echo "  No eval_* sections found in experiment preset" >&2
        return 0
    fi

    local EVAL_GROUP=$(echo "$COMMON_JSON" | jq -r '.eval_group')
    local LOGGING_ENTITY=$(echo "$COMMON_JSON" | jq -r '.logging_entity')

    # Pre-load dataset paths (same for all eval configs)
    if [[ "$TEMPLATE_TYPE" == "ocean" ]]; then
        local DATASETS_JSON=$(yq eval -o json '{
            "eval_coupled": .'"${EVAL_GROUP}"'.coupled_ocean,
            "eval_zarr": .'"${EVAL_GROUP}"'.ocean
        }' "$DATASETS_YAML")
    else  # atmos
        local DATASETS_JSON=$(yq eval -o json '{
            "eval_coupled": .'"${EVAL_GROUP}"'.coupled_sea_ice,
            "eval_zarr": .'"${EVAL_GROUP}"'.atmos
        }' "$DATASETS_YAML")
    fi

    local EVAL_COUPLED=$(echo "$DATASETS_JSON" | jq -r '.eval_coupled')
    local EVAL_ZARR=$(echo "$DATASETS_JSON" | jq -r '.eval_zarr')

    # Pre-load all eval-specific data in one batch
    local EVAL_DATA_JSON=$(yq eval -o json 'to_entries | map(select(.key | test("^eval_"))) | map({
        "key": .key,
        "n_forward_steps": .value.'"${TEMPLATE_TYPE}"'.n_forward_steps,
        "forward_steps_in_memory": .value.'"${TEMPLATE_TYPE}"'.forward_steps_in_memory,
        "times": .value.times
    })' "$EXPERS_FILE")

    # Iterate over each eval_* key
    while IFS= read -r EVAL_KEY; do
        if [[ -z "$EVAL_KEY" ]]; then
            continue
        fi

        # Extract the name part (e.g., "ICx1" from "eval_ICx1")
        local EVAL_NAME="${EVAL_KEY#eval_}"
        local OUTPUT_PATH="$EXPERIMENT_DIR/evaluator-config-${EVAL_NAME}.yaml"

        echo "  - Creating evaluator-config-${EVAL_NAME}.yaml" >&2

        cp "$TEMPLATE_PATH" "$OUTPUT_PATH"

        # Extract eval-specific data for this eval_key
        local EVAL_SPECIFIC=$(echo "$EVAL_DATA_JSON" | jq -r '.[] | select(.key == "'"$EVAL_KEY"'")')
        local N_FORWARD_STEPS=$(echo "$EVAL_SPECIFIC" | jq -r '.n_forward_steps')
        local FORWARD_STEPS_IN_MEMORY=$(echo "$EVAL_SPECIFIC" | jq -r '.forward_steps_in_memory')
        local TIMES=$(echo "$EVAL_SPECIFIC" | jq -r '.times')

        # Build batched yq expression
        local YQ_EXPR=""

        # Handle datasets (same for all evals)
        if [[ "$EVAL_COUPLED" != "null" ]]; then
            YQ_EXPR=".loader.dataset.merge[0].file_pattern = \"$EVAL_COUPLED\""
        fi
        if [[ "$EVAL_ZARR" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .loader.dataset.merge[1].file_pattern = \"$EVAL_ZARR\""
        fi

        # Handle forward steps
        if [[ -n "$N_FORWARD_STEPS" && "$N_FORWARD_STEPS" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .n_forward_steps = $N_FORWARD_STEPS"
        fi

        # Handle forward steps in memory
        if [[ -n "$FORWARD_STEPS_IN_MEMORY" && "$FORWARD_STEPS_IN_MEMORY" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .forward_steps_in_memory = $FORWARD_STEPS_IN_MEMORY"
        fi

        # Handle times
        if [[ "$TIMES" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .loader.start_indices.times = load(\"$EXPERS_FILE\").${EVAL_KEY}.times"
        fi

        # Handle logging
        if [[ -n "$LOGGING_ENTITY" && "$LOGGING_ENTITY" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .logging.entity = \"$LOGGING_ENTITY\""
        fi

        # Apply all modifications in a single yq call
        if [[ -n "$YQ_EXPR" ]]; then
            yq -i "$YQ_EXPR" "$OUTPUT_PATH"
        fi
    done <<< "$EVAL_KEYS"
}

# Populate coupled evaluation configs from experiment metadata
# Args:
#   $1 - EXPERIMENT_DIR (directory where configs will be created)
#   $2 - EXPERIMENT_NAME (experiment file name without .yaml extension)
populate_eval_coupled_configs_from_experiment() {
    local EXPERIMENT_DIR="$1"
    local EXPERIMENT_NAME="$2"
    local DATASETS_YAML="$REPO_ROOT/job_runner/datasets.yaml"
    local EXPERS_FILE="$REPO_ROOT/job_runner/expers/${EXPERIMENT_NAME}.yaml"
    local TEMPLATE_PATH="$REPO_ROOT/job_runner/config_templates/eval_coupled.yaml"

    # Pre-load all common data in one batch
    local COMMON_JSON=$(yq eval -o json '{
        "eval_keys": [keys[] | select(test("^eval_"))],
        "inference_group": (.datasets.inference // .datasets.validation // .datasets.train),
        "eval_group": (.datasets.eval // .datasets.inference // .datasets.validation // .datasets.train),
        "logging_entity": .logging.entity
    }' "$EXPERS_FILE")

    local EVAL_KEYS=$(echo "$COMMON_JSON" | jq -r '.eval_keys[]')
    if [[ -z "$EVAL_KEYS" ]]; then
        echo "  No eval_* sections found in experiment preset" >&2
        return 0
    fi

    local EVAL_GROUP=$(echo "$COMMON_JSON" | jq -r '.eval_group')
    local LOGGING_ENTITY=$(echo "$COMMON_JSON" | jq -r '.logging_entity')

    # Pre-load dataset paths (same for all eval configs)
    local DATASETS_JSON=$(yq eval -o json '{
        "eval_coupled_o": .'"${EVAL_GROUP}"'.coupled_ocean,
        "eval_o": .'"${EVAL_GROUP}"'.ocean,
        "eval_coupled_a": .'"${EVAL_GROUP}"'.coupled_atmos,
        "eval_a": .'"${EVAL_GROUP}"'.atmos
    }' "$DATASETS_YAML")

    local EVAL_COUPLED_O=$(echo "$DATASETS_JSON" | jq -r '.eval_coupled_o')
    local EVAL_O=$(echo "$DATASETS_JSON" | jq -r '.eval_o')
    local EVAL_COUPLED_A=$(echo "$DATASETS_JSON" | jq -r '.eval_coupled_a')
    local EVAL_A=$(echo "$DATASETS_JSON" | jq -r '.eval_a')

    # Pre-load all eval-specific data in one batch
    local EVAL_DATA_JSON=$(yq eval -o json 'to_entries | map(select(.key | test("^eval_"))) | map({
        "key": .key,
        "n_coupled_steps": .value.coupled.n_coupled_steps,
        "coupled_steps_in_memory": .value.coupled.coupled_steps_in_memory,
        "subset": .value.coupled.subset,
        "times": .value.times
    })' "$EXPERS_FILE")

    # Iterate over each eval_* key
    while IFS= read -r EVAL_KEY; do
        if [[ -z "$EVAL_KEY" ]]; then
            continue
        fi

        # Extract the name part (e.g., "ICx1" from "eval_ICx1")
        local EVAL_NAME="${EVAL_KEY#eval_}"
        local OUTPUT_PATH="$EXPERIMENT_DIR/evaluator-config-${EVAL_NAME}.yaml"

        echo "  - Creating evaluator-config-${EVAL_NAME}.yaml" >&2

        cp "$TEMPLATE_PATH" "$OUTPUT_PATH"

        # Extract eval-specific data for this eval_key
        local EVAL_SPECIFIC=$(echo "$EVAL_DATA_JSON" | jq -r '.[] | select(.key == "'"$EVAL_KEY"'")')
        local N_COUPLED_STEPS=$(echo "$EVAL_SPECIFIC" | jq -r '.n_coupled_steps')
        local COUPLED_STEPS_IN_MEMORY=$(echo "$EVAL_SPECIFIC" | jq -r '.coupled_steps_in_memory')
        local SUBSET=$(echo "$EVAL_SPECIFIC" | jq -r '.subset')
        local TIMES=$(echo "$EVAL_SPECIFIC" | jq -r '.times')

        # Build batched yq expression
        local YQ_EXPR=""

        # Handle datasets (same for all evals)
        if [[ "$EVAL_COUPLED_O" != "null" ]]; then
            YQ_EXPR=".loader.dataset.ocean.merge[0].file_pattern = \"$EVAL_COUPLED_O\""
        fi
        if [[ "$EVAL_O" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .loader.dataset.ocean.merge[1].file_pattern = \"$EVAL_O\""
        fi
        if [[ "$EVAL_COUPLED_A" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .loader.dataset.atmosphere.merge[0].file_pattern = \"$EVAL_COUPLED_A\""
        fi
        if [[ "$EVAL_A" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .loader.dataset.atmosphere.merge[1].file_pattern = \"$EVAL_A\""
        fi

        # Handle coupled steps
        if [[ -n "$N_COUPLED_STEPS" && "$N_COUPLED_STEPS" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .n_coupled_steps = $N_COUPLED_STEPS"
        fi

        # Handle coupled steps in memory
        if [[ -n "$COUPLED_STEPS_IN_MEMORY" && "$COUPLED_STEPS_IN_MEMORY" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .coupled_steps_in_memory = $COUPLED_STEPS_IN_MEMORY"
        fi

        # Handle subset
        if [[ "$SUBSET" != "null" ]]; then
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR .loader.dataset.ocean.merge[].subset = load(\"$EXPERS_FILE\").${EVAL_KEY}.coupled.subset |
                .loader.dataset.atmosphere.merge[].subset = load(\"$EXPERS_FILE\").${EVAL_KEY}.coupled.subset"
        else
            if [[ -n "$YQ_EXPR" ]]; then YQ_EXPR="$YQ_EXPR |"; fi
            YQ_EXPR="$YQ_EXPR del(.loader.dataset.ocean.merge[].subset) |
                del(.loader.dataset.atmosphere.merge[].subset)"
        fi

        # Handle times
        if [[ "$TIMES" != "null" ]]; then
            YQ_EXPR="$YQ_EXPR | .loader.start_indices.times = load(\"$EXPERS_FILE\").${EVAL_KEY}.times"
        fi

        # Handle logging
        if [[ -n "$LOGGING_ENTITY" && "$LOGGING_ENTITY" != "null" ]]; then
            YQ_EXPR="$YQ_EXPR | .logging.entity = \"$LOGGING_ENTITY\""
        fi

        # Apply all modifications in a single yq call
        yq -i "$YQ_EXPR" "$OUTPUT_PATH"
    done <<< "$EVAL_KEYS"
}


