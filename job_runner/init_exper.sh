#!/bin/bash
# Initialize new experiment directory with pre-configured templates and datasets
# Usage: init_exper.sh --exper <experiment_preset> --name <experiment_name> --template <template_name>

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared library
source "$SCRIPT_DIR/lib_init_exper.sh"

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 --exper <experiment_preset> --name <experiment_name> [--template <template_name>]

Initialize a new experiment directory with pre-configured templates and datasets.

Required Arguments:
  --exper <name>      Experiment preset from job_runner/expers/ (e.g., cm4_piControl)
  --name <name>       Name for the experiment directory (e.g., my_ocean_exper)

Optional Arguments:
  --template <name>   Template config to use (default: all templates)
                      Available templates in job_runner/config_templates/:
                      - ocean   (creates train-config.yaml)
                      - atmos   (creates train-config.yaml)
                      - coupled (creates train-config-template.yaml)
                      - (omit to create all three templates)

Examples:
  # Create all three templates (ocean, atmos, coupled) using the cm4_piControl preset
  $0 --exper cm4_piControl --name my_experiment

  # Create only an ocean experiment using the cm4_piControl preset
  $0 --exper cm4_piControl --name ocean_test --template ocean

  # Create only an atmosphere experiment using the same preset
  $0 --exper cm4_piControl --name atmos_test --template atmos

  # Create only a coupled experiment that trains on 1pctCO2 and evaluates on piControl
  $0 --exper cm4-train_1pctCO2-eval_piControl --name coupled_test --template coupled

Output (uncoupled):
  Creates directory: configs/experiments/<name>/<template>/
  - train-config.yaml            (populated with preset-specific values)
  - finetune-config-template.yaml (populated for finetuning)
  - evaluator-config-*.yaml      (one per eval_* section in preset)
  - training.txt                 (header only, ready for job entries)
  - finetuning.txt               (header only, ready for job entries)
  - experiments.txt              (header only, for tracking launched jobs)
  - resuming.txt                 (header only, for resuming jobs)

Output (coupled):
  Creates directory: configs/experiments/<name>/<template>/
  - train-config-template.yaml   (populated with preset-specific values)
  - finetune-config-template.yaml (populated for finetuning)
  - evaluator-config-*.yaml      (one per eval_* section in preset)
  - pretraining.txt              (header only, ready for job entries)
  - finetuning.txt               (header only, ready for job entries)
  - experiments.txt              (header only, for tracking launched jobs)
  - resuming.txt                 (header only, for resuming jobs)

EOF
}

# Parse command line arguments
EXPERIMENT=""
NAME=""
TEMPLATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --exper)
            EXPERIMENT="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$EXPERIMENT" || -z "$NAME" ]]; then
    echo "Error: Missing required arguments. Run ./job_runner/init_exper.sh --help for info on usage."
    echo
    exit 1
fi

# If no template specified, use all templates
if [[ -z "$TEMPLATE" ]]; then
    TEMPLATES=("ocean" "atmos" "coupled")
else
    TEMPLATES=("$TEMPLATE")
fi

# Get repo root
if [[ -z "${REPO_ROOT:-}" ]]; then
    REPO_ROOT=$(git rev-parse --show-toplevel)
    export REPO_ROOT
fi

# Validate that experiment preset exists
EXPERS_DIR="$REPO_ROOT/job_runner/expers"
EXPERS_FILE="$EXPERS_DIR/${EXPERIMENT}.yaml"

if [[ ! -f "$EXPERS_FILE" ]]; then
    echo "Error: Experiment preset file not found: $EXPERS_FILE"
    echo
    echo "Available experiment presets:"
    ls -1 "$EXPERS_DIR" | grep '\.yaml$' | sed 's/\.yaml$//' || true
    exit 1
fi

# Extract dataset groups for display
DATASETS_YAML="$REPO_ROOT/job_runner/datasets.yaml"
TRAIN_DATASET=$(yq -r ".datasets.train" "$EXPERS_FILE" 2>/dev/null)
VALIDATION_DATASET=$(yq -r ".datasets.validation // \"${TRAIN_DATASET}\"" "$EXPERS_FILE" 2>/dev/null)
INFERENCE_DATASET=$(yq -r ".datasets.inference // \"${VALIDATION_DATASET}\"" "$EXPERS_FILE" 2>/dev/null)
EVAL_DATASET=$(yq -r ".datasets.eval // \"${INFERENCE_DATASET}\"" "$EXPERS_FILE" 2>/dev/null)

# Ensure referenced dataset groups exist in datasets.yaml
for DATASET_GROUP in "$TRAIN_DATASET" "$VALIDATION_DATASET" "$INFERENCE_DATASET" "$EVAL_DATASET"; do
    if [[ -n "$DATASET_GROUP" && "$DATASET_GROUP" != "null" ]] && ! yq -e ".${DATASET_GROUP}" "$DATASETS_YAML" > /dev/null 2>&1; then
        echo "Error: Dataset group '${DATASET_GROUP}' referenced by experiment '${EXPERIMENT}' not found in datasets.yaml"
        exit 1
    fi
done

echo "WARNING: init_exper.sh currently assumes config templates use MergeDatasetConfig" >&2
echo
echo "Initializing experiment directory..."
echo "  Experiment preset: $EXPERIMENT"
echo "  Train dataset group: $TRAIN_DATASET"
echo "  Validation dataset group: $VALIDATION_DATASET"
echo "  Inference dataset group: $INFERENCE_DATASET"
echo "  Experiment name: $NAME"
echo "  Templates: ${TEMPLATES[*]}"
echo

# Process each template
for TEMPLATE in "${TEMPLATES[@]}"; do
    echo "=========================================="
    echo "Processing template: $TEMPLATE"
    echo "=========================================="
    echo

    # Validate that template exists
    TEMPLATE_PATH="$REPO_ROOT/job_runner/config_templates/${TEMPLATE}.yaml"
    if [[ ! -f "$TEMPLATE_PATH" ]]; then
        echo "Error: Template file not found: $TEMPLATE_PATH"
        echo
        echo "Available templates in job_runner/config_templates/:"
        ls -1 "$REPO_ROOT/job_runner/config_templates/" | grep '\.yaml$' | sed 's/\.yaml$//' || true
        exit 1
    fi

    # Create experiment directory
    EXPERIMENT_DIR=$(create_experiment_directory "$NAME" "$TEMPLATE")
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    echo "Created directory: $EXPERIMENT_DIR"

    # Determine output config path
    OUTPUT_CONFIG="$EXPERIMENT_DIR/train-config.yaml"

    # Populate config based on template type
    case "$TEMPLATE" in
        ocean)
            echo "Populating ocean config using preset $EXPERIMENT..."
            if ! populate_ocean_config_from_experiment "$TEMPLATE_PATH" "$OUTPUT_CONFIG" "$EXPERIMENT"; then
                echo "Error: Failed to populate config"
                rmdir "$EXPERIMENT_DIR" 2>/dev/null || true
                exit 1
            fi
            echo "  - Created train-config.yaml using datasets from preset $EXPERIMENT"

            # Create finetuning config
            echo "Creating finetuning config..."
            FINETUNE_TEMPLATE="$REPO_ROOT/job_runner/config_templates/finetune_uncoupled.yaml"
            FINETUNE_OUTPUT="$EXPERIMENT_DIR/finetune-config-template.yaml"
            if populate_finetune_config_from_experiment "$FINETUNE_TEMPLATE" "$FINETUNE_OUTPUT" "$EXPERIMENT" "ocean"; then
                echo "  - Created finetune-config-template.yaml"
            else
                echo "  Warning: Failed to create finetuning config"
            fi

            # Create evaluation configs
            echo "Creating evaluation configs..."
            if populate_eval_configs_from_experiment "$EXPERIMENT_DIR" "$EXPERIMENT" "ocean"; then
                :  # Success - function outputs its own messages
            else
                echo "  Warning: Failed to create evaluation configs"
            fi
            ;;

        atmos)
            echo "Populating atmosphere config using preset $EXPERIMENT..."
            if ! populate_atmos_config_from_experiment "$TEMPLATE_PATH" "$OUTPUT_CONFIG" "$EXPERIMENT"; then
                echo "Error: Failed to populate config"
                rmdir "$EXPERIMENT_DIR" 2>/dev/null || true
                exit 1
            fi
            echo "  - Created train-config.yaml using datasets from preset $EXPERIMENT"

            # Create finetuning config
            echo "Creating finetuning config..."
            FINETUNE_TEMPLATE="$REPO_ROOT/job_runner/config_templates/finetune_uncoupled.yaml"
            FINETUNE_OUTPUT="$EXPERIMENT_DIR/finetune-config-template.yaml"
            if populate_finetune_config_from_experiment "$FINETUNE_TEMPLATE" "$FINETUNE_OUTPUT" "$EXPERIMENT" "atmos"; then
                echo "  - Created finetune-config-template.yaml"
            else
                echo "  Warning: Failed to create finetuning config"
            fi

            # Create evaluation configs
            echo "Creating evaluation configs..."
            if populate_eval_configs_from_experiment "$EXPERIMENT_DIR" "$EXPERIMENT" "atmos"; then
                :  # Success - function outputs its own messages
            else
                echo "  Warning: Failed to create evaluation configs"
            fi
        ;;

    coupled)
            # Coupled configs create train-config-template.yaml (not train-config.yaml)
            OUTPUT_CONFIG="$EXPERIMENT_DIR/train-config-template.yaml"
            echo "Populating coupled config using preset $EXPERIMENT..."
            if ! populate_coupled_config_from_experiment "$TEMPLATE_PATH" "$OUTPUT_CONFIG" "$EXPERIMENT"; then
                echo "Error: Failed to populate config"
                rmdir "$EXPERIMENT_DIR" 2>/dev/null || true
                exit 1
            fi
            echo "  - Created train-config-template.yaml using datasets from preset $EXPERIMENT"

            # Create finetuning config
            echo "Creating finetuning config..."
            FINETUNE_TEMPLATE="$REPO_ROOT/job_runner/config_templates/finetune_coupled.yaml"
            FINETUNE_OUTPUT="$EXPERIMENT_DIR/finetune-config-template.yaml"
            if populate_finetune_coupled_config_from_experiment "$FINETUNE_TEMPLATE" "$FINETUNE_OUTPUT" "$EXPERIMENT"; then
                echo "  - Created finetune-config-template.yaml"
            else
                echo "  Warning: Failed to create finetuning config"
            fi

            # Create evaluation configs
            echo "Creating evaluation configs..."
            if populate_eval_coupled_configs_from_experiment "$EXPERIMENT_DIR" "$EXPERIMENT"; then
                :  # Success - function outputs its own messages
            else
                echo "  Warning: Failed to create evaluation configs"
            fi
            ;;

        *)
            echo "Warning: Template type '$TEMPLATE' not specifically handled"
            echo "  - Copying template as-is (datasets not substituted)"
            cp "$TEMPLATE_PATH" "$OUTPUT_CONFIG"
            ;;
esac

# Create input txt files with headers
echo "Creating input files..."
create_input_txt_files "$EXPERIMENT_DIR" "$TEMPLATE"

# List created files
echo
echo "Successfully created experiment directory with the following files:"
ls -1 "$EXPERIMENT_DIR" | sed 's/^/  - /'

echo
echo "Next steps:"
case "$TEMPLATE" in
        coupled)
            echo "  1. Review the generated train-config-template.yaml"
            echo "  2. Add job entries to pretraining.txt (one per line)"
            echo "     Note: Coupled training requires pre-trained ocean and atmosphere models"
            echo "  3. Run training jobs:"
            echo "     cd $REPO_ROOT"
            echo "     ./job_runner/coupled_train.sh configs/experiments/${NAME}/${TEMPLATE} ."
            ;;
        ocean|atmos)
            echo "  1. Review the generated train-config.yaml"
            echo "  2. Add job entries to training.txt (one per line)"
            echo "  3. Run training jobs:"
            echo "     cd $REPO_ROOT"
            echo "     ./job_runner/uncoupled_train.sh configs/experiments/${NAME}/${TEMPLATE} ."
            ;;
        *)
            echo "  1. Review the generated config file"
            echo "  2. Add job entries to appropriate .txt file"
            echo "  3. Run the corresponding job_runner script"
            ;;
    esac
    echo

    # Relative path for display
    REL_PATH="configs/experiments/${NAME}/${TEMPLATE}"
    echo "  Experiment directory: $REL_PATH"
    echo
done

echo "=========================================="
echo "All templates initialized successfully!"
echo "=========================================="
