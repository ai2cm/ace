import argparse

import wandb


def wandb_to_beaker_experiment(
    project: str, wandb_id: str, entity: str = "ai2cm"
) -> str:
    """Given a wandb run ID, return corresponding beaker experiment ID"""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{wandb_id}")
    return run.config["environment"]["BEAKER_EXPERIMENT_ID"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_id", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--entity", type=str, default="ai2cm")
    args = parser.parse_args()
    experiment_id = wandb_to_beaker_experiment(
        project=args.project, wandb_id=args.wandb_id, entity=args.entity
    )
    # print for capture in bash scripts
    print(experiment_id)
