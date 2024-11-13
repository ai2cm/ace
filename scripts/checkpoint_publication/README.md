## Checkpoint publication

Scripts to prepare checkpoints for uploading to Ai2 Hugging Face. For now, just
strips optimizer state to decrease filesize by 3x and renames to desired
filename. Can then do manual creation/upload to Hugging Face.

The `prep_ace_ckpts.sh` script will download the ACE, ACE-E3SM and ACE2-ERA5 checkpoints,
strip the optimizers out of them, and save under the filenames intended for use
in new Hugging Face model collection, in the `final_checkpoints` directory.
