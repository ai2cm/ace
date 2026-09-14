"""Patch a checkpoint to add the total energy budget correction.

Usage: python patch-ckpt-energy-corrector.py /ckpt.tar /ckpt_ec.tar
"""
import sys, torch

src, dst = sys.argv[1], sys.argv[2]
print(f"Loading checkpoint from {src}")
ckpt = torch.load(src, map_location="cpu", weights_only=False)
corrector = ckpt["stepper"]["config"]["corrector"]
print(f"Original corrector: {corrector}")
corrector["total_energy_budget_correction"] = {
    "method": "constant_temperature",
    "constant_unaccounted_heating": 0.0,
}
print(f"Patched corrector: {corrector}")
torch.save(ckpt, dst)
print(f"Saved patched checkpoint to {dst}")
