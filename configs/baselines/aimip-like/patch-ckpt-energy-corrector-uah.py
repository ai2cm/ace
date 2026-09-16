"""Patch a checkpoint to add total energy budget correction with a specified
constant_unaccounted_heating value.

Usage: python patch-ckpt-energy-corrector-uah.py <src> <dst> <uah_wm2>
"""
import sys, torch

src, dst, uah = sys.argv[1], sys.argv[2], float(sys.argv[3])
print(f"Loading checkpoint from {src}")
ckpt = torch.load(src, map_location="cpu", weights_only=False)
corrector = ckpt["stepper"]["config"]["corrector"]
corrector["total_energy_budget_correction"] = {
    "method": "constant_temperature",
    "constant_unaccounted_heating": uah,
}
print(f"Patched corrector with constant_unaccounted_heating={uah}")
torch.save(ckpt, dst)
print(f"Saved to {dst}")
