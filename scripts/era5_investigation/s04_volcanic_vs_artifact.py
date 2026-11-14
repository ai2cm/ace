"""
Investigate whether the 1986 and 1993 jumps in Finding 4 are volcanic signals
(El Chichón 1982, Pinatubo 1991) or reanalysis artifacts.

Key discriminators:
1. Timing: Volcanic effects should appear gradually after eruption, not as sharp
   year-boundary jumps coinciding with ERA5 stream merges.
2. Signature: Volcanic aerosol causes stratospheric warming + drying (sulfate aerosol).
   El Chichón: erupted March 1982, effects peak 1982-1984.
   Pinatubo: erupted June 1991, effects peak 1991-1993.
3. Multi-variable consistency: Real volcanic signals should show correlated changes
   across temperature and water vapor; artifacts may be variable-specific.
4. Sub-annual structure: If volcanic, the signal should emerge mid-year (post-eruption),
   not at Jan 1 stream boundaries.
"""

import numpy as np
import xarray as xr

stats = xr.open_dataset("/tmp/era5_samples/yearly_stats.nc")
years = stats["year"].values


def print_section(title):
    print(f"\n{'='*80}")
    print(title)
    print("=" * 80)


# 1. Detailed timeline around El Chichón (1982) and Pinatubo (1991)
print_section("1. Year-by-year global means around volcanic events")

strat_vars = [
    "air_temperature_0",
    "air_temperature_1",
    "specific_total_water_0",
    "specific_total_water_1",
]

print("\n--- El Chichón (erupted March 28, 1982) ---")
print(f"{'Year':<8}", end="")
for var in strat_vars:
    print(f"{var:>28}", end="")
print()
for y in range(1979, 1992):
    idx = np.where(years == y)[0]
    if len(idx) == 0:
        continue
    i = idx[0]
    print(f"{y:<8}", end="")
    for var in strat_vars:
        val = stats[f"{var}_mean"].values[i]
        print(f"{val:>28.6g}", end="")
    print()

print("\n--- Mt. Pinatubo (erupted June 15, 1991) ---")
print(f"{'Year':<8}", end="")
for var in strat_vars:
    print(f"{var:>28}", end="")
print()
for y in range(1988, 1998):
    idx = np.where(years == y)[0]
    if len(idx) == 0:
        continue
    i = idx[0]
    print(f"{y:<8}", end="")
    for var in strat_vars:
        val = stats[f"{var}_mean"].values[i]
        print(f"{val:>28.6g}", end="")
    print()

# 2. Compare jump timing with ERA5 stream boundaries
print_section("2. ERA5 stream boundaries vs volcanic eruption timing")
print("""
ERA5 production streams and merge dates:
  Stream 1: 1940-1979 (merged ~2019)
  Stream 2: 1979-Aug 1992 (merged ~2019)
  Stream 3: Sep 1992-2000 (merged ~2019)

Key: Stream 2→3 boundary is at Sep 1992, very close to Pinatubo's aftermath.
The 1986 date does NOT correspond to a known stream boundary.

El Chichón erupted March 1982 → effects 1982-1984 (well within Stream 2)
Pinatubo erupted June 1991 → effects 1991-1993 (spans Stream 2→3 boundary!)
""")

# 3. Year-over-year changes (sigma) for all years
print_section("3. All large year-over-year jumps in stratospheric variables")
for var in strat_vars:
    means = stats[f"{var}_mean"].values
    diffs = np.diff(means)
    diff_std = np.std(diffs)
    if diff_std == 0:
        continue
    normalized = diffs / diff_std
    # Show all jumps > 1.5 sigma
    big = np.where(np.abs(normalized) > 1.5)[0]
    if len(big) > 0:
        print(f"\n  {var}:")
        for idx in big:
            direction = "↑" if normalized[idx] > 0 else "↓"
            print(
                f"    {years[idx]}→{years[idx+1]}: {normalized[idx]:>+6.2f}σ "
                f"{direction} ({means[idx]:.6g} → {means[idx+1]:.6g})"
            )

# 4. Expected volcanic signatures
print_section("4. Expected vs observed volcanic signatures")
print("""
Expected physical effects of major volcanic eruptions on stratosphere:
  - Sulfate aerosol → stratospheric WARMING (absorption of terrestrial IR + solar UV)
  - Sulfate aerosol → surface COOLING
  - Stratospheric water vapor: complex - initial injection of SO2,
    but aerosol heating can increase tropical upwelling → more water vapor transport
  - Effects peak 1-2 years after eruption, decay over ~3-5 years

El Chichón (1982):
  Expected: air_temperature_0 INCREASE in 1982-1984, then decay

Pinatubo (1991):
  Expected: air_temperature_0 INCREASE in 1991-1993, then decay
  Pinatubo was ~3x larger than El Chichón
""")

# Check if the temperature changes match expected volcanic warming
print("Observed stratospheric temperature changes:")
for eruption_name, eruption_year in [("El Chichón", 1982), ("Pinatubo", 1991)]:
    print(f"\n  {eruption_name} ({eruption_year}):")
    for var in ["air_temperature_0", "air_temperature_1"]:
        means = stats[f"{var}_mean"].values
        pre_idx = np.where(years == eruption_year - 1)[0]
        peak_idx = np.where(years == eruption_year + 1)[0]
        post_idx = np.where(years == eruption_year + 3)[0]
        if len(pre_idx) and len(peak_idx) and len(post_idx):
            pre = means[pre_idx[0]]
            peak = means[peak_idx[0]]
            post = means[post_idx[0]]
            print(
                f"    {var}: pre={pre:.4f}K, peak(+1yr)={peak:.4f}K, "
                f"post(+3yr)={post:.4f}K"
            )
            print(
                f"      Change at peak: {peak-pre:+.4f}K, "
                f"Recovery: {post-peak:+.4f}K"
            )

# 5. Check specific_total_water_0 around 1986
print_section("5. Detailed look at the 1986 specific_total_water_0 jump")
var = "specific_total_water_0"
means = stats[f"{var}_mean"].values
stds = stats[f"{var}_std"].values
print(f"\n  Year-by-year {var} (global mean):")
for y in range(1982, 1992):
    idx = np.where(years == y)[0]
    if len(idx) == 0:
        continue
    i = idx[0]
    marker = " <-- largest jump" if y == 1986 else ""
    print(f"    {y}: mean={means[i]:.6e}, std={stds[i]:.6e}{marker}")

print("""
Note: If this were a volcanic drying signal from El Chichón (1982),
we would expect the drying to begin in 1982-1983 and gradually recover,
NOT a sharp drop 4 years later in 1986. A 4-year lag is inconsistent
with known volcanic aerosol lifetime (~2-3 years for stratospheric aerosol).
However, there ARE secondary dynamical effects that could persist longer.
""")

# 6. Look at the 2000 jump context
print_section("6. Year 2000 specific_total_water_0 jump context")
print("This IS a known ERA5 stream boundary (Stream 3→4 merge at Jan 2000)")
for y in range(1997, 2004):
    idx = np.where(years == y)[0]
    if len(idx) == 0:
        continue
    i = idx[0]
    means_val = stats[f"{var}_mean"].values[i]
    print(f"    {y}: mean={means_val:.6e}")
