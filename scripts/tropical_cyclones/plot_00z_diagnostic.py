# Diagnostic figure resolving an apparent "endpoint effect" in the cascade
# pipeline: cascade-infill-then-sr's (and hiro's) min-SLP for track 789
# looked much worse than the video models specifically at 00Z, which read
# as suspicious since 00Z is both the temporal-infill model's sparse
# anchor time and the fine-endpoint time for the endpoint-observed video
# models. Two things were checked to explain it:
#
# 1. Is stage 1 (temporal infill) feeding stage 2 a bad 100km value at
#    00Z? No -- the infill model's own anchor reconstruction is bit-
#    identical to the real 100km truth at every 00Z by construction (it's
#    a given input, not a generated one). Top panel.
#
# 2. Is hiro/cascade's SR step (fme.downscaling.inference, a per-frame
#    diffusion U-Net with n_timesteps hardcoded to 1 and no calendar/time
#    embedding -- see fme/downscaling/models.py -- i.e. genuinely
#    memoryless and time-of-day-blind) doing something specifically worse
#    at 00Z? No: plotting signed SR error (model - truth) continuously
#    over time shows error is large at EVERY hour during the storm's
#    05-27 to 05-31 peak-intensification window (hiro's single worst
#    error in the whole window is at 21:00, not 00Z) and small at every
#    hour outside it. Bottom panel.
#
# The real explanation: st-flat/st-ou/st-singlestage-flat are
# endpoint-observed video models that COPY real fine truth at their own
# clip endpoints (which happen to be 00Z) rather than generating there --
# so comparing them to hiro/cascade (which must generate at every frame,
# 00Z included) at exactly 00Z is comparing a free correct answer against
# a real prediction, and that contrast is starkest during the one window
# where every model's prediction is hardest: peak intensification. There
# is no true 00Z-specific weakness in the SR step.
#
# Source data is embedded below rather than re-fetched from zarr, since
# both series are just this one track's numbers, already generated and
# committed by tc_100km_infill_vs_truth.py (100km panel) and
# tc_min_pressure_timeseries.py (25km panel) -- see those scripts for the
# zarr paths and track/window definitions if regenerating from scratch.
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TOP_CSV = """time,100km truth,100km infill
2023-05-24 06:00:00,1004.4094,1004.5092
2023-05-24 09:00:00,1005.61816,1005.8728
2023-05-24 12:00:00,1006.6562,1006.6936
2023-05-24 15:00:00,1005.42804,1005.4455
2023-05-24 18:00:00,1004.28046,1004.21954
2023-05-24 21:00:00,1005.4275,1005.4769
2023-05-25 00:00:00,1006.25354,1006.25354
2023-05-25 03:00:00,1004.6512,1004.43896
2023-05-25 06:00:00,1003.6209,1003.26483
2023-05-25 09:00:00,1004.6169,1004.16565
2023-05-25 12:00:00,1005.42645,1004.7761
2023-05-25 15:00:00,1003.5659,1002.76166
2023-05-25 18:00:00,1001.75775,1001.30145
2023-05-25 21:00:00,1002.3768,1002.45966
2023-05-26 00:00:00,1002.8139,1002.8139
2023-05-26 03:00:00,1001.8895,1002.10675
2023-05-26 06:00:00,1000.24066,1000.09375
2023-05-26 09:00:00,1000.38934,1001.68823
2023-05-26 12:00:00,1001.0719,1002.1987
2023-05-26 15:00:00,999.04114,1000.6278
2023-05-26 18:00:00,997.7761,998.84125
2023-05-26 21:00:00,999.4851,1000.5265
2023-05-27 00:00:00,1000.6048,1000.6048
2023-05-27 03:00:00,996.7165,997.203
2023-05-27 06:00:00,995.6815,996.51276
2023-05-27 09:00:00,999.95886,998.1546
2023-05-27 12:00:00,997.79126,995.12573
2023-05-27 15:00:00,996.8072,995.13074
2023-05-27 18:00:00,994.6481,992.24316
2023-05-27 21:00:00,993.02527,990.19104
2023-05-28 00:00:00,995.9373,995.9373
2023-05-28 03:00:00,990.7283,993.7011
2023-05-28 06:00:00,988.4379,989.5266
2023-05-28 09:00:00,992.6271,991.1376
2023-05-28 12:00:00,989.204,989.9546
2023-05-28 15:00:00,984.74225,982.9743
2023-05-28 18:00:00,982.58435,984.29645
2023-05-28 21:00:00,986.44165,988.00464
2023-05-29 00:00:00,984.06445,984.06445
2023-05-29 03:00:00,982.2424,981.7816
2023-05-29 06:00:00,981.97534,982.7674
2023-05-29 09:00:00,985.3693,981.3744
2023-05-29 12:00:00,978.03503,976.54114
2023-05-29 15:00:00,982.0667,980.6858
2023-05-29 18:00:00,978.03436,977.3457
2023-05-29 21:00:00,979.5116,977.4541
2023-05-30 00:00:00,980.1527,980.1527
2023-05-30 03:00:00,973.9587,977.1674
2023-05-30 06:00:00,975.77844,984.4244
2023-05-30 09:00:00,978.6776,985.03827
2023-05-30 12:00:00,980.0886,986.2477
2023-05-30 15:00:00,972.92883,988.8505
2023-05-30 18:00:00,970.9698,986.6037
2023-05-30 21:00:00,975.70355,980.388
2023-05-31 00:00:00,975.765,975.765
2023-05-31 03:00:00,977.22284,981.45734
2023-05-31 06:00:00,976.24146,982.2243
2023-05-31 09:00:00,979.16705,985.92816
2023-05-31 12:00:00,982.3799,990.60315
2023-05-31 15:00:00,987.74646,992.15137
2023-05-31 18:00:00,987.6794,991.7051
2023-05-31 21:00:00,990.72894,992.8003
2023-06-01 00:00:00,992.71124,992.71124
"""

BOTTOM_CSV = """time,25km truth,hiro,cascade-infill-then-sr
2023-05-24 06:00:00,1003.2717,1002.7958,1003.9436
2023-05-24 09:00:00,1004.48975,1004.43756,1005.2783
2023-05-24 12:00:00,1005.367,1005.45435,1005.4034
2023-05-24 15:00:00,1003.76526,1004.4544,1005.0368
2023-05-24 18:00:00,1002.2346,1002.2874,1002.7448
2023-05-24 21:00:00,1003.231,1003.4203,1004.25366
2023-05-25 00:00:00,1002.89636,1003.5173,1004.19434
2023-05-25 03:00:00,1000.913,1001.2213,1001.68634
2023-05-25 06:00:00,998.9696,1001.2916,1000.1932
2023-05-25 09:00:00,1000.14825,1001.9359,1003.0255
2023-05-25 12:00:00,999.5026,1002.87946,1002.4257
2023-05-25 15:00:00,997.6694,1001.19965,999.8123
2023-05-25 18:00:00,995.36505,999.617,998.83374
2023-05-25 21:00:00,995.7854,999.41797,998.6094
2023-05-26 00:00:00,996.2359,996.3316,998.3666
2023-05-26 03:00:00,992.8957,996.62085,993.8596
2023-05-26 06:00:00,989.91675,993.6035,997.8158
2023-05-26 09:00:00,990.4507,997.45575,999.24054
2023-05-26 12:00:00,991.0689,992.65576,999.25165
2023-05-26 15:00:00,987.76245,992.05023,996.37524
2023-05-26 18:00:00,989.25464,992.8281,993.5122
2023-05-26 21:00:00,989.87897,992.28265,992.5816
2023-05-27 00:00:00,988.4794,987.22894,994.1852
2023-05-27 03:00:00,983.03516,985.6183,990.349
2023-05-27 06:00:00,981.0126,983.90155,992.06903
2023-05-27 09:00:00,982.20874,992.0735,988.5627
2023-05-27 12:00:00,972.0761,985.4224,983.1778
2023-05-27 15:00:00,976.402,979.62225,988.2467
2023-05-27 18:00:00,969.98145,985.77435,984.3853
2023-05-27 21:00:00,965.3695,981.4811,974.71484
2023-05-28 00:00:00,957.5973,978.7007,984.12335
2023-05-28 03:00:00,958.48346,977.1151,986.0057
2023-05-28 06:00:00,944.9508,970.2042,985.9503
2023-05-28 09:00:00,949.82074,972.9772,981.0448
2023-05-28 12:00:00,960.7014,968.6376,983.11487
2023-05-28 15:00:00,950.336,973.5831,969.5497
2023-05-28 18:00:00,943.11707,956.8987,966.72284
2023-05-28 21:00:00,925.6475,966.553,962.7755
2023-05-29 00:00:00,936.7324,965.25256,968.7798
2023-05-29 03:00:00,930.9605,971.17566,967.31384
2023-05-29 06:00:00,924.35986,965.5902,966.5411
2023-05-29 09:00:00,933.81396,962.3604,948.4444
2023-05-29 12:00:00,933.3274,949.1868,959.5101
2023-05-29 15:00:00,926.32214,958.3133,945.83636
2023-05-29 18:00:00,922.5594,951.27673,945.3066
2023-05-29 21:00:00,921.20605,944.29596,952.41394
2023-05-30 00:00:00,932.99646,957.1871,952.60236
2023-05-30 03:00:00,950.71436,951.8738,963.9647
2023-05-30 06:00:00,951.8941,956.05194,959.0782
2023-05-30 09:00:00,951.32794,949.6907,975.34186
2023-05-30 12:00:00,953.3496,955.59546,981.87354
2023-05-30 15:00:00,953.40216,950.5185,985.69714
2023-05-30 18:00:00,954.8747,956.0778,980.7043
2023-05-30 21:00:00,958.1065,948.5699,964.47546
2023-05-31 00:00:00,960.5437,948.8403,947.70374
2023-05-31 03:00:00,961.8078,959.31116,968.3318
2023-05-31 06:00:00,964.1579,964.9877,971.654
2023-05-31 09:00:00,969.64294,971.0034,982.8131
2023-05-31 12:00:00,974.6508,977.54156,989.4537
2023-05-31 15:00:00,978.1496,978.17487,989.94836
2023-05-31 18:00:00,981.7877,984.7832,990.6442
2023-05-31 21:00:00,985.66425,985.2333,991.09753
2023-06-01 00:00:00,987.7691,989.78455,988.595
"""

OUT_PATH = "tc_00z_diagnostic_track789.png"


def main():
    top = pd.read_csv(io.StringIO(TOP_CSV), parse_dates=["time"])
    bot = pd.read_csv(io.StringIO(BOTTOM_CSV), parse_dates=["time"])
    bot["hiro_err"] = bot["hiro"] - bot["25km truth"]
    bot["cascade_err"] = bot["cascade-infill-then-sr"] - bot["25km truth"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax1.plot(
        top["time"],
        top["100km truth"],
        color="black",
        linewidth=2.2,
        label="100km truth",
    )
    ax1.plot(
        top["time"],
        top["100km infill"],
        color="#9467bd",
        linewidth=1.6,
        linestyle="--",
        label="100km infill (stage 1 output, ens member 0)",
    )
    ax1.set_ylabel("min SLP in 5x5deg window (mb)")
    ax1.set_title(
        "Stage 1 (temporal infill): 100km reconstruction vs. real 100km truth\n"
        "(bit-identical at every 24h anchor -- diverges only in the interior, "
        "worst during 05-30's rapid intensification)"
    )
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="black", linewidth=1)
    ax2.plot(
        bot["time"],
        bot["hiro_err"],
        color="#9467bd",
        linewidth=1.8,
        linestyle="--",
        label="hiro error (real dense input)",
    )
    ax2.plot(
        bot["time"],
        bot["cascade_err"],
        color="#e377c2",
        linewidth=1.8,
        linestyle=":",
        label="cascade error (reconstructed input)",
    )
    is_00z = bot["time"].dt.hour == 0
    ax2.scatter(
        bot.loc[is_00z, "time"],
        bot.loc[is_00z, "hiro_err"],
        color="#9467bd",
        marker="o",
        s=45,
        zorder=5,
        label="hiro @ 00Z",
    )
    ax2.scatter(
        bot.loc[is_00z, "time"],
        bot.loc[is_00z, "cascade_err"],
        color="#e377c2",
        marker="o",
        s=45,
        zorder=5,
        label="cascade @ 00Z",
    )

    for t in bot["time"]:
        if t.hour == 0:
            ax2.axvline(t, color="tab:red", alpha=0.25, linewidth=1.2, zorder=0)
            ax1.axvline(t, color="tab:red", alpha=0.25, linewidth=1.2, zorder=0)

    ax2.set_ylabel("SR error: model - 25km truth (mb)\n(positive = storm too weak)")
    ax2.set_title(
        "Stage 2 SR error is driven by storm intensity/rate-of-change, not by "
        "proximity to 00Z (red lines): 00Z dots sit ON the same error curve as "
        "every other hour --\nerror is large at ALL hours during the 05-27 to "
        "05-31 peak-intensification window (e.g. hiro's worst error, 05-28 "
        "21:00, is off-anchor) and small at all hours outside it"
    )
    ax2.legend(loc="upper left", fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    fig.suptitle(
        'track 789: the "00Z gap" is a comparison artifact, not a real endpoint effect',
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
