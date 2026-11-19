"""Model constants. Every value here is either a documented default or is re-fitted from data."""

POSITIONS = ("top", "jng", "mid", "bot", "sup")

# Shrinkage strength for champion deviations: dev = (mean - league_mean) * n / (n + K_SHRINK).
# Small samples are pulled toward zero so a champion seen 3 times cannot dominate a forecast.
K_SHRINK = 3.0

# Reference value for how many minutes one unit of average draft deviation is worth.
# The model re-fits this from out-of-fold data; the constant is only used for documentation/tests.
DEFAULT_MULTIPLIER = 4.5

LEAGUE_SHRINK = 30.0   # shrinkage for per-league offsets
BIAS_BINS = 5          # bins for the graduated prediction-bias correction
BIAS_SHRINK = 40.0     # shrinkage for each bin's correction

Z80 = 1.2816           # normal quantile for an 80% central interval

# Live nowcast: minutes shaved off the forecast per 1k of gold lead at 15:00.
NOWCAST_SLOPE = 0.8
NOWCAST_LEAD_CLAMP_K = 6.0   # leads beyond this are not extrapolated (linearity is untested out there)

RANDOM_SEED = 7
NOWCAST_REF_LEAD_K = 1.6    # typical lead already priced into the pre-game forecast; refit from data
