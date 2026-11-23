"""
Rating bands used for ANALYSIS (reporting results per band, conditioning models).

The dataset was sampled in bands of 200 rating points (100-299, 300-499, ...).
The 2700-2899 band only has about 150 positions (no test positions), which is
too few to report on its own, so every rating from TOP_BAND_START upwards is
put into one top band, "2500+".

This only changes how rows are GROUPED; the dataset files are not changed.
Every script that reports per band should use band_of / band_label from here,
so all results use the same bands.
"""
MIN_RATING = 100
BAND_WIDTH = 200
TOP_BAND_START = 2500


def band_of(rating, min_rating=MIN_RATING, band_width=BAND_WIDTH, top_band_start=TOP_BAND_START):
    """Lower edge of the rating band, e.g. 1234 -> 1100, 2750 -> 2500 (top band)."""
    band = min_rating + (rating - min_rating) // band_width * band_width
    return min(band, top_band_start)


def band_label(band, band_width=BAND_WIDTH, top_band_start=TOP_BAND_START):
    """Readable name: 1100 -> '1100-1299', 2500 -> '2500+'."""
    if band >= top_band_start:
        return f"{top_band_start}+"
    return f"{band}-{band + band_width - 1}"
