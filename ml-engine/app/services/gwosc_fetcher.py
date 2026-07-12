import argparse
import logging

import numpy as np
from gwpy.timeseries import TimeSeries

from app.core.config import settings

logger = logging.getLogger(__name__)

# GWOSC still returns a (NaN-filled) array for gaps where a detector wasn't
# observing, rather than an error, so we have to check for that ourselves.
MAX_NAN_FRACTION = 0.01


def _assert_valid_segment(segment: TimeSeries, detector: str, gps_time: float) -> None:
    nan_fraction = float(np.isnan(segment.value).mean())
    if nan_fraction > MAX_NAN_FRACTION:
        raise ValueError(
            f"{detector} has no valid data ({nan_fraction:.0%} gaps) around GPS {gps_time}. "
            "The detector was likely offline or not in observing mode at this time; "
            "try a different detector for this segment."
        )


def fetch_strain_segment(
    gps_time: float,
    detector: str = settings.default_detector,
    duration: int = settings.default_duration_seconds,
) -> TimeSeries:
    half_window = duration / 2
    start = gps_time - half_window
    end = gps_time + half_window

    logger.info(
        "Fetching %ss of %s strain data around GPS %s [%s, %s]",
        duration,
        detector,
        gps_time,
        start,
        end,
    )

    segment = TimeSeries.fetch_open_data(detector, start, end, host=settings.gwosc_host)
    _assert_valid_segment(segment, detector, gps_time)
    return segment


def fetch_strain_as_arrays(
    gps_time: float,
    detector: str = settings.default_detector,
    duration: int = settings.default_duration_seconds,
):
    segment = fetch_strain_segment(gps_time, detector, duration)
    return {
        "strain": segment.value,
        "sample_rate": float(segment.sample_rate.value),
        "t0": float(segment.t0.value),
        "detector": detector,
    }


def fetch_whitened_strain_as_arrays(
    gps_time: float,
    detector: str = settings.default_detector,
    duration: int = settings.default_duration_seconds,
    psd_padding: float = 16.0,
):
    """Fetch strain with extra context on each side to estimate a stable PSD,
    whiten against it, then crop back to the requested window.

    Raw LIGO strain is dominated by low-frequency seismic/thermal drift many
    orders of magnitude larger than the broadband noise floor and any
    transient signal, so it looks almost flat when plotted directly.
    Whitening flattens the noise spectrum, which is why virtually every GW
    analysis (including GWOSC's own tutorials) whitens before inspecting or
    modeling a segment. This is what both the visualizers and the denoising
    model operate on.
    """
    fetch_duration = duration + 2 * psd_padding
    segment = fetch_strain_segment(gps_time, detector, fetch_duration)

    whitened = segment.whiten()
    window_start = gps_time - duration / 2
    window_end = gps_time + duration / 2
    cropped = whitened.crop(window_start, window_end)

    return {
        "strain": cropped.value,
        "sample_rate": float(cropped.sample_rate.value),
        "t0": float(cropped.t0.value),
        "detector": detector,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a raw LIGO/Virgo strain segment from GWOSC around a GPS timestamp."
    )
    parser.add_argument("gps_time", type=float, help="Central GPS timestamp of the segment")
    parser.add_argument("--detector", default=settings.default_detector, help="Detector code, e.g. H1, L1, V1")
    parser.add_argument(
        "--duration", type=int, default=settings.default_duration_seconds, help="Total segment duration in seconds"
    )
    parser.add_argument("--output", help="Optional path to save the segment as a .gwf frame file")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()

    segment = fetch_strain_segment(args.gps_time, args.detector, args.duration)

    print(f"Fetched {len(segment)} samples at {segment.sample_rate} for {args.detector}")
    print(f"Segment span: [{segment.t0}, {segment.t0 + segment.duration}]")

    if args.output:
        segment.write(args.output)
        print(f"Saved segment to {args.output}")


if __name__ == "__main__":
    main()
