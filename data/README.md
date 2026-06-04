# Data

## `nyc_taxi.csv`

New York City taxi demand aggregated into 30-minute buckets between
2014-07-01 and 2015-01-31 (10,320 points).

- **Source:** [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB)
  — `data/realKnownCause/nyc_taxi.csv`
- **Copyright:** © Numenta Inc., distributed under the MIT license.
- **Known anomalies:** five human-labelled events (NYC Marathon, Thanksgiving,
  Christmas, New Year's, the Jan 2015 snowstorm), taken from NAB's
  `labels/combined_windows.json` and encoded in
  [app/dataset.py](../app/dataset.py).

Used here to validate the detector against real, labelled anomalies rather
than only the synthetic simulator. See the "Validation on real data" section
of the project README.
