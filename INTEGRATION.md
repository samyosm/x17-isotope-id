# Integration with the real detector

## What the detector must provide

For every hit:

- bar number
- energy in raw units (ADC or charge)
- timestamp

For every run:

- duration
- which source was placed at which bar (known, used only to check the answer)

Use the same trigger, high voltage and threshold as normal data taking.

## One number per bar: the gain

The model works in MeV, the detector in raw units. Find one gain per bar (MeV per raw unit).

- Put Cs-137 near the bar and take a few minutes of data.
- Read the Compton edge position in raw units. It is 0.477 MeV.
- Gain = 0.477 / edge position.

It only needs to be accurate to about 10%, the range the model was trained on.

## Position

Not needed. The model gets almost all of its information from the energy histogram. Retrain without the position features, and nothing about position has to be calibrated.

## What has to be built

A reader for real data. It does the same as the simulation windows:

1. Convert energy to MeV with the bar's gain.
2. Cut one bar's hits into windows of known duration.
3. Compute the same features as the simulated windows.
4. Give them to the trained model.

The model then returns a probability for each of Cs-137, Co-60, Bi-207, Na-22 and background.

## The lab test

1. Put one known source at one bar. Take a run.
2. Run the model on that run.
3. Compare the answer with the known source.

Repeat for every source and several bars. Report every mismatch. Do not change the simulation to make the results look better.

## Excess identification (current method)

`ExcessAnalyzer` (`src/excess.py`) compares a window with a reference period. Per scintillator it takes the histogram of `sqrt((U-PU)(D-PD))` for pairs with both PMTs at least `[excess] threshold` ADC; the registry counts these in ClickHouse. The excess is the window minus the reference scaled by the number of triggers. For each source (`templates/excess_templates.json`, from the simulation through the measured detector response, see `src/response.py`) and each candidate bar, the expected excess has a free amplitude per distance class, constrained to not grow with distance; the background of each bin is profiled out using both counts (on/off likelihood). `TS` is 2 ln of the likelihood gain over no source. `detection_ts` (30) is set above the largest `TS` of source-free windows tested (about 18); the source is named only when it leads the next by `margin_ts`.

Validated on the 21 October runs: Cs-137 identified next to bar 4; Bi-207 detected but not named; Th-228 and Co-60 not seen; no false detections in 0.5 to 4 h windows without a source. Th is not simulated, Co-60 and Na-22 are too weak to be seen today.
