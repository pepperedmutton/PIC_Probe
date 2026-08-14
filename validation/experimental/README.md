# Experimental validation data

This directory contains derived numerical data. It does not contain the source PDF or a source image.

## Cenian 2005, Figure 2

The CSV records the measured negative-bias branch for the internally consistent `r_p/lambda_D = 0.26` case.

The source is A. Cenian et al., *Journal of Applied Physics* 97, 123310 (2005). The DOI is [10.1063/1.1938275](https://doi.org/10.1063/1.1938275).

The source gives these conditions:

- Pure, nonflowing argon plasma
- No magnetic field is reported for this experimental case
- Pressure: 1.3 mTorr
- Electron density: `7.15e13 m^-3`
- Electron temperature: `1.9 eV`
- Ion temperature: `0.025 eV`
- Probe radius: `313 micrometers`
- Probe length: `47 mm`
- Debye length: approximately `1.21 mm`
- Probe voltage reference: plasma potential

The CSV current uses the sign in the source figure. Thus, the current is negative in the ion collection region.

Compare it with `PICSimulation.avg_current`. This value is electron-current magnitude minus ion-current magnitude. Do not use `avg_conventional_current`, which reverses the source sign.

The values came from the irregular solid experimental curve in Figure 2. The JSON file records the pixel calibration and the digitization uncertainty.

The digitization uncertainty is not the full experimental uncertainty. The source does not give a point-by-point uncertainty for this curve.

The source gets density and temperature from the same probe characteristic. Thus, this dataset does not give an independent blind validation of these input parameters.

The second case in Figure 2 is not included. Its printed density exponent does not agree with its stated probe-radius to Debye-length ratio.

No second person has independently digitized this dataset. Use it as a first external validation dataset, not as the only production release datum.

The article reports that finite probe-length effects can become important when the probe-bias magnitude is above 50 V. The `-55 V` and `-60 V` values therefore cannot validate an infinite-cylinder model without an end-effect correction or uncertainty bound.

The article's numerical model used outer radii of about 18, 40, and 55 Debye lengths at `-10 V`, `-30 V`, and `-50 V`. These are about 21.8, 48.5, and 66.7 mm for this case. A 12 mm domain is only a pilot domain.
