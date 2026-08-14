# Cross-section inputs for validation

The validation uses three complete LXCat selections retrieved on 14 August 2026:

- Phelps electron-Ar: one effective momentum-transfer process, one total
  excitation process, and one ionization process. This is the primary input
  because Cenian et al. used the Phelps electron data.
- Biagi, Magboltz 8.97: one electron-Ar elastic process, 44 excitation
  processes, and one ionization process. This is a sensitivity input.
- Phelps: the Backscat and Isotropic components for Ar+ in Ar.

The Cenian article cites a 1997 Phelps electron table. The 2026 LXCat
retrieval is from the same named database, but this repository has not shown
that every tabulated value is identical to the 1997 version. Treat this as a
model-data uncertainty until that comparison is complete.

The raw LXCat files are not in this repository. LXCat states that database
contributors retain ownership and that third parties must direct users to
LXCat to obtain data. Download all three selections from the LXCat data center,
accept its terms, and put the files in `.validation_private/lxcat/` with the
names in `lxcat_manifest.json`.

Before a run, verify all SHA-256 values against `lxcat_manifest.json`. A hash
mismatch means that the selected data or its retrieval version is different.

The ion mapping is explicit. The Phelps Backscat component is treated as
resonant charge exchange only for the symmetric Ar+/Ar pair. The Isotropic
component is used by the equal-mass isotropic elastic operator. Set
`CONFIRM_SYMMETRIC_BACKSCATTER_AS_CEX=True` only for this recorded selection.
Strict loading rejects an unconfirmed Backscat mapping.

Sources:

- https://us.lxcat.net/data/
- https://us.lxcat.net/instructions/how_reference.php
- https://us.lxcat.net/instructions/redistribution.php
