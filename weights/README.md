# Included inference weights

These compact PyTorch checkpoints are required by the default inference pipeline.
They were trained on 180-DPI TuxGuitar 2.0.1 `score_tab`/`tab_only` renders.

| File | Role | Size | SHA-256 |
| --- | --- | ---: | --- |
| `atomic_symbol_cnn.pt` | printed atomic symbol/time-signature classifier | 0.85 MiB | `4A3213B7CCF74AB4CCBEB326A95612E5ED1892A8D826332E730590A2FC8CE03F` |
| `rhythm_context_cnn.pt` | event rhythm, dot, rest and tuplet context | 3.40 MiB | `7091D621192495C6C3E0D3DCA461CC6C39F8A96C2B026FDC9E72AD91B7BDBEB0` |
| `score_event_locator.pt` | x-axis score event locator | 0.65 MiB | `FCC68401296689F0D86E802D9BF455C18FB241D3FA99DBFB44F17D571C2F0E8A` |
| `tab_symbol_detector.pt` | TAB fret-number/X detector | 2.29 MiB | `1EAC8705753D2B08FC44A64437FC942B00CEDC78EA28F5C4DF172DA3B463BDFB` |
| `tie_context_cnn.pt` | tie-presence and tie-relation context | 3.43 MiB | `75604120FB653ECB5EABB5DE99299BE237DF4892C3399E915DC107A3A7CB92C8` |

After training and validating replacement models, run
`scripts/promote_models.ps1` to copy the selected checkpoints here.
