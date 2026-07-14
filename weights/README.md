# Included inference weights

These compact PyTorch checkpoints are required by the default inference pipeline.
They were trained on 180-DPI TuxGuitar 2.0.1 `score_tab`/`tab_only` renders.

| File | Role | Size | SHA-256 |
| --- | --- | ---: | --- |
| `atomic_symbol_cnn.pt` | printed atomic symbol/time-signature classifier | 0.85 MiB | `4A3213B7CCF74AB4CCBEB326A95612E5ED1892A8D826332E730590A2FC8CE03F` |
| `pick_stroke_context_cnn.pt` | up/down pick-stroke event context override | 3.38 MiB | `6FD5AD00625F2B36F7EBF359FF4D7BAF53488C7968F4860587F05FFA38D459DB` |
| `rhythm_context_cnn.pt` | event rhythm, dot, rest and tuplet context | 3.42 MiB | `16704E26BCBC7A22B48325D0359CF08A1BB0112B42CE5E96778425DD39566428` |
| `score_event_locator.pt` | x-axis score event locator | 0.65 MiB | `FCC68401296689F0D86E802D9BF455C18FB241D3FA99DBFB44F17D571C2F0E8A` |
| `tab_event_locator.pt` | pure-TAB x-axis note/rest event locator | 0.65 MiB | `B0B2BED82EFE28326849C2A365965429A24DEA5954260705615AF408AFD634A0` |
| `tab_rhythm_context_cnn.pt` | pure-TAB voice/rhythm/rest/dot/tuplet context | 3.42 MiB | `B77B1C576531023FF56E203C09F1A9BC31063065352EDDF6D1CAD743C6B50610` |
| `tab_symbol_detector.pt` | TAB fret-number/X detector | 2.29 MiB | `FDD4313D2692D978DF38C201287127C12B13C88026180A746F07EC604EF5168D` |
| `tab_technique_context_cnn.pt` | pure-TAB multi-label playing-technique context | 3.38 MiB | `5DF0942BB9737A3C528087E00729C20DE5531DBAA8527FE8F37EFE28D73CB7F5` |
| `tab_tie_context_cnn.pt` | pure-TAB tie-presence and string relation context | 3.43 MiB | `64D9A6E530391B53D945EF71D2F7FA3BCC58E9809CAF8E42B9F79C19F864869C` |
| `technique_context_cnn.pt` | multi-label playing-technique context | 3.38 MiB | `DE00A866E7C5B7D2E05E5B99F706317D27C702610070A554A26F800D53B3D877` |
| `tie_context_cnn.pt` | tie-presence and tie-relation context | 3.43 MiB | `8CB350816935EEBD3FB58EDC5E777AF72D871EFEE45D14465F29E57179F327E0` |

After training and validating replacement models, run
`scripts/promote_models.ps1` to copy the selected checkpoints here.

The eleven checkpoints total 28.29 MiB. Score+TAB and pure-TAB use separate
event/rhythm/tie/technique models; the atomic symbol and TAB digit/X models are shared.
