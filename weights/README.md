# Included inference weights

These compact PyTorch checkpoints are required by the default inference pipeline.
They were trained on source-disjoint 180-DPI TuxGuitar 2.0.1 and Guitar Pro
8.1.2.37 `score_tab`/`tab_only` renders. Test songs are excluded from domain
adaptation data.

| File | Role | Size | SHA-256 |
| --- | --- | ---: | --- |
| `atomic_symbol_cnn.pt` | printed atomic symbol/time-signature classifier | 0.85 MiB | `4A3213B7CCF74AB4CCBEB326A95612E5ED1892A8D826332E730590A2FC8CE03F` |
| `fret_token_cnn.pt` | event-conditioned blank/X/fret 0-36 classifier | 0.51 MiB | `2E4219CF7AE0D7CBF7B55C16794612C5FEE163FE2410033A735CC0EC8DC8CE2D` |
| `pick_stroke_context_cnn.pt` | up/down pick-stroke event context override | 3.38 MiB | `6FD5AD00625F2B36F7EBF359FF4D7BAF53488C7968F4860587F05FFA38D459DB` |
| `rhythm_context_cnn.pt` | event rhythm, dot, rest and tuplet context | 3.41 MiB | `1000264CD048DC5E59ABA3DBF18D6E22ADAB6E648A731C2A7EE3F69B14682DD4` |
| `score_event_locator.pt` | x-axis score event locator | 0.65 MiB | `FCC68401296689F0D86E802D9BF455C18FB241D3FA99DBFB44F17D571C2F0E8A` |
| `tab_event_locator.pt` | pure-TAB x-axis note/rest event locator | 0.65 MiB | `B0B2BED82EFE28326849C2A365965429A24DEA5954260705615AF408AFD634A0` |
| `tab_rhythm_context_cnn.pt` | pure-TAB voice/rhythm/rest/dot/tuplet context | 3.41 MiB | `2EFCEB9E2EAEC785DD14438CE80EA3375B743F6372ADD56F0F0564CCF0EF52EB` |
| `tab_symbol_detector.pt` | TAB fret-number/X detector | 2.29 MiB | `FDD4313D2692D978DF38C201287127C12B13C88026180A746F07EC604EF5168D` |
| `tab_technique_context_cnn.pt` | pure-TAB multi-label playing-technique context | 3.38 MiB | `CFB67BA44B610CB689B93B21675FAA2ACB8B806E3922EACBB57924E73C86F9EE` |
| `tab_tie_context_cnn.pt` | pure-TAB tie-presence and string relation context | 3.43 MiB | `64D9A6E530391B53D945EF71D2F7FA3BCC58E9809CAF8E42B9F79C19F864869C` |
| `technique_context_cnn.pt` | multi-label playing-technique context; unsafe score+TAB hammer output disabled | 3.38 MiB | `E8DADA782F84ADBD2E17061F3153BEEE2A34B60906879B28D1CEB309E408FF9A` |
| `tie_context_cnn.pt` | tie-presence and tie-relation context | 3.43 MiB | `0F0201533824F6C59377C7870329DB08C8E8B826D25D8A997E6A727F036DF57A` |

After training and validating replacement models, run
`scripts/promote_models.ps1` to copy the selected checkpoints here.

The twelve checkpoints total 28.81 MiB. Score+TAB and pure-TAB use separate
event/rhythm/tie/technique models; the atomic symbol and TAB digit/X models are shared.
