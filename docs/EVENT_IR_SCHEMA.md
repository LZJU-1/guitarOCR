# GuitarOCR Event IR

`guitarocr.pipeline.infer_tuxguitar_score_tab_page` writes `score_ir.json`. It is the current
boundary between image recognition and later musical validation/GP writing.
It deliberately keeps unknown values as `null` instead of inventing defaults.

## Structure

```json
{
  "schema_version": "1.0",
  "tracks": [
    {
      "track": 1,
      "string_count": 6,
      "string_tuning_midi": null,
      "capo": null,
      "tempo_quarter": null,
      "measures": [
        {
          "number": 1,
          "page_measure_index": 1,
          "system_index": 0,
          "time_signature": [4, 4],
          "time_signature_source": "printed",
          "printed_time_signature": {
            "numerator": 4,
            "denominator": 4,
            "confidence": 0.99,
            "bbox": [250.0, 410.0, 19.0, 64.0]
          },
          "events": [
            {
              "order": 0,
              "x": 410.35,
              "locator_confidence": 0.97,
              "tab_match": "matched",
              "tab_x_delta": 0.20,
              "voices": [
                {
                  "voice": 0,
                  "state": "note",
                  "duration_value": 4,
                  "dot": "none",
                  "division": "1:1",
                  "onset": {"text": "0/1", "whole_notes": 0.0},
                  "duration_fraction": {"text": "1/4", "whole_notes": 0.25},
                  "end": {"text": "1/4", "whole_notes": 0.25},
                  "confidence": {},
                  "candidates": {}
                }
              ],
              "notes": [
                {
                  "string": 2,
                  "fret": 2,
                  "voice": 0,
                  "source": "printed_tab",
                  "tie_in": false,
                  "tie_out": true,
                  "dead": false,
                  "effects": {
                    "palm_mute": true,
                    "slide": false,
                    "vibrato": false
                  }
                }
              ],
              "technique_prediction": {
                "positive": {"palm_mute": true},
                "probabilities": {"palm_mute": 0.997}
              },
              "tie_relation": {
                "visual_probability": 0.99,
                "score_note_count": 2,
                "attacked_tab_note_count": 1,
                "missing_score_note_count": 1,
                "candidate": true,
                "status": "unresolved_partial_or_nonadjacent"
              }
            }
          ],
          "orphan_tab_events": [],
          "rhythm_audit": {
            "status": "audited",
            "capacity": {"text": "1/1", "whole_notes": 1.0},
            "voices": {
              "voice_0": {
                "status": "exact",
                "total": {"text": "1/1", "whole_notes": 1.0},
                "delta": {"text": "0/1", "whole_notes": 0.0},
                "correction_proposal": null
              }
            }
          }
        }
      ]
    }
  ]
}
```

## Association rules

- Standard-score events define the event sequence and rhythm crops.
- TAB digits/X within 0.6 TAB-line spacing are attached to the nearest score event.
- Notes are assigned to a voice only when exactly one predicted voice has state
  `note`; otherwise `voice` remains `null`.
- A score-only event may be a rest, a tied continuation with no redrawn TAB
  digit, or a recognition error. These cases are not silently collapsed.
- An unmatched TAB event is retained in `orphan_tab_events`.
- `number` stays `null` for single-page inference unless
  `--measure-number-offset` is supplied.
- Printed time signatures are recognized from page pixels. The most recent
  printed value is carried through following measures and pages;
  `time_signature_source` records `printed`, `carried`, or `unknown`.
- Duration arithmetic uses exact rational whole-note fractions. A non-exact
  voice receives an auditable `correction_proposal` when available. The
  measure solver may select a candidate only when the configured probability
  gate passes; a unique-measure closure is limited to one non-destructive
  candidate that exactly fills an otherwise invalid measure. Selection reason
  and original CNN candidates remain in `rhythm_audit`.
- Each active predicted voice receives rational `onset`, `duration_fraction`,
  and `end` values. They are derived from that voice's original CNN sequence;
  an earlier error can therefore shift later onsets and will be visible in the
  audit instead of being hidden.
- Tie recognition first produces an event-level `tie_relation`. A visual arc is
  accepted as a candidate only when the predicted score-note count exceeds the
  attacked notes printed in TAB. This rejects visually similar slurs and
  hammer/pull curves.
- Adjacent full-event continuations and constrained partial continuations can
  be resolved by matching missing score-note positions to preceding TAB notes.
  Copied continuation notes use
  `source: tie_continuation`, set `tie_in`, update the source note's `tie_out`,
  and create a top-level `tie_edges` record. Ambiguous, non-adjacent and
  unsupported cross-system candidates remain explicit rather than guessed.
- `technique_prediction` stores the event-context CNN probabilities and
  thresholded labels. Each resolved note receives an `effects` mapping; `X`
  also sets `dead`. The exporter uses this mapping and reports any GP5-format
  downgrade. Beat-level `pick_up`/`pick_down` come from a dedicated override
  checkpoint; `pick_sequence_resolution` records rest suppression, direction
  conflict resolution, or a probability-supported single-hole fill in strict
  alternate picking.
- `guitarocr.pipeline.infer_tuxguitar_score_tab_document` accepts a PDF or ordered page images,
  assigns document-level measure numbers, carries time signatures, and writes
  one `document_score_ir.json` plus per-page IR and overlays. PDF pages retain
  `source_pdf` and `pdf_page` provenance and are rendered at the model's fixed
  180 DPI.

## Current boundary

The IR contains event order, rhythm, visibly printed or tie-recovered
string/fret values, X/dead semantics, time signatures, recognized tempo,
document-level measure numbers, measure-duration audits, tie edges, and
event/note technique labels. It is the supported input to the current primary-
voice GP5 writer. It is not a lossless container for title/key-signature/page
layout metadata, complete multi-track/multi-voice structure, or every ambiguous
cross-system relation; unknown tuning/capo values remain explicit until export
defaults are chosen and reported.
