# GuitarOCR recognition architecture

## Objective

The target is regular, machine-generated guitar notation exported by software such as TuxGuitar. The first supported input is a paired standard-score and tablature layout. Handwriting is out of scope.

The system must recover musical events, not merely classify visible glyphs. A playable Guitar Pro reconstruction requires measure structure, event timing, duration, voices, string/fret assignments, and selected relationships such as ties.

## Processing hierarchy

```text
page
  -> score systems
  -> paired standard-score / TAB staves
  -> measures
  -> overlapping event proposals
  -> local atomic symbols
  -> cross-event relationships
  -> Event IR
  -> Score IR
  -> GP writer
```

The image is not partitioned into disjoint connected components. Staff lines cross glyphs; noteheads touch stems; beams join multiple events; ties span two events. Every detected object therefore retains its coordinates in the original measure image.

## Two recognition channels

### Local event channel

An event proposal is anchored by TAB digits/X, standard-score noteheads, or a rest. Nearby anchors with approximately the same x coordinate are grouped as one onset. Event crops include padding and may overlap adjacent crops.

The local channel recognizes atomic classes suitable for a compact CNN:

- TAB/time digits `0-9`;
- TAB dead note `X`;
- filled, open, dead-X, and harmonic noteheads;
- whole, half, quarter, eighth, sixteenth, thirty-second, and sixty-fourth rests;
- sharp, flat, and natural accidentals;
- treble and bass clefs;
- augmentation dots.

Digits are atomic classes. Adjacent digit instances are grouped into multi-digit fret values such as `10`, `12`, and `19` after classification.

### Structural and relationship channel

The following marks cannot be understood reliably from an isolated symbol crop:

- score/TAB staff lines and measure barlines;
- stems and shared beams;
- ties and slurs;
- tuplets and range brackets;
- repeat barlines and alternate endings;
- technique ranges such as let-ring and palm-mute lines.

They are detected on the complete measure or system image and converted into edges or ranges connecting event nodes.

## Minimum Event IR

```json
{
  "measure": 3,
  "voice": 0,
  "order": 4,
  "x_anchor": 826.5,
  "onset": "3/8",
  "duration": "1/8",
  "dotted": false,
  "rest": false,
  "notes": [
    {
      "string": 2,
      "fret": 3,
      "tie_in": false,
      "tie_out": true,
      "effects": []
    }
  ]
}
```

## Minimum Score IR

```json
{
  "tracks": [
    {
      "string_tuning_midi": [64, 59, 55, 50, 45, 40],
      "capo": 0,
      "measures": [
        {
          "number": 1,
          "time_signature": [4, 4],
          "tempo_quarter": 120,
          "events": []
        }
      ]
    }
  ]
}
```

The GP writer consumes Score IR rather than raw CNN predictions. This keeps visual recognition separate from musical validation and file serialization.

## Reconstruction levels

### Level A: TAB transcription

- detect six TAB lines and measure boundaries;
- recognize fret digits/X and string assignment;
- group vertically aligned values into chords;
- retain left-to-right event order.

This recovers fingering but is not sufficient for a playable GP file when rhythm is absent.

### Level B: playable core GP

- add time signatures, rests, duration, dots, voices, and ties;
- add track tuning, capo, and tempo;
- validate that event durations fill each measure.

This is the first end-to-end reconstruction milestone.

### Level C: high-fidelity GP

- add repeats and alternate endings;
- add bends, slides, hammer/pull, harmonics, vibrato, palm mute, let ring, grace notes, and dynamics;
- restore text and layout metadata when useful.

## First CNN baseline

The first model is an atomic-symbol classifier, not a page recognizer. Training samples are generated from the same TuxGuitar 2.0.1 vector painters used by the PDF exporter, then augmented with staff-line backgrounds, translation, scale, rotation, blur, noise, and print-like contrast changes.

The baseline answers:

```text
Given a pre-cropped atomic symbol, which semantic class is it?
```

It does not yet answer:

```text
Where are all symbols on a page?
Which symbols belong to the same event?
Which events are connected by a beam or tie?
```

Those require the event proposal and relationship stages above. Synthetic held-out accuracy measures whether the classifier learned the generated symbol vocabulary; page-crop evaluation must be added before claiming real-PDF accuracy.

## Score/TAB consistency checks

For paired notation, the two representations provide a useful constraint:

```text
pitch_from_tab = open_string_midi[string] + fret
```

The result should agree with the pitch inferred from clef, staff position, key signature, and accidental. This constraint can correct digit, string-line, accidental, and tuning mistakes during Event IR validation.

## TuxGuitar page-coordinate ground truth

For the first renderer-specific baseline, `TGPrintLayout` is replayed with the
same 550 x 800 page size, margins, scale, selected track, and `tab_only` style
used by the PDF writer. At the `paintMeasure` boundary TuxGuitar has already
assigned the page, measure position, beat x positions, TAB origin, and string
spacing. Those values are recorded before conversion to the actual PNG size.

The annotation hierarchy is:

```text
page -> measure -> TAB staff/string y -> beat/event -> digit or dead-X box
```

Multi-digit frets retain both the individual glyph boxes required by the
detector and the shared fret/event metadata needed to reconstruct values such
as 12 or 21. Tied notes that TuxGuitar does not redraw as fret digits are not
incorrectly labeled as visible glyphs. Page overlays are mandatory visual QA;
semantic validation independently reconstructs the expected visible glyphs
from the original GP labels.

## Implemented TuxGuitar pixel-only Level A pipeline

The coordinate exporter above is a training-data and evaluation tool. It is
not used by page inference. Given only a rendered `tab_only` page, the current
pipeline performs:

```text
page pixels
  -> horizontal-run TAB staff detection
  -> vertical-run measure boundary detection
  -> height-preserving overlapping measure tiles
  -> compact digit/X detector
  -> page-coordinate projection and duplicate suppression
  -> nearest-string assignment
  -> multi-digit fret grouping
  -> x-aligned onset event grouping
```

Keeping staff height fixed is important. Squeezing a wide measure into a fixed
width made fret glyphs only a few pixels tall and substantially reduced recall;
overlapping horizontal tiles preserve the original visual scale. The pixel
geometry detector also uses the longest horizontal run rather than a fixed
fraction of page width so that short final systems are retained.

On the current renderer-specific corpus, pixel geometry recovers all 334 TAB
staffs and 958 measures on 63 pages. On source-disjoint test songs, full-page
symbol F1 is 99.61%, matched-symbol string accuracy is 100%, and exact Level A
onset-event F1 is 99.01%. These figures measure TuxGuitar `tab_only` output and
must not be interpreted as cross-renderer or complete-GP accuracy.

Level A events merge simultaneous GP voices at the same onset because the TAB
digits alone do not reliably reveal voice identity. Voice separation, duration,
rests, dots, beams, ties, time signatures, tuning, tempo, and effects remain
Level B/Level C work.

## Implemented TuxGuitar score_tab rhythm-context baseline

The first Level B experiment replays TuxGuitar's paired standard-score/TAB
layout and records score lines, event anchors, both voices, note positions,
duration values, dots, tuplet divisions, stem direction, joins and beam counts.
The logical coordinates are converted to rendered-PNG pixels and independently
checked against the original GP semantics. QA overlays were inspected on the
actual PDF-rendered pages to confirm that score lines, event anchors and both
voices' noteheads align with the raster image.

Each training example is a 256 x 192 crop centred on a ground-truth event. It
keeps neighbouring score context because stems, beams and ties cross event
boundaries. A compact 879,358-parameter CNN predicts, for each of two voices:

```text
state (empty/note/rest) + duration + augmentation dot + tuplet division
```

The corpus currently contains 31 source songs, 110 pages, 958 measures, 6,558
event crops and 6,746 visible voice instances. On source-disjoint test songs,
primary-voice full-semantic accuracy is 89.59% and primary-voice duration
accuracy is 95.05%. Exact two-voice event accuracy is 76.04%.

These results do not solve the complete Level B task. Only two source songs
contain a visible second voice; just one contributes such examples to training,
and its duration distribution differs sharply from the held-out song. The
second-voice full-semantic result is consequently only 2.97%. The evaluation
of the rhythm CNN alone uses ground-truth event centres; the page locator below
removes that dependency. Ties, time signatures and integration with the Level A
TAB output remain separate next steps.

## Implemented pixel-only score event locator

The event locator accepts a complete TuxGuitar `score_tab` page without layout
labels. Long horizontal runs recover score and TAB lines. Five-line standard
staffs are paired with the following TAB staff, and measure boundaries are
taken from TAB vertical runs so that note stems cannot become false barlines.
Score measures are then scale-preserving 512 x 192 tiles.

A 161,538-parameter CNN collapses the two-dimensional score features along the
vertical axis and predicts a one-dimensional CenterNet-style event-column
heatmap plus sub-pixel x offsets. A chord therefore produces one onset column,
not one detection per notehead. Detected columns create the same 256 x 192
context crops consumed by the rhythm CNN.

On 14 source-disjoint test pages, pixel geometry recovers all 39 paired systems
and all 95 measures. For 793 ground-truth events, recall is 100%, precision is
99.87%, F1 is 99.94%, and mean x error is 0.33 pixels at the fixed render scale.
Running the rhythm CNN from detected rather than ground-truth centres preserves
the existing 89.59% primary-visible-voice exact result and 76.04% exact
two-voice event result. The remaining error is therefore dominated by rhythm
semantics, especially the underrepresented second voice, rather than event
position.

The current page pipeline is renderer-specific. The association with Level A
string/fret events, printed time signatures, measure-duration constraints and
conservative tie relations is implemented below; partial/cross-system ties and
GP serialization remain later playable-core steps.

## Implemented score/TAB Event IR association

The `tab_only` digit/X detector is applied unchanged to the TAB region paired
with each detected standard staff. TAB events and score events are associated
within each measure by x distance. Standard notation remains the event/rhythm
anchor; TAB supplies visibly printed string/fret values. Unmatched TAB events
and score-only events are retained instead of being silently discarded.

On source-disjoint `score_tab` test pages, visible TAB event-location F1 is
99.74%, exact fingering accuracy on matched events is 98.59%, and note-level F1
is 98.92%. Combining pixel-only location, predicted primary-voice rhythm and
visible TAB fingering gives 87.39% primary-core exact recall. Requiring both
predicted voices and fingering to be exact gives 74.78%.

This is an intermediate recognition metric, not GP reconstruction accuracy.
Tied notes may have no redrawn TAB glyph; voice assignment is ambiguous when
multiple voices are active; exact onsets, tuning, capo, tempo and effects are
still unknown. The Event IR uses `null` for these fields and keeps recognition
confidence and association status for later musical validation.

## Implemented time signatures and exact measure-capacity audit

Printed time signatures are found in the standard staff and their stacked digit
components are classified by the existing atomic-symbol CNN. The last printed
value is propagated through subsequent measures and ordered pages. On the fixed
TuxGuitar corpus, all 44 printed signatures and all 958 propagated measure
values are correct. The supported observed vocabulary is `1/4`, `2/4`, `3/4`,
`4/4`, `6/4`, `6/8`, `9/4`, and `10/8`; cross-renderer claims require new data.

Predicted durations, dots and tuplet ratios are summed as exact rational
whole-note fractions for each voice. Underfilled or overfilled measures retain
the original CNN output and receive an advisory minimum-cost candidate when the
CNN's alternate classes can exactly fill the measure. A relative-probability
threshold of 0.20 was selected on validation data. It yields 6/6 correct
high-confidence proposals on validation and 6/6 on the independent test split.
On test, primary-rhythm exact F1 rises from 88.97% to 89.66% if those candidates
are applied, fully correct measures rise from 64/95 to 69/95, and sequential
primary-onset accuracy rises from 81.85% to 85.15%.

The same rational pass writes predicted per-voice `onset`, `duration_fraction`
and `end` positions. These are sequentially derived rather than guessed from x
pixels, so a rhythm mistake shifts later onsets and is exposed by the measure
audit. Corrected candidates remain separate from the original timeline.

`guitarocr.pipeline.infer_tuxguitar_score_tab_document` accepts a PDF or ordered page-image set,
loads the four compact models once, carries signatures and measure numbers
across pages, and writes a combined `document_score_ir.json`. PDFs are rendered
by Poppler at the training scale of 180 DPI in grayscale and retain page-level
source provenance. A six-page regression PDF renders pixel-identically to its
database PNGs and produces the same discrete IR semantics. Complete tie
association, track metadata and GP serialization remain the next layers.

## Implemented tie-candidate and conservative relationship stage

The corpus contains 510 tied notes in 347 events. Partial chord continuation is
common: 173 events combine tied notes with newly attacked notes. Twenty-five
tied notes cross a system or page boundary. Treating every visible arc or every
note in its event as a tie would therefore corrupt the score.

An 888,096-parameter event-context CNN reuses the rhythm backbone and predicts
tie presence, tied-note count, total score-note count and tied target-y bins.
Because ties and slur/hammer-pull curves can be visually identical, the page
pipeline accepts a candidate only when total score-note count exceeds newly
printed TAB notes. Splits follow the pretrained rhythm model: test sources are
unseen by both pretraining and tie training.

On 16 source-disjoint test tie events among 793 total events, the combined
candidate has 100% precision, 81.25% recall and 89.66% F1. Count is exact on
10/13 accepted events and target-y F1 is 57.14%. The IR automatically resolves
only adjacent, full-event continuations with no new TAB attack. Nine test notes
are resolved and all nine are correct, covering 39.13% of the 23 tied test
notes. All ambiguous partial, non-adjacent and cross-system relations remain
explicit candidates for the later notehead/TAB association stage.

## Current production path (2026-07-14)

The implemented production path is now:

```text
score_tab PDF
  -> Poppler page raster
  -> staff/system/measure geometry
  -> event-column locator
  -> rhythm + TAB digit/X + tie + technique CNNs
  -> score/TAB association
  -> exact-rational measure constraint and document context
  -> Score Event IR
  -> GP5 plan
  -> TuxGuitar GP5 write/readback validation
  -> optional preview PDF
```

The readback validator checks every planned measure and event, including onset,
duration, dot/division, string/fret, tie state, and GP5-representable note
effects. It fails the export instead of reporting success when TuxGuitar changes
the data during serialization. TuxGuitar's GP5 model makes dead-note and slide
mutually exclusive on one note; the exporter preserves the visible X, records
the downgrade, and leaves both meanings intact in IR.

The rhythm model now covers double dots and tuplets. A measure-level dynamic
program searches the CNN alternatives for an exact rational fill. A deliberately
narrow unique-candidate closure is permitted only when one non-destructive
candidate fills an otherwise invalid measure with sufficient model probability.
This keeps the correction auditable in `rhythm_audit` rather than turning the
constraint solver into an unconstrained score generator.

TAB association handles multi-digit frets, isolated single TAB events, missing
glyphs on tie continuations, partial chord ties, and current-event technique
predictions. Technique recognition uses compact multi-label event CNNs; it is
useful for common dead, palm-mute, vibrato and bend cases but remains a long-tail
problem, especially for slide/ghost/accent. Exact GP7/8 GPIF labels can augment
renderer-derived training semantics. Beat-level up/down PickStroke labels use a
second checkpoint that only overrides those two outputs, avoiding regression in
the original 13 note-effect classes. A narrow sequence pass suppresses pick
strokes on rests, resolves dual-direction logits, and fills a single hole only
inside an otherwise strict alternate-picking run with supporting visual probability.

### Three-layout boundary

Page geometry classifies TuxGuitar renders as `score_tab`, `tab_only`, or
`score_only`:

- `score_tab` is the only complete PDF-to-GP path. Standard notation anchors
  time/rhythm, and TAB resolves guitar string/fret.
- `tab_only` has implemented staff/measure geometry, digit/X detection, and
  event grouping. Its rhythm dataset and inference groundwork exist, but a
  reliable rhythm-to-GP path is not yet the default.
- `score_only` can be classified but cannot uniquely determine guitar fingering:
  one written pitch often has several valid string/fret realizations. It needs a
  learned fingering prior or explicit user constraints before exact GP output.

This boundary is surfaced in the README and CLI rather than silently routing all
three layouts through a model trained for paired score/TAB pages.

### Why the default context model is not a 1B document VLM

GLM-OCR 0.9B and HunyuanOCR are general document OCR models. Their token output
does not directly supervise the precise event x coordinate, voice, rational
duration, string, fret, or cross-event relation required here. The released
runtime therefore remains seven task-specific CNN checkpoints totaling about
17.4 MiB plus deterministic music constraints. General OCR/VLMs may help as
offline weak-label teachers, metadata OCR, or failure-page routers. A future
10--30M event-sequence Transformer over CNN features is a better-sized model for
musical context and can be distilled from exact GP sequences.
