import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import app.tuxguitar.graphics.control.TGLayout;
import app.tuxguitar.graphics.control.print.TGPrintSettings;
import app.tuxguitar.io.base.TGFileFormat;
import app.tuxguitar.io.base.TGFileFormatManager;
import app.tuxguitar.io.base.TGFileFormatUtils;
import app.tuxguitar.io.base.TGSongReaderHandle;
import app.tuxguitar.io.base.TGSongReaderHelper;
import app.tuxguitar.io.base.TGSongStreamContext;
import app.tuxguitar.io.base.TGSongWriterHandle;
import app.tuxguitar.io.gtp.GP5InputStreamPlugin;
import app.tuxguitar.io.gtp.GP5OutputStreamPlugin;
import app.tuxguitar.io.pdf.PDFSongWriter;
import app.tuxguitar.io.pdf.PDFSongWriterPlugin;
import app.tuxguitar.song.factory.TGFactory;
import app.tuxguitar.song.managers.TGSongManager;
import app.tuxguitar.song.models.TGBeat;
import app.tuxguitar.song.models.TGDivisionType;
import app.tuxguitar.song.models.TGDuration;
import app.tuxguitar.song.models.TGMeasure;
import app.tuxguitar.song.models.TGMeasureHeader;
import app.tuxguitar.song.models.TGNote;
import app.tuxguitar.song.models.TGSong;
import app.tuxguitar.song.models.TGString;
import app.tuxguitar.song.models.TGTrack;
import app.tuxguitar.song.models.TGVoice;
import app.tuxguitar.song.models.effects.TGEffectBend;
import app.tuxguitar.util.TGContext;
import app.tuxguitar.util.plugin.TGPlugin;

/** Builds a minimal TuxGuitar song from GuitarOCR's deliberately simple TSV plan. */
public final class TuxGuitarIrGp5Writer {
    private record NotePlan(
            int string, int fret, boolean tied, boolean dead, boolean vibrato,
            boolean slide, boolean hammer, boolean bend, boolean ghost,
            boolean accent, boolean palmMute, boolean staccato,
            boolean letRing, boolean tapping) {}
    private record EventPlan(
            int measureIndex,
            int eventOrder,
            int voiceIndex,
            long onsetNumerator,
            long onsetDenominator,
            int durationValue,
            String dot,
            int divisionEnters,
            int divisionTimes,
            String state,
            List<NotePlan> notes,
            int pickStroke) {}
    private record MeasurePlan(int index, int sourceNumber, int numerator, int denominator) {}
    private record Plan(String title, int tempo, List<Integer> tuning,
                        List<MeasurePlan> measures, List<EventPlan> events) {}

    private static final class PluginSet implements AutoCloseable {
        private final TGContext context = new TGContext();
        private final List<TGPlugin> plugins = new ArrayList<>();

        PluginSet() throws Exception {
            this.plugins.add(new GP5InputStreamPlugin());
            this.plugins.add(new GP5OutputStreamPlugin());
            this.plugins.add(new PDFSongWriterPlugin());
            for (TGPlugin plugin : this.plugins) {
                plugin.connect(this.context);
            }
        }

        TGContext context() {
            return this.context;
        }

        @Override
        public void close() {
            Collections.reverse(this.plugins);
            for (TGPlugin plugin : this.plugins) {
                try {
                    plugin.disconnect(this.context);
                } catch (Throwable ignored) {
                    // The output files have already been closed; cleanup is best effort.
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 4) {
            throw new IllegalArgumentException(
                    "Usage: TuxGuitarIrGp5Writer PLAN.tsv OUTPUT.gp5 "
                    + "[PREVIEW.pdf] [score_tab|tab_only|score_only]");
        }
        Path planPath = Path.of(args[0]).toAbsolutePath();
        Path gp5Path = Path.of(args[1]).toAbsolutePath();
        Path previewPath = args.length >= 3 ? Path.of(args[2]).toAbsolutePath() : null;
        String previewLayout = args.length == 4 ? args[3] : "score_tab";
        Plan plan = readPlan(planPath);
        TGSongManager manager = new TGSongManager();
        TGSong song = buildSong(plan, manager);
        Files.createDirectories(gp5Path.getParent());

        try (PluginSet plugins = new PluginSet()) {
            writeGp5(plugins.context(), song, manager, gp5Path);
            TGSong readback = readGp5(plugins.context(), manager, gp5Path);
            normalizePreciseStarts(readback, manager);
            int matched = countMatchedEvents(plan, readback, false, true);
            int fullyMatched = countMatchedEvents(plan, readback, true, false);
            if (previewPath != null) {
                Files.createDirectories(previewPath.getParent());
                writePreviewPdf(plugins.context(), readback, manager, previewPath, previewLayout);
            }
            TGTrack readTrack = readback.getTrack(0);
            System.out.println("OUTPUT_GP5=" + gp5Path);
            System.out.println("MEASURES=" + plan.measures().size());
            System.out.println("PLAN_EVENTS=" + plan.events().size());
            System.out.println("PLAN_NOTES=" + countPlanNotes(plan));
            System.out.println("READBACK_MEASURES=" + readback.countMeasureHeaders());
            System.out.println("READBACK_TRACKS=" + readback.countTracks());
            System.out.println("READBACK_BEATS=" + countNonEmptyBeats(readTrack));
            System.out.println("READBACK_VOICE_EVENTS=" + countNonEmptyVoiceEvents(readTrack));
            System.out.println("READBACK_MATCHED_EVENTS=" + matched + "/" + plan.events().size());
            System.out.println("READBACK_FULLY_MATCHED_EVENTS="
                    + fullyMatched + "/" + plan.events().size());
            System.out.println("READBACK_LOSSY_SEMANTIC_EVENTS=" + (matched - fullyMatched));
            if (previewPath != null) {
                System.out.println("PREVIEW_PDF=" + previewPath);
                System.out.println("PREVIEW_LAYOUT=" + previewLayout);
            }
            if (readback.countMeasureHeaders() != plan.measures().size() || matched != plan.events().size()) {
                throw new IllegalStateException("GP5 readback did not preserve every planned measure/event");
            }
        }
    }

    private static Plan readPlan(Path path) throws Exception {
        String title = "GuitarOCR result";
        int tempo = 120;
        List<Integer> tuning = new ArrayList<>();
        List<MeasurePlan> measures = new ArrayList<>();
        List<EventPlan> events = new ArrayList<>();
        int planVersion = 0;
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank()) continue;
            String[] fields = line.split("\\t", -1);
            if (fields[0].equals("GUITAROCR_PLAN")) {
                if (fields.length == 2) planVersion = Integer.parseInt(fields[1]);
            } else if (fields[0].equals("META")) {
                if (fields[1].equals("TITLE_B64")) {
                    title = new String(Base64.getDecoder().decode(fields[2]), StandardCharsets.UTF_8);
                } else if (fields[1].equals("TEMPO")) {
                    tempo = Integer.parseInt(fields[2]);
                } else if (fields[1].equals("TUNING")) {
                    for (String value : fields[2].split(",")) tuning.add(Integer.parseInt(value));
                }
            } else if (fields[0].equals("MEASURE")) {
                measures.add(new MeasurePlan(
                        Integer.parseInt(fields[1]), Integer.parseInt(fields[2]),
                        Integer.parseInt(fields[3]), Integer.parseInt(fields[4])));
            } else if (fields[0].equals("EVENT")) {
                int shift = planVersion >= 3 ? 1 : 0;
                int voiceIndex = planVersion >= 3 ? Integer.parseInt(fields[3]) : 0;
                String[] division = fields[7 + shift].split(":", 2);
                List<NotePlan> notes = new ArrayList<>();
                if (!fields[9 + shift].equals("-")) {
                    for (String encoded : fields[9 + shift].split(",")) {
                        String[] note = encoded.split(":");
                        notes.add(new NotePlan(
                                Integer.parseInt(note[0]), Integer.parseInt(note[1]), note[2].equals("1"),
                                flag(note, 3), flag(note, 4), flag(note, 5), flag(note, 6),
                                flag(note, 7), flag(note, 8), flag(note, 9), flag(note, 10),
                                flag(note, 11), flag(note, 12), flag(note, 13)));
                    }
                }
                events.add(new EventPlan(
                        Integer.parseInt(fields[1]), Integer.parseInt(fields[2]),
                        voiceIndex,
                        Long.parseLong(fields[3 + shift]), Long.parseLong(fields[4 + shift]),
                        Integer.parseInt(fields[5 + shift]), fields[6 + shift],
                         Integer.parseInt(division[0]), Integer.parseInt(division[1]),
                         fields[8 + shift], notes,
                         fields.length >= 11 + shift ? Integer.parseInt(fields[10 + shift]) : 0));
            }
        }
        if (planVersion < 1 || planVersion > 3) {
            throw new IllegalArgumentException("Unsupported or missing GuitarOCR plan header");
        }
        if (measures.isEmpty()) throw new IllegalArgumentException("Plan has no measures");
        if (tuning.isEmpty()) throw new IllegalArgumentException("Plan has no tuning");
        measures.sort(Comparator.comparingInt(MeasurePlan::index));
        return new Plan(title, tempo, tuning, measures, events);
    }

    private static boolean flag(String[] fields, int index) {
        return fields.length > index && fields[index].equals("1");
    }

    private static TGSong buildSong(Plan plan, TGSongManager manager) {
        TGFactory factory = manager.getFactory();
        TGSong song = manager.newSong();
        song.setName(plan.title());
        song.setAuthor("GuitarOCR MVP");
        song.setComments("Generated from raster score+TAB IR; unsupported semantics may be absent.");
        TGTrack track = song.getTrack(0);
        track.setName("GuitarOCR");
        track.setNumber(1);
        track.setStrings(createStrings(factory, plan.tuning()));
        while (track.countMeasures() > 0) track.removeMeasure(0);
        while (song.countMeasureHeaders() > 0) song.removeMeasureHeader(0);

        long measureStart = TGDuration.QUARTER_TIME;
        Map<Integer, TGMeasure> measureByIndex = new HashMap<>();
        for (MeasurePlan item : plan.measures()) {
            TGMeasureHeader header = factory.newHeader();
            header.setNumber(item.index() + 1);
            header.setStart(measureStart);
            header.setPreciseStart(TGDuration.toPreciseTime(measureStart));
            header.getTimeSignature().setNumerator(item.numerator());
            header.getTimeSignature().getDenominator().setValue(item.denominator());
            header.getTempo().setQuarterValue(plan.tempo());
            song.addMeasureHeader(header);
            TGMeasure measure = factory.newMeasure(header);
            track.addMeasure(measure);
            measureByIndex.put(item.index(), measure);
            measureStart += fractionOfWholeToTicks(item.numerator(), item.denominator());
        }

        Map<String, TGBeat> beatByPosition = new HashMap<>();
        for (EventPlan item : plan.events()) {
            TGMeasure measure = measureByIndex.get(item.measureIndex());
            if (measure == null) throw new IllegalArgumentException("Event references a missing measure");
            long relativeStart = fractionOfWholeToTicks(item.onsetNumerator(), item.onsetDenominator());
            long relativePrecise = fractionOfWholeToPrecise(item.onsetNumerator(), item.onsetDenominator());
            String beatKey = item.measureIndex() + ":" + relativePrecise;
            TGBeat beat = beatByPosition.get(beatKey);
            if (beat == null) {
                beat = factory.newBeat();
                beat.setStart(measure.getStart() + relativeStart);
                beat.setPreciseStart(measure.getPreciseStart() + relativePrecise);
                measure.addBeat(beat);
                beatByPosition.put(beatKey, beat);
            }
            if (item.pickStroke() != 0) beat.getPickStroke().setDirection(item.pickStroke());
            TGVoice voice = beat.getVoice(item.voiceIndex());
            if (voice == null) {
                voice = factory.newVoice(item.voiceIndex());
                beat.setVoice(item.voiceIndex(), voice);
            }
            voice.setEmpty(false);
            voice.getDuration().setValue(item.durationValue());
            voice.getDuration().setDotted(item.dot().equals("single"));
            voice.getDuration().setDoubleDotted(item.dot().equals("double"));
            TGDivisionType division = voice.getDuration().getDivision();
            division.setEnters(item.divisionEnters());
            division.setTimes(item.divisionTimes());
            if (item.state().equals("note")) {
                for (NotePlan notePlan : item.notes()) {
                    TGNote note = factory.newNote();
                    note.setString(notePlan.string());
                    note.setValue(notePlan.fret());
                    note.setTiedNote(notePlan.tied());
                    note.setVelocity(95);
                    note.getEffect().setVibrato(notePlan.vibrato());
                    note.getEffect().setSlide(notePlan.slide());
                    note.getEffect().setHammer(notePlan.hammer());
                    note.getEffect().setGhostNote(notePlan.ghost());
                    note.getEffect().setAccentuatedNote(notePlan.accent());
                    note.getEffect().setPalmMute(notePlan.palmMute());
                    note.getEffect().setStaccato(notePlan.staccato());
                    note.getEffect().setLetRing(notePlan.letRing());
                    note.getEffect().setTapping(notePlan.tapping());
                    if (notePlan.bend()) {
                        TGEffectBend bend = factory.newEffectBend();
                        bend.addPoint(0, 0);
                        bend.addPoint(TGEffectBend.MAX_POSITION_LENGTH, 2 * TGEffectBend.SEMITONE_LENGTH);
                        note.getEffect().setBend(bend);
                    }
                    // TGNoteEffect treats slide and dead-note as mutually exclusive.
                    // Apply dead last as a safety net so an X is never emitted as 0.
                    note.getEffect().setDeadNote(notePlan.dead());
                    voice.addNote(note);
                }
            }
        }
        manager.orderBeats(song);
        manager.updatePreciseStart(song);
        manager.autoCompleteSilences(song);
        manager.orderBeats(song);
        manager.updatePreciseStart(song);
        return song;
    }

    private static List<TGString> createStrings(TGFactory factory, List<Integer> tuning) {
        List<TGString> strings = new ArrayList<>();
        for (int index = 0; index < tuning.size(); index++) {
            TGString value = factory.newString();
            value.setNumber(index + 1);
            value.setValue(tuning.get(index));
            strings.add(value);
        }
        return strings;
    }

    private static long fractionOfWholeToTicks(long numerator, long denominator) {
        return Math.round((double) numerator * (TGDuration.QUARTER_TIME * 4L) / denominator);
    }

    private static long fractionOfWholeToPrecise(long numerator, long denominator) {
        return Math.round((double) numerator * TGDuration.WHOLE_PRECISE_DURATION / denominator);
    }

    private static void writeGp5(TGContext context, TGSong song, TGSongManager manager, Path path)
            throws Exception {
        TGFileFormat format = TGFileFormatManager.getInstance(context).findWriterFileFormatByCode("gp5");
        if (format == null) throw new IllegalStateException("TuxGuitar GP5 writer format is unavailable");
        TGSongWriterHandle handle = new TGSongWriterHandle();
        handle.setSong(song);
        handle.setFactory(manager.getFactory());
        handle.setFormat(format);
        handle.setContext(new TGSongStreamContext());
        try (BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(path.toFile()))) {
            handle.setOutputStream(output);
            TGFileFormatManager.getInstance(context).write(handle);
        }
        if (!Files.isRegularFile(path) || Files.size(path) == 0) {
            throw new IllegalStateException("GP5 writer produced an empty file");
        }
    }

    private static TGSong readGp5(TGContext context, TGSongManager manager, Path path) throws Exception {
        TGSongReaderHandle handle = new TGSongReaderHandle();
        handle.setFactory(manager.getFactory());
        handle.setContext(new TGSongStreamContext());
        handle.getContext().setAttribute(
                TGSongReaderHelper.ATTRIBUTE_FORMAT_CODE, TGFileFormatUtils.getFileFormatCode(path.toString()));
        try (BufferedInputStream input = new BufferedInputStream(new FileInputStream(path.toFile()))) {
            handle.setInputStream(input);
            TGFileFormatManager.getInstance(context).read(handle);
        }
        if (handle.getSong() == null || handle.getSong().isEmpty()) {
            throw new IllegalStateException("TuxGuitar could not read its generated GP5 file");
        }
        return handle.getSong();
    }

    private static void normalizePreciseStarts(TGSong song, TGSongManager manager) {
        for (java.util.Iterator<TGTrack> tracks = song.getTracks(); tracks.hasNext();) {
            TGTrack track = tracks.next();
            for (java.util.Iterator<TGMeasure> measures = track.getMeasures(); measures.hasNext();) {
                for (TGBeat beat : measures.next().getBeats()) {
                    if (beat.getPreciseStart() == null) {
                        beat.setPreciseStart(TGDuration.toPreciseTime(beat.getStart()));
                    }
                }
            }
        }
        manager.updatePreciseStart(song);
    }

    private static void writePreviewPdf(
            TGContext context,
            TGSong song,
            TGSongManager manager,
            Path path,
            String layout) throws Exception {
        TGPrintSettings settings = new TGPrintSettings();
        int style = TGLayout.DISPLAY_COMPACT | TGLayout.DISPLAY_MODE_BLACK_WHITE;
        if ("tab_only".equals(layout)) {
            style |= TGLayout.DISPLAY_TABLATURE;
        } else if ("score_only".equals(layout)) {
            style |= TGLayout.DISPLAY_SCORE;
        } else if ("score_tab".equals(layout)) {
            style |= TGLayout.DISPLAY_SCORE | TGLayout.DISPLAY_TABLATURE;
        } else {
            throw new IllegalArgumentException("Unsupported preview layout: " + layout);
        }
        settings.setStyle(style);
        settings.setFromMeasure(1);
        settings.setToMeasure(song.countMeasureHeaders());
        settings.setTrackNumber(1);
        TGSongStreamContext streamContext = new TGSongStreamContext();
        streamContext.setAttribute(TGPrintSettings.ATTRIBUTE_PRINT_STYLES, settings);
        streamContext.setAttribute(TGPrintSettings.ATTRIBUTE_PRINT_ZOOM, Integer.valueOf(100));
        TGSongWriterHandle writer = new TGSongWriterHandle();
        writer.setSong(song);
        writer.setFactory(manager.getFactory());
        writer.setFormat(PDFSongWriter.FILE_FORMAT);
        writer.setContext(streamContext);
        try (BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(path.toFile()))) {
            writer.setOutputStream(output);
            TGFileFormatManager.getInstance(context).write(writer);
        }
        if (!Files.isRegularFile(path) || Files.size(path) == 0) {
            throw new IllegalStateException("PDF preview writer produced an empty file");
        }
    }

    private static int countPlanNotes(Plan plan) {
        int count = 0;
        for (EventPlan event : plan.events()) count += event.notes().size();
        return count;
    }

    private static int countNonEmptyBeats(TGTrack track) {
        int count = 0;
        for (int measureIndex = 0; measureIndex < track.countMeasures(); measureIndex++) {
            for (TGBeat beat : track.getMeasure(measureIndex).getBeats()) {
                TGVoice voice = beat.getVoice(0);
                if (voice != null && !voice.isEmpty()) count++;
            }
        }
        return count;
    }

    private static int countNonEmptyVoiceEvents(TGTrack track) {
        int count = 0;
        for (int measureIndex = 0; measureIndex < track.countMeasures(); measureIndex++) {
            for (TGBeat beat : track.getMeasure(measureIndex).getBeats()) {
                for (int voiceIndex = 0; voiceIndex < beat.countVoices(); voiceIndex++) {
                    TGVoice voice = beat.getVoice(voiceIndex);
                    if (voice != null && !voice.isEmpty()) count++;
                }
            }
        }
        return count;
    }

    private static int countMatchedEvents(
            Plan plan, TGSong song, boolean requireAllSemantics, boolean logUnmatched) {
        TGTrack track = song.getTrack(0);
        int matches = 0;
        for (EventPlan expected : plan.events()) {
            TGMeasure measure = track.getMeasure(expected.measureIndex());
            long start = measure.getStart()
                    + fractionOfWholeToTicks(expected.onsetNumerator(), expected.onsetDenominator());
            boolean matched = false;
            for (TGBeat beat : measure.getBeats()) {
                if (beat.getStart() != start) continue;
                TGVoice voice = beat.getVoice(expected.voiceIndex());
                if (voice == null || voice.isEmpty()) continue;
                if (voice.getDuration().getValue() != expected.durationValue()) continue;
                if (voice.getDuration().isDotted() != expected.dot().equals("single")) continue;
                if (voice.getDuration().isDoubleDotted() != expected.dot().equals("double")) continue;
                if (voice.getDuration().getDivision().getEnters() != expected.divisionEnters()) continue;
                if (voice.getDuration().getDivision().getTimes() != expected.divisionTimes()) continue;
                if (requireAllSemantics && expected.pickStroke() != 0
                        && beat.getPickStroke().getDirection() != expected.pickStroke()) continue;
                if (!notesMatch(expected, voice, requireAllSemantics)) continue;
                matches++;
                matched = true;
                break;
            }
            if (!matched && logUnmatched) {
                StringBuilder actualAtStart = new StringBuilder();
                for (TGBeat beat : measure.getBeats()) {
                    if (beat.getStart() != start) continue;
                    TGVoice voice = beat.getVoice(expected.voiceIndex());
                    if (voice == null) continue;
                    actualAtStart.append("[dur=").append(voice.getDuration().getValue())
                            .append(" dot=").append(voice.getDuration().isDotted())
                            .append(" div=").append(voice.getDuration().getDivision().getEnters())
                            .append(":").append(voice.getDuration().getDivision().getTimes())
                            .append(" rest=").append(voice.isRestVoice())
                            .append(" pick=").append(beat.getPickStroke().getDirection())
                            .append(" notes=");
                    for (TGNote note : voice.getNotes()) {
                        actualAtStart.append(note.getString()).append(":").append(note.getValue())
                                .append(":").append(note.isTiedNote())
                                .append(":dead=").append(note.getEffect().isDeadNote())
                                .append(":slide=").append(note.getEffect().isSlide())
                                .append(":vibrato=").append(note.getEffect().isVibrato())
                                .append(":palm=").append(note.getEffect().isPalmMute())
                                .append(",");
                    }
                    actualAtStart.append("]");
                }
                System.out.println("UNMATCHED_EVENT=m" + (expected.measureIndex() + 1)
                        + " e" + expected.eventOrder() + " v" + expected.voiceIndex()
                        + " start=" + start
                        + " dur=" + expected.durationValue() + " dot=" + expected.dot()
                        + " div=" + expected.divisionEnters() + ":" + expected.divisionTimes()
                        + " state=" + expected.state() + " notes=" + expected.notes()
                        + " actual=" + actualAtStart);
            }
        }
        return matches;
    }

    private static boolean notesMatch(
            EventPlan expected, TGVoice actual, boolean requireAllSemantics) {
        if (expected.state().endsWith("rest")) return actual.isRestVoice();
        if (actual.countNotes() != expected.notes().size()) return false;
        for (NotePlan expectedNote : expected.notes()) {
            TGNote actualNote = null;
            for (TGNote candidate : actual.getNotes()) {
                if (candidate.getString() == expectedNote.string()) {
                    actualNote = candidate;
                    break;
                }
            }
            if (actualNote == null) return false;
            if (actualNote.isTiedNote() != expectedNote.tied()) return false;
            if (!expectedNote.tied() && actualNote.getValue() != expectedNote.fret()) return false;
            // A dead note is the printed X token and therefore belongs to the
            // structural note identity, not the optional technique layer.
            if (actualNote.getEffect().isDeadNote() != expectedNote.dead()) return false;
            if (requireAllSemantics && !effectsMatch(expectedNote, actualNote)) return false;
        }
        return true;
    }

    private static boolean effectsMatch(NotePlan expected, TGNote actual) {
        return actual.getEffect().isDeadNote() == expected.dead()
                && actual.getEffect().isVibrato() == expected.vibrato()
                && actual.getEffect().isSlide() == expected.slide()
                && actual.getEffect().isHammer() == expected.hammer()
                && actual.getEffect().isBend() == expected.bend()
                && actual.getEffect().isGhostNote() == expected.ghost()
                && actual.getEffect().isAccentuatedNote() == expected.accent()
                && actual.getEffect().isPalmMute() == expected.palmMute()
                && actual.getEffect().isStaccato() == expected.staccato()
                && actual.getEffect().isLetRing() == expected.letRing()
                && actual.getEffect().isTapping() == expected.tapping();
    }
}
