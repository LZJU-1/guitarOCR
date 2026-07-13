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
import app.tuxguitar.util.TGContext;
import app.tuxguitar.util.plugin.TGPlugin;

/** Builds a minimal TuxGuitar song from GuitarOCR's deliberately simple TSV plan. */
public final class TuxGuitarIrGp5Writer {
    private record NotePlan(int string, int fret, boolean tied) {}
    private record EventPlan(
            int measureIndex,
            int eventOrder,
            long onsetNumerator,
            long onsetDenominator,
            int durationValue,
            String dot,
            int divisionEnters,
            int divisionTimes,
            String state,
            List<NotePlan> notes) {}
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
        if (args.length < 2 || args.length > 3) {
            throw new IllegalArgumentException("Usage: TuxGuitarIrGp5Writer PLAN.tsv OUTPUT.gp5 [PREVIEW.pdf]");
        }
        Path planPath = Path.of(args[0]).toAbsolutePath();
        Path gp5Path = Path.of(args[1]).toAbsolutePath();
        Path previewPath = args.length == 3 ? Path.of(args[2]).toAbsolutePath() : null;
        Plan plan = readPlan(planPath);
        TGSongManager manager = new TGSongManager();
        TGSong song = buildSong(plan, manager);
        Files.createDirectories(gp5Path.getParent());

        try (PluginSet plugins = new PluginSet()) {
            writeGp5(plugins.context(), song, manager, gp5Path);
            TGSong readback = readGp5(plugins.context(), manager, gp5Path);
            normalizePreciseStarts(readback, manager);
            int matched = countMatchedEvents(plan, readback);
            if (previewPath != null) {
                Files.createDirectories(previewPath.getParent());
                writePreviewPdf(plugins.context(), readback, manager, previewPath);
            }
            TGTrack readTrack = readback.getTrack(0);
            System.out.println("OUTPUT_GP5=" + gp5Path);
            System.out.println("MEASURES=" + plan.measures().size());
            System.out.println("PLAN_EVENTS=" + plan.events().size());
            System.out.println("PLAN_NOTES=" + countPlanNotes(plan));
            System.out.println("READBACK_MEASURES=" + readback.countMeasureHeaders());
            System.out.println("READBACK_TRACKS=" + readback.countTracks());
            System.out.println("READBACK_BEATS=" + countNonEmptyBeats(readTrack));
            System.out.println("READBACK_MATCHED_EVENTS=" + matched + "/" + plan.events().size());
            if (previewPath != null) {
                System.out.println("PREVIEW_PDF=" + previewPath);
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
        boolean header = false;
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank()) continue;
            String[] fields = line.split("\\t", -1);
            if (fields[0].equals("GUITAROCR_PLAN")) {
                header = fields.length == 2 && fields[1].equals("1");
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
                String[] division = fields[7].split(":", 2);
                List<NotePlan> notes = new ArrayList<>();
                if (!fields[9].equals("-")) {
                    for (String encoded : fields[9].split(",")) {
                        String[] note = encoded.split(":", 3);
                        notes.add(new NotePlan(
                                Integer.parseInt(note[0]), Integer.parseInt(note[1]), note[2].equals("1")));
                    }
                }
                events.add(new EventPlan(
                        Integer.parseInt(fields[1]), Integer.parseInt(fields[2]),
                        Long.parseLong(fields[3]), Long.parseLong(fields[4]),
                        Integer.parseInt(fields[5]), fields[6],
                        Integer.parseInt(division[0]), Integer.parseInt(division[1]),
                        fields[8], notes));
            }
        }
        if (!header) throw new IllegalArgumentException("Unsupported or missing GuitarOCR plan header");
        if (measures.isEmpty()) throw new IllegalArgumentException("Plan has no measures");
        if (tuning.isEmpty()) throw new IllegalArgumentException("Plan has no tuning");
        measures.sort(Comparator.comparingInt(MeasurePlan::index));
        return new Plan(title, tempo, tuning, measures, events);
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

        for (EventPlan item : plan.events()) {
            TGMeasure measure = measureByIndex.get(item.measureIndex());
            if (measure == null) throw new IllegalArgumentException("Event references a missing measure");
            TGBeat beat = factory.newBeat();
            long relativeStart = fractionOfWholeToTicks(item.onsetNumerator(), item.onsetDenominator());
            long relativePrecise = fractionOfWholeToPrecise(item.onsetNumerator(), item.onsetDenominator());
            beat.setStart(measure.getStart() + relativeStart);
            beat.setPreciseStart(measure.getPreciseStart() + relativePrecise);
            TGVoice voice = beat.getVoice(0);
            if (voice == null) {
                voice = factory.newVoice(0);
                beat.setVoice(0, voice);
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
                    voice.addNote(note);
                }
            }
            measure.addBeat(beat);
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
            TGContext context, TGSong song, TGSongManager manager, Path path) throws Exception {
        TGPrintSettings settings = new TGPrintSettings();
        settings.setStyle(TGLayout.DISPLAY_COMPACT | TGLayout.DISPLAY_MODE_BLACK_WHITE
                | TGLayout.DISPLAY_SCORE | TGLayout.DISPLAY_TABLATURE);
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

    private static int countMatchedEvents(Plan plan, TGSong song) {
        TGTrack track = song.getTrack(0);
        int matches = 0;
        for (EventPlan expected : plan.events()) {
            TGMeasure measure = track.getMeasure(expected.measureIndex());
            long start = measure.getStart()
                    + fractionOfWholeToTicks(expected.onsetNumerator(), expected.onsetDenominator());
            boolean matched = false;
            for (TGBeat beat : measure.getBeats()) {
                if (beat.getStart() != start) continue;
                TGVoice voice = beat.getVoice(0);
                if (voice == null || voice.isEmpty()) continue;
                if (voice.getDuration().getValue() != expected.durationValue()) continue;
                if (voice.getDuration().isDotted() != expected.dot().equals("single")) continue;
                if (voice.getDuration().isDoubleDotted() != expected.dot().equals("double")) continue;
                if (voice.getDuration().getDivision().getEnters() != expected.divisionEnters()) continue;
                if (voice.getDuration().getDivision().getTimes() != expected.divisionTimes()) continue;
                if (!notesMatch(expected, voice)) continue;
                matches++;
                matched = true;
                break;
            }
            if (!matched) {
                StringBuilder actualAtStart = new StringBuilder();
                for (TGBeat beat : measure.getBeats()) {
                    if (beat.getStart() != start) continue;
                    TGVoice voice = beat.getVoice(0);
                    if (voice == null) continue;
                    actualAtStart.append("[dur=").append(voice.getDuration().getValue())
                            .append(" dot=").append(voice.getDuration().isDotted())
                            .append(" div=").append(voice.getDuration().getDivision().getEnters())
                            .append(":").append(voice.getDuration().getDivision().getTimes())
                            .append(" rest=").append(voice.isRestVoice()).append(" notes=");
                    for (TGNote note : voice.getNotes()) {
                        actualAtStart.append(note.getString()).append(":").append(note.getValue())
                                .append(":").append(note.isTiedNote()).append(",");
                    }
                    actualAtStart.append("]");
                }
                System.out.println("UNMATCHED_EVENT=m" + (expected.measureIndex() + 1)
                        + " e" + expected.eventOrder() + " start=" + start
                        + " dur=" + expected.durationValue() + " dot=" + expected.dot()
                        + " div=" + expected.divisionEnters() + ":" + expected.divisionTimes()
                        + " state=" + expected.state() + " notes=" + expected.notes()
                        + " actual=" + actualAtStart);
            }
        }
        return matches;
    }

    private static boolean notesMatch(EventPlan expected, TGVoice actual) {
        if (expected.state().endsWith("rest")) return actual.isRestVoice();
        if (actual.countNotes() != expected.notes().size()) return false;
        List<String> expectedNotes = new ArrayList<>();
        for (NotePlan note : expected.notes()) {
            expectedNotes.add(note.tied()
                    ? note.string() + ":T"
                    : note.string() + ":" + note.fret() + ":false");
        }
        List<String> actualNotes = new ArrayList<>();
        for (TGNote note : actual.getNotes()) {
            actualNotes.add(note.isTiedNote()
                    ? note.getString() + ":T"
                    : note.getString() + ":" + note.getValue() + ":false");
        }
        Collections.sort(expectedNotes);
        Collections.sort(actualNotes);
        return expectedNotes.equals(actualNotes);
    }
}
