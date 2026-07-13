import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;

import app.tuxguitar.graphics.control.TGBeatGroup;
import app.tuxguitar.graphics.control.TGBeatImpl;
import app.tuxguitar.graphics.control.TGController;
import app.tuxguitar.graphics.control.TGFactoryImpl;
import app.tuxguitar.graphics.control.TGLayout;
import app.tuxguitar.graphics.control.TGLayoutStyles;
import app.tuxguitar.graphics.control.TGMeasureImpl;
import app.tuxguitar.graphics.control.TGNoteImpl;
import app.tuxguitar.graphics.control.TGTrackSpacing;
import app.tuxguitar.graphics.control.TGVoiceImpl;
import app.tuxguitar.graphics.control.print.TGPrintController;
import app.tuxguitar.graphics.control.print.TGPrintDocument;
import app.tuxguitar.graphics.control.print.TGPrintLayout;
import app.tuxguitar.graphics.control.print.TGPrintSettings;
import app.tuxguitar.io.base.TGFileFormatManager;
import app.tuxguitar.io.base.TGFileFormatUtils;
import app.tuxguitar.io.base.TGSongReaderHandle;
import app.tuxguitar.io.base.TGSongReaderHelper;
import app.tuxguitar.io.base.TGSongStreamContext;
import app.tuxguitar.io.gtp.GP1InputStreamPlugin;
import app.tuxguitar.io.gtp.GP2InputStreamPlugin;
import app.tuxguitar.io.gtp.GP3InputStreamPlugin;
import app.tuxguitar.io.gtp.GP4InputStreamPlugin;
import app.tuxguitar.io.gtp.GP5InputStreamPlugin;
import app.tuxguitar.io.pdf.PDFDocument;
import app.tuxguitar.io.pdf.PDFLayoutStyles;
import app.tuxguitar.io.pdf.PDFResourceFactory;
import app.tuxguitar.io.pdf.PDFSongWriterPlugin;
import app.tuxguitar.song.managers.TGSongManager;
import app.tuxguitar.song.models.TGBeat;
import app.tuxguitar.song.models.TGDuration;
import app.tuxguitar.song.models.TGMeasure;
import app.tuxguitar.song.models.TGNote;
import app.tuxguitar.song.models.TGSong;
import app.tuxguitar.song.models.TGTrack;
import app.tuxguitar.song.models.TGVoice;
import app.tuxguitar.ui.resource.UIInset;
import app.tuxguitar.ui.resource.UIPainter;
import app.tuxguitar.ui.resource.UISize;
import app.tuxguitar.util.TGContext;
import app.tuxguitar.util.plugin.TGPlugin;

/**
 * Exports TuxGuitar score+TAB event positions together with rhythm semantics.
 *
 * The output deliberately labels musical events rather than pretending that a
 * beamed group can be decomposed into independent connected components.  The
 * companion Python builder maps these 550 x 800 logical coordinates to PNG
 * pixels, creates event crops, and draws visual QA overlays.
 */
public final class TuxGuitarScoreRhythmAnnotationBuilder {
    private static final float PAGE_WIDTH = 550f;
    private static final float PAGE_HEIGHT = 800f;
    private static final float MARGIN_TOP = 20f;
    private static final float MARGIN_BOTTOM = 20f;
    private static final float MARGIN_LEFT = 20f;
    private static final float MARGIN_RIGHT = 20f;

    private final TGContext context;
    private final List<TGPlugin> plugins;

    private TuxGuitarScoreRhythmAnnotationBuilder() throws Exception {
        this.context = new TGContext();
        this.plugins = new ArrayList<>();
        connectPlugins();
    }

    private void connectPlugins() throws Exception {
        this.plugins.add(new GP1InputStreamPlugin());
        this.plugins.add(new GP2InputStreamPlugin());
        this.plugins.add(new GP3InputStreamPlugin());
        this.plugins.add(new GP4InputStreamPlugin());
        this.plugins.add(new GP5InputStreamPlugin());
        this.plugins.add(new app.tuxguitar.io.gpx.v6.GPXInputStreamPlugin());
        this.plugins.add(new app.tuxguitar.io.gpx.v7.GPXInputStreamPlugin());
        this.plugins.add(new PDFSongWriterPlugin());
        for (TGPlugin plugin : this.plugins) {
            plugin.connect(this.context);
        }
    }

    private void disconnectPlugins() {
        List<TGPlugin> reversed = new ArrayList<>(this.plugins);
        Collections.reverse(reversed);
        for (TGPlugin plugin : reversed) {
            try {
                plugin.disconnect(this.context);
            } catch (Throwable ignored) {
                // Preserve completed files when the standalone process exits.
            }
        }
    }

    private void run(Path databaseRoot, int limit) throws Exception {
        Path sourceRoot = databaseRoot.resolve("source/gp");
        Path outputRoot = databaseRoot.resolve("labels/layout/score_tab_rhythm");
        Files.createDirectories(outputRoot);

        List<Path> sources;
        try (Stream<Path> stream = Files.list(sourceRoot)) {
            sources = stream
                    .filter(Files::isRegularFile)
                    .filter(path -> !path.getFileName().toString().startsWith("."))
                    .sorted(Comparator.comparing(path -> path.getFileName().toString()))
                    .toList();
        }
        if (limit > 0 && sources.size() > limit) {
            sources = new ArrayList<>(sources.subList(0, limit));
        }

        int sourceCount = 0;
        int pageCount = 0;
        int measureCount = 0;
        int eventCount = 0;
        int visibleVoiceCount = 0;
        for (Path source : sources) {
            String fileName = source.getFileName().toString();
            int dot = fileName.lastIndexOf('.');
            String sourceId = dot > 0 ? fileName.substring(0, dot) : fileName;
            TGSong song = readSong(source);
            TGTrack track = chooseTrack(song);
            if (track == null) {
                throw new IOException("No supported guitar/bass track in " + source);
            }

            AnnotationCollector collector = createAnnotations(sourceId, song, track);
            Files.writeString(outputRoot.resolve(sourceId + ".json"),
                    collector.toJson(track.getNumber()), StandardCharsets.UTF_8);
            sourceCount++;
            pageCount += collector.pages.size();
            measureCount += collector.measureCount();
            eventCount += collector.eventCount();
            visibleVoiceCount += collector.visibleVoiceCount();
            System.out.printf(Locale.ROOT,
                    "ANNOTATED source=%s pages=%d measures=%d events=%d visible_voices=%d%n",
                    sourceId, collector.pages.size(), collector.measureCount(), collector.eventCount(),
                    collector.visibleVoiceCount());
        }
        System.out.printf(Locale.ROOT,
                "COMPLETE sources=%d pages=%d measures=%d events=%d visible_voices=%d output=%s%n",
                sourceCount, pageCount, measureCount, eventCount, visibleVoiceCount, outputRoot);
    }

    private TGSong readSong(Path input) throws Exception {
        TGSongManager manager = new TGSongManager();
        TGSongReaderHandle handle = new TGSongReaderHandle();
        handle.setFactory(manager.getFactory());
        handle.setInputStream(new FileInputStream(input.toFile()));
        handle.setContext(new TGSongStreamContext());
        handle.getContext().setAttribute(TGSongReaderHelper.ATTRIBUTE_FORMAT_CODE,
                TGFileFormatUtils.getFileFormatCode(input.toString()));
        TGFileFormatManager.getInstance(this.context).read(handle);
        TGSong song = handle.getSong();
        if (song == null || song.isEmpty()) {
            throw new IOException("Empty song: " + input);
        }
        Iterator<TGTrack> tracks = song.getTracks();
        while (tracks.hasNext()) {
            Iterator<TGMeasure> measures = tracks.next().getMeasures();
            while (measures.hasNext()) {
                for (TGBeat beat : measures.next().getBeats()) {
                    if (beat.getPreciseStart() == null) {
                        beat.setPreciseStart(TGDuration.toPreciseTime(beat.getStart()));
                    }
                }
            }
        }
        manager.updatePreciseStart(song);
        manager.autoCompleteSilences(song);
        return song;
    }

    private TGTrack chooseTrack(TGSong song) {
        TGTrack fallback = null;
        Iterator<TGTrack> iterator = song.getTracks();
        while (iterator.hasNext()) {
            TGTrack track = iterator.next();
            if (!track.isPercussion() && track.stringCount() >= 4 && track.stringCount() <= 8) {
                if (track.stringCount() == 6) return track;
                if (fallback == null) fallback = track;
            }
        }
        return fallback;
    }

    private AnnotationCollector createAnnotations(String sourceId, TGSong song, TGTrack selectedTrack)
            throws Exception {
        int style = TGLayout.DISPLAY_COMPACT
                | TGLayout.DISPLAY_MODE_BLACK_WHITE
                | TGLayout.DISPLAY_SCORE
                | TGLayout.DISPLAY_TABLATURE;
        TGPrintSettings settings = new TGPrintSettings();
        settings.setStyle(style);
        settings.setFromMeasure(1);
        settings.setToMeasure(song.countMeasureHeaders());
        settings.setTrackNumber(selectedTrack.getNumber());

        TGSongManager clonedManager = new TGSongManager(new TGFactoryImpl());
        TGSong clonedSong = song.clone(clonedManager.getFactory());
        TGLayoutStyles styles = new PDFLayoutStyles(this.context);
        PDFResourceFactory factory = new PDFResourceFactory();
        TGController controller = new TGPrintController(clonedSong, clonedManager, factory, styles);

        AnnotationCollector collector = new AnnotationCollector(sourceId);
        TrackingPrintDocument document = new TrackingPrintDocument(new PDFDocument(
                this.context,
                new UISize(PAGE_WIDTH, PAGE_HEIGHT),
                new UIInset(MARGIN_TOP, MARGIN_LEFT, MARGIN_RIGHT, MARGIN_BOTTOM),
                OutputStream.nullOutputStream()));
        AnnotatedPrintLayout layout = new AnnotatedPrintLayout(controller, settings, document, collector);
        layout.loadStyles(1f);
        layout.updateSong();
        layout.makeDocument(document);
        controller.getResourceBuffer().disposeAllResources();
        layout.disposeLayout();
        return collector;
    }

    private static final class AnnotatedPrintLayout extends TGPrintLayout {
        private final TrackingPrintDocument document;
        private final AnnotationCollector collector;

        AnnotatedPrintLayout(TGController controller, TGPrintSettings settings,
                TrackingPrintDocument document, AnnotationCollector collector) {
            super(controller, settings);
            this.document = document;
            this.collector = collector;
        }

        @Override
        public void paintMeasure(TGMeasureImpl measure, UIPainter painter, float spacing) {
            super.paintMeasure(measure, painter, spacing);
            recordMeasure(this.document.currentPage, measure);
        }

        private void recordMeasure(int pageIndex, TGMeasureImpl measure) {
            int measureNumber = measure.getHeader().getNumber();
            int stringCount = measure.getTrackImpl().stringCount();
            float measureWidth = measure.getWidth(this) + measure.getSpacing();
            float scoreLineSpacing = getScoreLineSpacing();
            float stringSpacing = getStringSpacing();
            float scoreOriginY = measure.getPosY()
                    + measure.getTs().getPosition(TGTrackSpacing.POSITION_SCORE_MIDDLE_LINES);
            float tabOriginY = measure.getPosY()
                    + measure.getTs().getPosition(TGTrackSpacing.POSITION_TABLATURE);

            List<Float> scoreLineY = new ArrayList<>();
            for (int line = 0; line < 5; line++) {
                scoreLineY.add(scoreOriginY + line * scoreLineSpacing);
            }
            List<Float> stringY = new ArrayList<>();
            for (int stringIndex = 0; stringIndex < stringCount; stringIndex++) {
                stringY.add(tabOriginY + stringIndex * stringSpacing);
            }

            MeasureAnnotation annotation = new MeasureAnnotation(
                    measureNumber - 1,
                    measureNumber,
                    new Rect(measure.getPosX(), measure.getPosY(), measureWidth, measure.getTs().getSize()),
                    new Rect(measure.getPosX(), scoreOriginY - scoreLineSpacing,
                            measureWidth, 6f * scoreLineSpacing),
                    scoreLineY,
                    new Rect(measure.getPosX(), tabOriginY - stringSpacing / 2f,
                            measureWidth, stringCount * stringSpacing),
                    stringY,
                    measure.getHeader().getTimeSignature().getNumerator(),
                    measure.getHeader().getTimeSignature().getDenominator().getValue());

            float contentX = measure.getPosX() + measure.getHeaderImpl().getLeftSpacing(this);
            int beatIndex = 0;
            for (TGBeat beatBase : measure.getBeats()) {
                TGBeatImpl beat = (TGBeatImpl) beatBase;
                float eventX = contentX + (2f * getScale()) + beat.getPosX() + beat.getSpacing(this);
                EventAnnotation event = new EventAnnotation(
                        String.format(Locale.ROOT, "m%03d_b%03d", measureNumber, beatIndex),
                        beat.getStart(), beat.getPreciseStart(), beatIndex, eventX,
                        beat.getPickStroke().getDirection());
                for (int voiceIndex = 0; voiceIndex < beat.countVoices(); voiceIndex++) {
                    TGVoice voiceBase = beat.getVoice(voiceIndex);
                    if (voiceBase == null) continue;
                    TGVoiceImpl voice = (TGVoiceImpl) voiceBase;
                    boolean visible = !voice.isEmpty() && !voice.isHiddenSilence();
                    TGDuration duration = voice.getDuration();
                    TGBeatGroup group = voice.getBeatGroup();
                    int direction = group == null ? voice.getDirection() : group.getDirection();
                    VoiceAnnotation voiceAnnotation = new VoiceAnnotation(
                            voiceIndex,
                            visible,
                            voice.isEmpty(),
                            voice.isRestVoice(),
                            duration.getValue(),
                            duration.isDotted(),
                            duration.isDoubleDotted(),
                            duration.getDivision().getEnters(),
                            duration.getDivision().getTimes(),
                            duration.getPreciseTime(),
                            direction,
                            voice.getJoinedType(),
                            beamCount(duration.getValue()),
                            voice.getMinY(),
                            voice.getMaxY());
                    int noteIndex = 0;
                    for (TGNote noteBase : voice.getNotes()) {
                        TGNoteImpl note = (TGNoteImpl) noteBase;
                        float centerY = scoreOriginY + note.getScorePosY();
                        float noteWidth = getScoreNoteWidth();
                        float noteHeight = scoreLineSpacing * 1.35f;
                        voiceAnnotation.notes.add(new NoteAnnotation(
                                noteIndex,
                                note.getString(),
                                note.getValue(),
                                note.isTiedNote(),
                                note.getEffect().isDeadNote(),
                                note.getEffect().isVibrato(),
                                note.getEffect().isBend(),
                                note.getEffect().isHammer(),
                                note.getEffect().isSlide(),
                                note.getEffect().isGhostNote(),
                                note.getEffect().isAccentuatedNote() || note.getEffect().isHeavyAccentuatedNote(),
                                note.getEffect().isHarmonic(),
                                note.getEffect().isGrace(),
                                note.getEffect().isPalmMute(),
                                note.getEffect().isStaccato(),
                                note.getEffect().isLetRing(),
                                note.getEffect().isTapping(),
                                centerY,
                                new Rect(eventX - noteWidth / 2f, centerY - noteHeight / 2f,
                                        noteWidth, noteHeight)));
                        noteIndex++;
                    }
                    event.voices.add(voiceAnnotation);
                }
                annotation.events.add(event);
                beatIndex++;
            }
            this.collector.add(pageIndex, annotation);
        }

        private static int beamCount(int durationValue) {
            int beams = 0;
            int value = durationValue;
            while (value > TGDuration.QUARTER) {
                beams++;
                value /= 2;
            }
            return beams;
        }
    }

    private static final class TrackingPrintDocument implements TGPrintDocument {
        private final TGPrintDocument delegate;
        int currentPage;

        TrackingPrintDocument(TGPrintDocument delegate) { this.delegate = delegate; }
        @Override public void start() { this.delegate.start(); }
        @Override public void finish() { this.delegate.finish(); }
        @Override public void pageStart() { this.currentPage++; this.delegate.pageStart(); }
        @Override public void pageFinish() { this.delegate.pageFinish(); }
        @Override public boolean isPaintable(int page) { return this.delegate.isPaintable(page); }
        @Override public boolean isTransparentBackground() { return this.delegate.isTransparentBackground(); }
        @Override public UIPainter getPainter() { return this.delegate.getPainter(); }
        @Override public UISize getSize() { return this.delegate.getSize(); }
        @Override public UIInset getMargins() { return this.delegate.getMargins(); }
    }

    private static final class AnnotationCollector {
        private final String sourceId;
        private final Map<Integer, List<MeasureAnnotation>> pages = new LinkedHashMap<>();

        AnnotationCollector(String sourceId) { this.sourceId = sourceId; }
        void add(int page, MeasureAnnotation measure) {
            this.pages.computeIfAbsent(page, ignored -> new ArrayList<>()).add(measure);
        }
        int measureCount() { return this.pages.values().stream().mapToInt(List::size).sum(); }
        int eventCount() {
            return this.pages.values().stream().flatMap(List::stream)
                    .mapToInt(measure -> measure.events.size()).sum();
        }
        int visibleVoiceCount() {
            return this.pages.values().stream().flatMap(List::stream)
                    .flatMap(measure -> measure.events.stream())
                    .flatMap(event -> event.voices.stream()).mapToInt(voice -> voice.visible ? 1 : 0).sum();
        }

        String toJson(int trackNumber) {
            StringBuilder out = new StringBuilder(256 * 1024);
            out.append("{\n")
                    .append("  \"schema_version\": \"1.0\",\n")
                    .append("  \"source_id\": ").append(quote(this.sourceId)).append(",\n")
                    .append("  \"layout\": \"score_tab\",\n")
                    .append("  \"task\": \"rhythm_events\",\n")
                    .append("  \"target_track_number\": ").append(trackNumber).append(",\n")
                    .append("  \"coordinate_space\": {\"name\": \"tuxguitar_pdf_top_left\", ")
                    .append("\"width\": 550.0, \"height\": 800.0},\n")
                    .append("  \"pages\": [\n");
            boolean firstPage = true;
            for (Map.Entry<Integer, List<MeasureAnnotation>> page : this.pages.entrySet()) {
                if (!firstPage) out.append(",\n");
                firstPage = false;
                out.append("    {\"page_index\": ").append(page.getKey()).append(", \"measures\": [\n");
                for (int i = 0; i < page.getValue().size(); i++) {
                    if (i > 0) out.append(",\n");
                    appendMeasure(out, page.getValue().get(i));
                }
                out.append("\n    ]}");
            }
            out.append("\n  ],\n")
                    .append("  \"summary\": {\"page_count\": ").append(this.pages.size())
                    .append(", \"measure_count\": ").append(measureCount())
                    .append(", \"event_count\": ").append(eventCount())
                    .append(", \"visible_voice_count\": ").append(visibleVoiceCount()).append("}\n")
                    .append("}\n");
            return out.toString();
        }

        private static void appendMeasure(StringBuilder out, MeasureAnnotation measure) {
            out.append("      {\"measure_index\":").append(measure.measureIndex)
                    .append(",\"measure_number\":").append(measure.measureNumber)
                    .append(",\"time_signature\":[").append(measure.timeNumerator).append(',')
                    .append(measure.timeDenominator).append(']')
                    .append(",\"bbox\":");
            appendRect(out, measure.bbox);
            out.append(",\"score_staff\":{\"bbox\":");
            appendRect(out, measure.scoreStaff);
            out.append(",\"line_y\":");
            appendNumbers(out, measure.scoreLineY);
            out.append("},\"tab_staff\":{\"bbox\":");
            appendRect(out, measure.tabStaff);
            out.append(",\"string_y\":");
            appendNumbers(out, measure.stringY);
            out.append("},\"events\":[");
            for (int i = 0; i < measure.events.size(); i++) {
                if (i > 0) out.append(',');
                appendEvent(out, measure.events.get(i));
            }
            out.append("]}");
        }

        private static void appendEvent(StringBuilder out, EventAnnotation event) {
            out.append("{\"event_id\":").append(quote(event.eventId))
                    .append(",\"beat_start\":").append(event.beatStart)
                    .append(",\"precise_start\":").append(event.preciseStart)
                    .append(",\"beat_index\":").append(event.beatIndex)
                    .append(",\"x\":").append(number(event.x))
                    .append(",\"pick_stroke\":").append(event.pickStroke)
                    .append(",\"voices\":[");
            for (int i = 0; i < event.voices.size(); i++) {
                if (i > 0) out.append(',');
                appendVoice(out, event.voices.get(i));
            }
            out.append("]}");
        }

        private static void appendVoice(StringBuilder out, VoiceAnnotation voice) {
            out.append("{\"voice_index\":").append(voice.voiceIndex)
                    .append(",\"visible\":").append(voice.visible)
                    .append(",\"empty\":").append(voice.empty)
                    .append(",\"rest\":").append(voice.rest)
                    .append(",\"duration_value\":").append(voice.durationValue)
                    .append(",\"dotted\":").append(voice.dotted)
                    .append(",\"double_dotted\":").append(voice.doubleDotted)
                    .append(",\"division_enters\":").append(voice.divisionEnters)
                    .append(",\"division_times\":").append(voice.divisionTimes)
                    .append(",\"precise_duration\":").append(voice.preciseDuration)
                    .append(",\"direction\":").append(voice.direction)
                    .append(",\"joined_type\":").append(voice.joinedType)
                    .append(",\"beam_count\":").append(voice.beamCount)
                    .append(",\"layout_min_y\":").append(number(voice.layoutMinY))
                    .append(",\"layout_max_y\":").append(number(voice.layoutMaxY))
                    .append(",\"notes\":[");
            for (int i = 0; i < voice.notes.size(); i++) {
                if (i > 0) out.append(',');
                NoteAnnotation note = voice.notes.get(i);
                out.append("{\"note_index\":").append(note.noteIndex)
                        .append(",\"string\":").append(note.stringNumber)
                        .append(",\"fret\":").append(note.fret)
                        .append(",\"tied\":").append(note.tied)
                        .append(",\"effects\":{")
                        .append("\"dead\":").append(note.dead).append(',')
                        .append("\"vibrato\":").append(note.vibrato).append(',')
                        .append("\"bend\":").append(note.bend).append(',')
                        .append("\"hammer\":").append(note.hammer).append(',')
                        .append("\"slide\":").append(note.slide).append(',')
                        .append("\"ghost\":").append(note.ghost).append(',')
                        .append("\"accent\":").append(note.accent).append(',')
                        .append("\"harmonic\":").append(note.harmonic).append(',')
                        .append("\"grace\":").append(note.grace).append(',')
                        .append("\"palm_mute\":").append(note.palmMute).append(',')
                        .append("\"staccato\":").append(note.staccato).append(',')
                        .append("\"let_ring\":").append(note.letRing).append(',')
                        .append("\"tapping\":").append(note.tapping).append('}')
                        .append(",\"center_y\":").append(number(note.centerY))
                        .append(",\"bbox\":");
                appendRect(out, note.bbox);
                out.append('}');
            }
            out.append("]}");
        }

        private static void appendNumbers(StringBuilder out, List<Float> values) {
            out.append('[');
            for (int i = 0; i < values.size(); i++) {
                if (i > 0) out.append(',');
                out.append(number(values.get(i)));
            }
            out.append(']');
        }

        private static void appendRect(StringBuilder out, Rect rect) {
            out.append('[').append(number(rect.x)).append(',').append(number(rect.y)).append(',')
                    .append(number(rect.width)).append(',').append(number(rect.height)).append(']');
        }
    }

    private static String number(float value) { return String.format(Locale.ROOT, "%.4f", value); }
    private static String quote(String value) {
        if (value == null) return "null";
        return '"' + value.replace("\\", "\\\\").replace("\"", "\\\"") + '"';
    }

    private record Rect(float x, float y, float width, float height) {}
    private record NoteAnnotation(int noteIndex, int stringNumber, int fret, boolean tied,
            boolean dead, boolean vibrato, boolean bend, boolean hammer, boolean slide,
            boolean ghost, boolean accent, boolean harmonic, boolean grace, boolean palmMute,
            boolean staccato, boolean letRing, boolean tapping,
            float centerY, Rect bbox) {}

    private static final class VoiceAnnotation {
        final int voiceIndex;
        final boolean visible;
        final boolean empty;
        final boolean rest;
        final int durationValue;
        final boolean dotted;
        final boolean doubleDotted;
        final int divisionEnters;
        final int divisionTimes;
        final long preciseDuration;
        final int direction;
        final int joinedType;
        final int beamCount;
        final float layoutMinY;
        final float layoutMaxY;
        final List<NoteAnnotation> notes = new ArrayList<>();

        VoiceAnnotation(int voiceIndex, boolean visible, boolean empty, boolean rest,
                int durationValue, boolean dotted, boolean doubleDotted, int divisionEnters,
                int divisionTimes, long preciseDuration, int direction, int joinedType,
                int beamCount, float layoutMinY, float layoutMaxY) {
            this.voiceIndex = voiceIndex;
            this.visible = visible;
            this.empty = empty;
            this.rest = rest;
            this.durationValue = durationValue;
            this.dotted = dotted;
            this.doubleDotted = doubleDotted;
            this.divisionEnters = divisionEnters;
            this.divisionTimes = divisionTimes;
            this.preciseDuration = preciseDuration;
            this.direction = direction;
            this.joinedType = joinedType;
            this.beamCount = beamCount;
            this.layoutMinY = layoutMinY;
            this.layoutMaxY = layoutMaxY;
        }
    }

    private static final class EventAnnotation {
        final String eventId;
        final long beatStart;
        final long preciseStart;
        final int beatIndex;
        final float x;
        final int pickStroke;
        final List<VoiceAnnotation> voices = new ArrayList<>();

        EventAnnotation(String eventId, long beatStart, long preciseStart, int beatIndex, float x,
                int pickStroke) {
            this.eventId = eventId;
            this.beatStart = beatStart;
            this.preciseStart = preciseStart;
            this.beatIndex = beatIndex;
            this.x = x;
            this.pickStroke = pickStroke;
        }
    }

    private static final class MeasureAnnotation {
        final int measureIndex;
        final int measureNumber;
        final Rect bbox;
        final Rect scoreStaff;
        final List<Float> scoreLineY;
        final Rect tabStaff;
        final List<Float> stringY;
        final int timeNumerator;
        final int timeDenominator;
        final List<EventAnnotation> events = new ArrayList<>();

        MeasureAnnotation(int measureIndex, int measureNumber, Rect bbox, Rect scoreStaff,
                List<Float> scoreLineY, Rect tabStaff, List<Float> stringY,
                int timeNumerator, int timeDenominator) {
            this.measureIndex = measureIndex;
            this.measureNumber = measureNumber;
            this.bbox = bbox;
            this.scoreStaff = scoreStaff;
            this.scoreLineY = scoreLineY;
            this.tabStaff = tabStaff;
            this.stringY = stringY;
            this.timeNumerator = timeNumerator;
            this.timeDenominator = timeDenominator;
        }
    }

    private static void usage() {
        System.out.println("Usage: TuxGuitarScoreRhythmAnnotationBuilder <database-root> [limit]");
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            usage();
            System.exit(2);
        }
        Path databaseRoot = Path.of(args[0]).toAbsolutePath().normalize();
        int limit = args.length >= 2 ? Integer.parseInt(args[1]) : 0;
        TuxGuitarScoreRhythmAnnotationBuilder builder = new TuxGuitarScoreRhythmAnnotationBuilder();
        try {
            builder.run(databaseRoot, limit);
        } finally {
            builder.disconnectPlugins();
        }
    }
}
