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

import app.tuxguitar.graphics.control.TGBeatImpl;
import app.tuxguitar.graphics.control.TGController;
import app.tuxguitar.graphics.control.TGFactoryImpl;
import app.tuxguitar.graphics.control.TGLayout;
import app.tuxguitar.graphics.control.TGLayoutStyles;
import app.tuxguitar.graphics.control.TGMeasureImpl;
import app.tuxguitar.graphics.control.TGNoteImpl;
import app.tuxguitar.graphics.control.TGTrackSpacing;
import app.tuxguitar.graphics.control.painters.TGNumberPainter;
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
 * Emits page-space ground truth for TuxGuitar's tab_only PDF layout.
 *
 * Coordinates use the same top-left 550 x 800 logical page space passed to
 * TGPrintLayout. The companion Python builder converts them to actual PNG
 * pixels and creates visual overlays.
 */
public final class TuxGuitarTabAnnotationBuilder {
    private static final float PAGE_WIDTH = 550f;
    private static final float PAGE_HEIGHT = 800f;
    private static final float MARGIN_TOP = 20f;
    private static final float MARGIN_BOTTOM = 20f;
    private static final float MARGIN_LEFT = 20f;
    private static final float MARGIN_RIGHT = 20f;

    private final TGContext context;
    private final List<TGPlugin> plugins;

    private TuxGuitarTabAnnotationBuilder() throws Exception {
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
                // The process is exiting; preserve completed annotation files.
            }
        }
    }

    private void run(Path databaseRoot, int limit, String layoutName) throws Exception {
        Path sourceRoot = databaseRoot.resolve("source/gp");
        if (!layoutName.equals("tab_only") && !layoutName.equals("score_tab")) {
            throw new IllegalArgumentException("Unsupported layout: " + layoutName);
        }
        Path outputRoot = databaseRoot.resolve(
                layoutName.equals("tab_only") ? "labels/layout/tab_only" : "labels/layout/score_tab_symbols");
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
        int symbolCount = 0;
        for (Path source : sources) {
            String fileName = source.getFileName().toString();
            int dot = fileName.lastIndexOf('.');
            String sourceId = dot > 0 ? fileName.substring(0, dot) : fileName;
            SongLoad loaded = readSong(source);
            TGTrack track = chooseTrack(loaded.song());
            if (track == null) {
                throw new IOException("No supported guitar/bass track in " + source);
            }

            AnnotationCollector collector = createAnnotations(sourceId, loaded.song(), track, layoutName);
            Path output = outputRoot.resolve(sourceId + ".json");
            Files.writeString(output, collector.toJson(track.getNumber()), StandardCharsets.UTF_8);

            sourceCount++;
            pageCount += collector.pages.size();
            measureCount += collector.measureCount();
            symbolCount += collector.symbolCount();
            System.out.printf(Locale.ROOT,
                    "ANNOTATED source=%s pages=%d measures=%d symbols=%d%n",
                    sourceId, collector.pages.size(), collector.measureCount(), collector.symbolCount());
        }
        System.out.printf(Locale.ROOT,
                "COMPLETE sources=%d pages=%d measures=%d symbols=%d output=%s%n",
                sourceCount, pageCount, measureCount, symbolCount, outputRoot);
    }

    private SongLoad readSong(Path input) throws Exception {
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
        return new SongLoad(song, manager);
    }

    private TGTrack chooseTrack(TGSong song) {
        TGTrack fallback = null;
        Iterator<TGTrack> iterator = song.getTracks();
        while (iterator.hasNext()) {
            TGTrack track = iterator.next();
            if (!track.isPercussion() && track.stringCount() >= 4 && track.stringCount() <= 8) {
                if (track.stringCount() == 6) {
                    return track;
                }
                if (fallback == null) {
                    fallback = track;
                }
            }
        }
        return fallback;
    }

    private AnnotationCollector createAnnotations(
            String sourceId, TGSong song, TGTrack selectedTrack, String layoutName)
            throws Exception {
        int style = TGLayout.DISPLAY_COMPACT
                | TGLayout.DISPLAY_MODE_BLACK_WHITE
                | TGLayout.DISPLAY_TABLATURE;
        if (layoutName.equals("score_tab")) {
            style |= TGLayout.DISPLAY_SCORE;
        }
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

        AnnotationCollector collector = new AnnotationCollector(sourceId, layoutName);
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

        AnnotatedPrintLayout(
                TGController controller,
                TGPrintSettings settings,
                TrackingPrintDocument document,
                AnnotationCollector collector) {
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
            float stringSpacing = getStringSpacing();
            float tabTop = measure.getPosY()
                    + measure.getTs().getPosition(TGTrackSpacing.POSITION_TABLATURE);
            float measureWidth = measure.getWidth(this) + measure.getSpacing();
            Rect staff = new Rect(
                    measure.getPosX(),
                    tabTop - (stringSpacing / 2f),
                    measureWidth,
                    ((stringCount - 1) * stringSpacing) + stringSpacing);
            List<Float> stringY = new ArrayList<>();
            for (int stringIndex = 0; stringIndex < stringCount; stringIndex++) {
                stringY.add(tabTop + (stringIndex * stringSpacing));
            }

            MeasureAnnotation annotation = new MeasureAnnotation(
                    measureNumber - 1,
                    measureNumber,
                    new Rect(measure.getPosX(), measure.getPosY(), measureWidth, measure.getTs().getSize()),
                    staff,
                    stringY);

            float contentX = measure.getPosX() + measure.getHeaderImpl().getLeftSpacing(this);
            int beatIndex = 0;
            for (TGBeat beatBase : measure.getBeats()) {
                TGBeatImpl beat = (TGBeatImpl) beatBase;
                float eventX = contentX + (2f * getScale()) + beat.getPosX() + beat.getSpacing(this);
                EventAnnotation event = new EventAnnotation(
                        String.format(Locale.ROOT, "m%03d_b%03d", measureNumber, beatIndex),
                        beat.getStart(), beat.getPreciseStart(), beatIndex, eventX);
                for (int voiceIndex = 0; voiceIndex < beat.countVoices(); voiceIndex++) {
                    TGVoice voice = beat.getVoice(voiceIndex);
                    if (voice != null) {
                        TGDuration duration = voice.getDuration();
                        event.voices.add(new VoiceAnnotation(
                                voiceIndex, !voice.isEmpty(), voice.isRestVoice(),
                                duration.getValue(), duration.isDotted(), duration.isDoubleDotted(),
                                duration.getDivision().getEnters(), duration.getDivision().getTimes()));
                    }
                    if (voice == null || voice.isEmpty() || voice.isRestVoice()) {
                        continue;
                    }
                    int noteIndex = 0;
                    for (TGNote noteBase : voice.getNotes()) {
                        TGNoteImpl note = (TGNoteImpl) noteBase;
                        if (!note.isTiedNote()) {
                            addNoteSymbols(annotation, note, measureNumber, beatIndex, voiceIndex,
                                    noteIndex, beat.getStart(), eventX, tabTop + note.getTabPosY());
                        }
                        noteIndex++;
                    }
                }
                annotation.events.add(event);
                beatIndex++;
            }
            this.collector.add(pageIndex, annotation);
        }

        private void addNoteSymbols(
                MeasureAnnotation measure,
                TGNoteImpl note,
                int measureNumber,
                int beatIndex,
                int voiceIndex,
                int noteIndex,
                long beatStart,
                float centerX,
                float centerY) {
            float noteSize = getStringSpacing() - 2f;
            String eventId = String.format(Locale.ROOT, "m%03d_b%03d_v%d",
                    measureNumber, beatIndex, voiceIndex);
            if (note.getEffect().isDeadNote()) {
                float width = 6f * getScale();
                measure.symbols.add(new SymbolAnnotation(
                        "dead_x", new Rect(centerX - width / 2f, centerY - noteSize / 2f, width, noteSize),
                        centerX, centerY, eventId, beatStart, beatIndex, voiceIndex, noteIndex,
                        note.getString(), note.getValue(), 0, 1, note.getEffect().isGhostNote()));
                return;
            }

            List<Integer> digits = TGNumberPainter.getDigits(note.getValue());
            float fullWidth = TGNumberPainter.getDigitsWidth(note.getValue(), noteSize);
            float cursorX = centerX - (fullWidth / 2f);
            for (int digitIndex = 0; digitIndex < digits.size(); digitIndex++) {
                int digit = digits.get(digitIndex);
                float width = TGNumberPainter.get(digit).getWidth() * noteSize;
                measure.symbols.add(new SymbolAnnotation(
                        "digit_" + digit, new Rect(cursorX, centerY - noteSize / 2f, width, noteSize),
                        cursorX + width / 2f, centerY, eventId, beatStart, beatIndex, voiceIndex, noteIndex,
                        note.getString(), note.getValue(), digitIndex, digits.size(),
                        note.getEffect().isGhostNote()));
                cursorX += width + (0.1f * noteSize);
            }
        }
    }

    private static final class TrackingPrintDocument implements TGPrintDocument {
        private final TGPrintDocument delegate;
        int currentPage;

        TrackingPrintDocument(TGPrintDocument delegate) {
            this.delegate = delegate;
        }

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
        private final String layoutName;
        private final Map<Integer, List<MeasureAnnotation>> pages = new LinkedHashMap<>();

        AnnotationCollector(String sourceId, String layoutName) {
            this.sourceId = sourceId;
            this.layoutName = layoutName;
        }

        void add(int page, MeasureAnnotation measure) {
            this.pages.computeIfAbsent(page, ignored -> new ArrayList<>()).add(measure);
        }

        int measureCount() {
            return this.pages.values().stream().mapToInt(List::size).sum();
        }

        int symbolCount() {
            return this.pages.values().stream()
                    .flatMap(List::stream)
                    .mapToInt(measure -> measure.symbols.size())
                    .sum();
        }

        String toJson(int trackNumber) {
            StringBuilder out = new StringBuilder(128 * 1024);
            out.append("{\n");
            out.append("  \"schema_version\": \"1.0\",\n");
            out.append("  \"source_id\": ").append(quote(this.sourceId)).append(",\n");
            out.append("  \"layout\": ").append(quote(this.layoutName)).append(",\n");
            out.append("  \"target_track_number\": ").append(trackNumber).append(",\n");
            out.append("  \"coordinate_space\": {\"name\": \"tuxguitar_pdf_top_left\", ")
                    .append("\"width\": 550.0, \"height\": 800.0},\n");
            out.append("  \"pages\": [\n");
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
            out.append("\n  ],\n");
            out.append("  \"summary\": {\"page_count\": ").append(this.pages.size())
                    .append(", \"measure_count\": ").append(measureCount())
                    .append(", \"symbol_count\": ").append(symbolCount()).append("}\n");
            out.append("}\n");
            return out.toString();
        }

        private static void appendMeasure(StringBuilder out, MeasureAnnotation measure) {
            out.append("      {\"measure_index\": ").append(measure.measureIndex)
                    .append(", \"measure_number\": ").append(measure.measureNumber)
                    .append(", \"bbox\": ");
            appendRect(out, measure.bbox);
            out.append(", \"tab_staff\": {\"bbox\": ");
            appendRect(out, measure.tabStaff);
            out.append(", \"string_y\": [");
            for (int i = 0; i < measure.stringY.size(); i++) {
                if (i > 0) out.append(',');
                out.append(number(measure.stringY.get(i)));
            }
            out.append("]}, \"symbols\": [");
            for (int i = 0; i < measure.symbols.size(); i++) {
                if (i > 0) out.append(',');
                appendSymbol(out, measure.symbols.get(i));
            }
            out.append("], \"events\": [");
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
                    .append(",\"x\":").append(number(event.x)).append(",\"voices\":[");
            for (int i = 0; i < event.voices.size(); i++) {
                if (i > 0) out.append(',');
                VoiceAnnotation voice = event.voices.get(i);
                out.append("{\"voice_index\":").append(voice.voiceIndex)
                        .append(",\"visible\":").append(voice.visible)
                        .append(",\"rest\":").append(voice.rest)
                        .append(",\"duration_value\":").append(voice.durationValue)
                        .append(",\"dotted\":").append(voice.dotted)
                        .append(",\"double_dotted\":").append(voice.doubleDotted)
                        .append(",\"division_enters\":").append(voice.divisionEnters)
                        .append(",\"division_times\":").append(voice.divisionTimes).append('}');
            }
            out.append("]}");
        }

        private static void appendSymbol(StringBuilder out, SymbolAnnotation symbol) {
            out.append("{\"class\":").append(quote(symbol.className)).append(",\"bbox\":");
            appendRect(out, symbol.bbox);
            out.append(",\"center\":[").append(number(symbol.centerX)).append(',')
                    .append(number(symbol.centerY)).append(']')
                    .append(",\"event_id\":").append(quote(symbol.eventId))
                    .append(",\"beat_start\":").append(symbol.beatStart)
                    .append(",\"beat_index\":").append(symbol.beatIndex)
                    .append(",\"voice_index\":").append(symbol.voiceIndex)
                    .append(",\"note_index\":").append(symbol.noteIndex)
                    .append(",\"string\":").append(symbol.stringNumber)
                    .append(",\"fret\":").append(symbol.fret)
                    .append(",\"glyph_index\":").append(symbol.glyphIndex)
                    .append(",\"glyph_count\":").append(symbol.glyphCount)
                    .append(",\"ghost\":").append(symbol.ghost)
                    .append('}');
        }

        private static void appendRect(StringBuilder out, Rect rect) {
            out.append('[').append(number(rect.x)).append(',').append(number(rect.y)).append(',')
                    .append(number(rect.width)).append(',').append(number(rect.height)).append(']');
        }
    }

    private static String number(float value) {
        return String.format(Locale.ROOT, "%.4f", value);
    }

    private static String quote(String value) {
        if (value == null) return "null";
        return '"' + value.replace("\\", "\\\\").replace("\"", "\\\"") + '"';
    }

    private record SongLoad(TGSong song, TGSongManager manager) {}
    private record Rect(float x, float y, float width, float height) {}
    private record SymbolAnnotation(
            String className, Rect bbox, float centerX, float centerY, String eventId,
            long beatStart, int beatIndex, int voiceIndex, int noteIndex, int stringNumber,
            int fret, int glyphIndex, int glyphCount, boolean ghost) {}
    private record VoiceAnnotation(int voiceIndex, boolean visible, boolean rest,
            int durationValue, boolean dotted, boolean doubleDotted,
            int divisionEnters, int divisionTimes) {}

    private static final class EventAnnotation {
        final String eventId;
        final long beatStart;
        final long preciseStart;
        final int beatIndex;
        final float x;
        final List<VoiceAnnotation> voices = new ArrayList<>();

        EventAnnotation(String eventId, long beatStart, long preciseStart, int beatIndex, float x) {
            this.eventId = eventId;
            this.beatStart = beatStart;
            this.preciseStart = preciseStart;
            this.beatIndex = beatIndex;
            this.x = x;
        }
    }

    private static final class MeasureAnnotation {
        final int measureIndex;
        final int measureNumber;
        final Rect bbox;
        final Rect tabStaff;
        final List<Float> stringY;
        final List<SymbolAnnotation> symbols = new ArrayList<>();
        final List<EventAnnotation> events = new ArrayList<>();

        MeasureAnnotation(int measureIndex, int measureNumber, Rect bbox, Rect tabStaff, List<Float> stringY) {
            this.measureIndex = measureIndex;
            this.measureNumber = measureNumber;
            this.bbox = bbox;
            this.tabStaff = tabStaff;
            this.stringY = stringY;
        }
    }

    private static void usage() {
        System.out.println("Usage: TuxGuitarTabAnnotationBuilder <database-root> [limit] [tab_only|score_tab]");
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            usage();
            System.exit(2);
        }
        Path databaseRoot = Path.of(args[0]).toAbsolutePath().normalize();
        int limit = args.length >= 2 ? Integer.parseInt(args[1]) : 0;
        String layoutName = args.length >= 3 ? args[2] : "tab_only";
        TuxGuitarTabAnnotationBuilder builder = new TuxGuitarTabAnnotationBuilder();
        try {
            builder.run(databaseRoot, limit, layoutName);
        } finally {
            builder.disconnectPlugins();
        }
    }
}
