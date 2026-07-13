import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import app.tuxguitar.graphics.control.TGLayout;
import app.tuxguitar.graphics.control.print.TGPrintSettings;
import app.tuxguitar.io.base.TGFileFormatManager;
import app.tuxguitar.io.base.TGFileFormatUtils;
import app.tuxguitar.io.base.TGSongReaderHandle;
import app.tuxguitar.io.base.TGSongReaderHelper;
import app.tuxguitar.io.base.TGSongStreamContext;
import app.tuxguitar.io.base.TGSongWriterHandle;
import app.tuxguitar.io.gtp.GP1InputStreamPlugin;
import app.tuxguitar.io.gtp.GP2InputStreamPlugin;
import app.tuxguitar.io.gtp.GP3InputStreamPlugin;
import app.tuxguitar.io.gtp.GP4InputStreamPlugin;
import app.tuxguitar.io.gtp.GP5InputStreamPlugin;
import app.tuxguitar.io.pdf.PDFSongWriter;
import app.tuxguitar.io.pdf.PDFSongWriterPlugin;
import app.tuxguitar.song.managers.TGSongManager;
import app.tuxguitar.song.models.TGBeat;
import app.tuxguitar.song.models.TGChord;
import app.tuxguitar.song.models.TGDuration;
import app.tuxguitar.song.models.TGMeasure;
import app.tuxguitar.song.models.TGMeasureHeader;
import app.tuxguitar.song.models.TGNote;
import app.tuxguitar.song.models.TGNoteEffect;
import app.tuxguitar.song.models.TGSong;
import app.tuxguitar.song.models.TGString;
import app.tuxguitar.song.models.TGTrack;
import app.tuxguitar.song.models.TGVoice;
import app.tuxguitar.util.TGContext;
import app.tuxguitar.util.plugin.TGPlugin;

/**
 * Deterministic, headless dataset exporter for TuxGuitar 2.0.1.
 *
 * It selects compact guitar songs from a large GP corpus, keeps the original
 * file as ground truth, exports three common notation layouts to PDF, and
 * emits a semantic JSON label for the selected guitar track.
 */
public class TuxGuitarDatasetBuilder {
    private static final List<String> EXTENSIONS = List.of("gp3", "gp4", "gp5", "gtp", "gpx");
    private static final int MIN_MEASURES = 4;
    private static final int MAX_MEASURES = 48;
    private static final int MIN_NOTES = 16;
    private static final int MAX_NOTES = 4000;
    private static final int MAX_ATTEMPTS_PER_FORMAT = 500;
    private static final long SHUFFLE_SEED = 20260713L;

    private static final Map<String, Integer> STYLES = new LinkedHashMap<>();
    static {
        int base = TGLayout.DISPLAY_COMPACT | TGLayout.DISPLAY_MODE_BLACK_WHITE;
        STYLES.put("tab_only", base | TGLayout.DISPLAY_TABLATURE);
        STYLES.put("score_tab", base | TGLayout.DISPLAY_SCORE | TGLayout.DISPLAY_TABLATURE);
        STYLES.put("score_only", base | TGLayout.DISPLAY_SCORE);
    }

    private final TGContext context;
    private final List<TGPlugin> plugins;
    private final Set<String> selectedHashes;
    private final List<String> manifestLines;
    private final Path sourceRoot;
    private final Path outputRoot;
    private final int perFormat;

    private TuxGuitarDatasetBuilder(Path sourceRoot, Path outputRoot, int perFormat) throws Exception {
        this.sourceRoot = sourceRoot;
        this.outputRoot = outputRoot;
        this.perFormat = perFormat;
        this.context = new TGContext();
        this.plugins = new ArrayList<>();
        this.selectedHashes = new HashSet<>();
        this.manifestLines = new ArrayList<>();
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
                // Process is terminating; a failed cleanup must not hide build results.
            }
        }
    }

    private void run() throws Exception {
        createDirectories();
        Map<String, List<Path>> candidates = discoverCandidates();
        Map<String, Integer> accepted = new LinkedHashMap<>();
        Map<String, Integer> attempted = new LinkedHashMap<>();

        for (String extension : EXTENSIONS) {
            accepted.put(extension, 0);
            attempted.put(extension, 0);
            List<Path> paths = candidates.getOrDefault(extension, List.of());
            for (Path path : paths) {
                if (accepted.get(extension) >= this.perFormat || attempted.get(extension) >= MAX_ATTEMPTS_PER_FORMAT) {
                    break;
                }
                attempted.put(extension, attempted.get(extension) + 1);
                try {
                    if (buildOne(path, extension)) {
                        accepted.put(extension, accepted.get(extension) + 1);
                        System.out.printf(Locale.ROOT, "ACCEPT %s %d/%d %s%n", extension,
                                accepted.get(extension), this.perFormat, path.getFileName());
                    }
                } catch (Throwable error) {
                    System.err.printf(Locale.ROOT, "SKIP %s: %s%n", path, error.getMessage());
                }
            }
        }

        Path manifest = this.outputRoot.resolve("manifests/sources.jsonl");
        Files.write(manifest, this.manifestLines, StandardCharsets.UTF_8);
        writeBuildSummary(accepted, attempted);

        int total = accepted.values().stream().mapToInt(Integer::intValue).sum();
        if (total == 0) {
            throw new IllegalStateException("No valid source scores were exported");
        }
        System.out.printf(Locale.ROOT, "DONE sources=%d manifest=%s%n", total, manifest);
    }

    private void createDirectories() throws IOException {
        Files.createDirectories(this.outputRoot.resolve("source/gp"));
        Files.createDirectories(this.outputRoot.resolve("labels/songs"));
        Files.createDirectories(this.outputRoot.resolve("manifests"));
        Files.createDirectories(this.outputRoot.resolve("logs"));
        for (String style : STYLES.keySet()) {
            Files.createDirectories(this.outputRoot.resolve("output/pdf").resolve(style));
        }
    }

    private Map<String, List<Path>> discoverCandidates() throws IOException {
        Map<String, List<Path>> candidates = new HashMap<>();
        for (String extension : EXTENSIONS) {
            candidates.put(extension, new ArrayList<>());
        }
        try (var stream = Files.walk(this.sourceRoot, 3)) {
            stream.filter(Files::isRegularFile).forEach(path -> {
                String extension = extension(path);
                if (candidates.containsKey(extension)) {
                    try {
                        long size = Files.size(path);
                        if (size >= 2_000 && size <= 2_000_000) {
                            candidates.get(extension).add(path);
                        }
                    } catch (IOException ignored) {
                        // Ignore unreadable candidate.
                    }
                }
            });
        }
        for (Map.Entry<String, List<Path>> entry : candidates.entrySet()) {
            entry.getValue().sort(Comparator.comparing(Path::toString));
            Collections.shuffle(entry.getValue(), new Random(SHUFFLE_SEED + entry.getKey().hashCode()));
            System.out.printf(Locale.ROOT, "CANDIDATES %s=%d%n", entry.getKey(), entry.getValue().size());
        }
        return candidates;
    }

    private boolean buildOne(Path input, String extension) throws Exception {
        String hash = sha256(input);
        if (!this.selectedHashes.add(hash)) {
            return false;
        }

        SongLoad loaded = readSong(input);
        TGTrack track = chooseTrack(loaded.song);
        if (track == null) {
            return false;
        }
        SongStats stats = collectStats(track);
        int measures = loaded.song.countMeasureHeaders();
        if (measures < MIN_MEASURES || measures > MAX_MEASURES ||
                stats.noteCount < MIN_NOTES || stats.noteCount > MAX_NOTES) {
            return false;
        }

        String id = hash.substring(0, 16);
        Path sourceCopy = this.outputRoot.resolve("source/gp").resolve(id + "." + extension);
        Path labelPath = this.outputRoot.resolve("labels/songs").resolve(id + ".json");
        List<Path> written = new ArrayList<>();
        try {
            Files.copy(input, sourceCopy, StandardCopyOption.REPLACE_EXISTING);
            written.add(sourceCopy);
            Files.writeString(labelPath, songToJson(id, hash, input, loaded.song, track, stats), StandardCharsets.UTF_8);
            written.add(labelPath);

            for (Map.Entry<String, Integer> style : STYLES.entrySet()) {
                Path pdf = this.outputRoot.resolve("output/pdf").resolve(style.getKey()).resolve(id + ".pdf");
                exportPdf(loaded.song, loaded.manager, track, style.getValue(), pdf);
                if (!Files.isRegularFile(pdf) || Files.size(pdf) < 1_000) {
                    throw new IOException("PDF export produced an empty file: " + pdf);
                }
                written.add(pdf);
            }

            this.manifestLines.add(sourceManifestJson(id, hash, extension, input, sourceCopy, labelPath,
                    loaded.song, track, stats));
            return true;
        } catch (Throwable error) {
            for (Path path : written) {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException ignored) {
                    // Best effort cleanup within the generated database only.
                }
            }
            throw error;
        }
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
            throw new IOException("Empty song");
        }
        // Legacy .gtp readers populate the approximate MIDI-tick start but may
        // leave preciseStart null.  Newer TuxGuitar layout code expects every
        // beat to have a precise value before sorting/normalising the song.
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

    private SongStats collectStats(TGTrack track) {
        SongStats stats = new SongStats();
        for (int measureIndex = 0; measureIndex < track.countMeasures(); measureIndex++) {
            TGMeasure measure = track.getMeasure(measureIndex);
            for (TGBeat beat : measure.getBeats()) {
                if (beat.isChordBeat()) {
                    stats.chordBeatCount++;
                }
                for (int voiceIndex = 0; voiceIndex < beat.countVoices(); voiceIndex++) {
                    TGVoice voice = beat.getVoice(voiceIndex);
                    if (voice == null || voice.isEmpty()) {
                        continue;
                    }
                    if (voice.isRestVoice()) {
                        stats.restVoiceCount++;
                    }
                    for (TGNote note : voice.getNotes()) {
                        stats.noteCount++;
                        stats.maxFret = Math.max(stats.maxFret, note.getValue());
                        if (note.getValue() >= 10) {
                            stats.multiDigitFretCount++;
                        }
                        if (note.getEffect().hasAnyEffect()) {
                            stats.effectNoteCount++;
                        }
                    }
                }
            }
        }
        return stats;
    }

    private void exportPdf(TGSong song, TGSongManager manager, TGTrack track, int style, Path output) throws Exception {
        TGPrintSettings settings = new TGPrintSettings();
        settings.setStyle(style);
        settings.setFromMeasure(1);
        settings.setToMeasure(song.countMeasureHeaders());
        settings.setTrackNumber(track.getNumber());

        TGSongStreamContext streamContext = new TGSongStreamContext();
        streamContext.setAttribute(TGPrintSettings.ATTRIBUTE_PRINT_STYLES, settings);
        streamContext.setAttribute(TGPrintSettings.ATTRIBUTE_PRINT_ZOOM, Integer.valueOf(100));

        TGSongWriterHandle writer = new TGSongWriterHandle();
        writer.setSong(song);
        writer.setFactory(manager.getFactory());
        writer.setFormat(PDFSongWriter.FILE_FORMAT);
        writer.setContext(streamContext);
        writer.setOutputStream(new BufferedOutputStream(new FileOutputStream(output.toFile())));
        TGFileFormatManager.getInstance(this.context).write(writer);
    }

    private String songToJson(String id, String hash, Path original, TGSong song, TGTrack track, SongStats stats) {
        StringBuilder json = new StringBuilder(64 * 1024);
        json.append("{\n");
        field(json, "schema_version", "1.0", true, 1);
        field(json, "id", id, true, 1);
        field(json, "sha256", hash, true, 1);
        field(json, "original_path", original.toString(), true, 1);
        json.append("  \"song\": {\n");
        field(json, "name", song.getName(), true, 2);
        field(json, "artist", song.getArtist(), true, 2);
        field(json, "album", song.getAlbum(), true, 2);
        field(json, "author", song.getAuthor(), false, 2);
        json.append("  },\n");
        json.append("  \"target_track\": {\n");
        field(json, "number", track.getNumber(), true, 2);
        field(json, "name", track.getName(), true, 2);
        field(json, "offset", track.getOffset(), true, 2);
        field(json, "string_count", track.stringCount(), true, 2);
        field(json, "max_fret", track.getMaxFret(), true, 2);
        json.append("    \"tuning\": [");
        for (int i = 0; i < track.getStrings().size(); i++) {
            TGString string = track.getStrings().get(i);
            if (i > 0) json.append(',');
            json.append("{\"string\":").append(string.getNumber())
                    .append(",\"midi_pitch\":").append(string.getValue()).append('}');
        }
        json.append("]\n  },\n");
        json.append("  \"statistics\": {");
        json.append("\"measure_count\":").append(song.countMeasureHeaders()).append(',');
        json.append("\"note_count\":").append(stats.noteCount).append(',');
        json.append("\"rest_voice_count\":").append(stats.restVoiceCount).append(',');
        json.append("\"chord_beat_count\":").append(stats.chordBeatCount).append(',');
        json.append("\"effect_note_count\":").append(stats.effectNoteCount).append(',');
        json.append("\"multi_digit_fret_count\":").append(stats.multiDigitFretCount).append(',');
        json.append("\"highest_fret\":").append(stats.maxFret).append("},\n");
        json.append("  \"measures\": [\n");

        for (int measureIndex = 0; measureIndex < track.countMeasures(); measureIndex++) {
            if (measureIndex > 0) json.append(",\n");
            TGMeasure measure = track.getMeasure(measureIndex);
            TGMeasureHeader header = measure.getHeader();
            json.append("    {\"index\":").append(measureIndex)
                    .append(",\"number\":").append(header.getNumber())
                    .append(",\"time_signature\":{\"numerator\":")
                    .append(header.getTimeSignature().getNumerator())
                    .append(",\"denominator\":")
                    .append(header.getTimeSignature().getDenominator().getValue())
                    .append("},\"tempo_quarter\":").append(header.getTempo().getQuarterValue())
                    .append(",\"repeat_open\":").append(header.isRepeatOpen())
                    .append(",\"repeat_close\":").append(header.getRepeatClose())
                    .append(",\"beats\":[");

            boolean firstBeat = true;
            for (TGBeat beat : measure.getBeats()) {
                if (!firstBeat) json.append(',');
                firstBeat = false;
                Long preciseStart = beat.getPreciseStart();
                if (preciseStart == null) {
                    preciseStart = TGDuration.toPreciseTime(beat.getStart());
                }
                json.append("{\"start\":").append(beat.getStart())
                        .append(",\"precise_start\":").append(preciseStart)
                        .append(",\"chord_name\":");
                TGChord chord = beat.getChord();
                json.append(chord == null ? "null" : quote(chord.getName()));
                json.append(",\"voices\":[");

                boolean firstVoice = true;
                for (int voiceIndex = 0; voiceIndex < beat.countVoices(); voiceIndex++) {
                    TGVoice voice = beat.getVoice(voiceIndex);
                    if (voice == null || voice.isEmpty()) continue;
                    if (!firstVoice) json.append(',');
                    firstVoice = false;
                    TGDuration duration = voice.getDuration();
                    json.append("{\"index\":").append(voice.getIndex())
                            .append(",\"rest\":").append(voice.isRestVoice())
                            .append(",\"duration\":{\"value\":").append(duration.getValue())
                            .append(",\"dotted\":").append(duration.isDotted())
                            .append(",\"double_dotted\":").append(duration.isDoubleDotted())
                            .append(",\"division_enters\":").append(duration.getDivision().getEnters())
                            .append(",\"division_times\":").append(duration.getDivision().getTimes())
                            .append(",\"precise_time\":").append(duration.getPreciseTime())
                            .append("},\"notes\":[");

                    for (int noteIndex = 0; noteIndex < voice.countNotes(); noteIndex++) {
                        if (noteIndex > 0) json.append(',');
                        TGNote note = voice.getNote(noteIndex);
                        int tuning = track.getString(note.getString()).getValue();
                        TGNoteEffect effect = note.getEffect();
                        json.append("{\"string\":").append(note.getString())
                                .append(",\"fret\":").append(note.getValue())
                                .append(",\"midi_pitch\":").append(tuning + note.getValue() + track.getOffset())
                                .append(",\"velocity\":").append(note.getVelocity())
                                .append(",\"tied\":").append(note.isTiedNote())
                                .append(",\"effects\":{")
                                .append("\"dead\":").append(effect.isDeadNote()).append(',')
                                .append("\"vibrato\":").append(effect.isVibrato()).append(',')
                                .append("\"bend\":").append(effect.isBend()).append(',')
                                .append("\"hammer\":").append(effect.isHammer()).append(',')
                                .append("\"slide\":").append(effect.isSlide()).append(',')
                                .append("\"ghost\":").append(effect.isGhostNote()).append(',')
                                .append("\"harmonic\":").append(effect.isHarmonic()).append(',')
                                .append("\"grace\":").append(effect.isGrace()).append(',')
                                .append("\"palm_mute\":").append(effect.isPalmMute()).append(',')
                                .append("\"staccato\":").append(effect.isStaccato()).append(',')
                                .append("\"let_ring\":").append(effect.isLetRing()).append(',')
                                .append("\"tapping\":").append(effect.isTapping())
                                .append("}}");
                    }
                    json.append("]}");
                }
                json.append("]}");
            }
            json.append("]}");
        }
        json.append("\n  ]\n}\n");
        return json.toString();
    }

    private String sourceManifestJson(String id, String hash, String extension, Path original,
            Path sourceCopy, Path labelPath, TGSong song, TGTrack track, SongStats stats) {
        String split = splitForHash(hash);
        return "{" +
                "\"id\":" + quote(id) + "," +
                "\"sha256\":" + quote(hash) + "," +
                "\"split\":" + quote(split) + "," +
                "\"source_format\":" + quote(extension) + "," +
                "\"original_path\":" + quote(original.toString()) + "," +
                "\"source_gp\":" + quote(relative(sourceCopy)) + "," +
                "\"label_json\":" + quote(relative(labelPath)) + "," +
                "\"target_track_number\":" + track.getNumber() + "," +
                "\"track_count\":" + song.countTracks() + "," +
                "\"measure_count\":" + song.countMeasureHeaders() + "," +
                "\"note_count\":" + stats.noteCount + "," +
                "\"multi_digit_fret_count\":" + stats.multiDigitFretCount + "," +
                "\"effect_note_count\":" + stats.effectNoteCount +
                "}";
    }

    private void writeBuildSummary(Map<String, Integer> accepted, Map<String, Integer> attempted) throws IOException {
        StringBuilder json = new StringBuilder();
        json.append("{\n  \"schema_version\": \"1.0\",\n");
        json.append("  \"generated_at\": ").append(quote(Instant.now().toString())).append(",\n");
        json.append("  \"source_root\": ").append(quote(this.sourceRoot.toString())).append(",\n");
        json.append("  \"per_format_target\": ").append(this.perFormat).append(",\n");
        json.append("  \"accepted\": {");
        boolean first = true;
        for (String extension : EXTENSIONS) {
            if (!first) json.append(',');
            first = false;
            json.append(quote(extension)).append(':').append(accepted.get(extension));
        }
        json.append("},\n  \"attempted\": {");
        first = true;
        for (String extension : EXTENSIONS) {
            if (!first) json.append(',');
            first = false;
            json.append(quote(extension)).append(':').append(attempted.get(extension));
        }
        json.append("}\n}\n");
        Files.writeString(this.outputRoot.resolve("manifests/build_summary.json"), json.toString(), StandardCharsets.UTF_8);
    }

    private String relative(Path path) {
        return this.outputRoot.relativize(path).toString().replace('\\', '/');
    }

    private static String splitForHash(String hash) {
        int hashPrefix = Integer.parseUnsignedInt(hash.substring(0, 8), 16);
        int bucket = Integer.remainderUnsigned(hashPrefix, 10);
        if (bucket == 0) return "test";
        if (bucket == 1) return "validation";
        return "train";
    }

    private static String extension(Path path) {
        String name = path.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot >= 0 ? name.substring(dot + 1).toLowerCase(Locale.ROOT) : "";
    }

    private static String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (var input = Files.newInputStream(path)) {
            byte[] buffer = new byte[64 * 1024];
            int length;
            while ((length = input.read(buffer)) >= 0) {
                if (length > 0) digest.update(buffer, 0, length);
            }
        }
        StringBuilder hex = new StringBuilder(64);
        for (byte value : digest.digest()) {
            hex.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return hex.toString();
    }

    private static void field(StringBuilder json, String name, String value, boolean comma, int indent) {
        indent(json, indent).append(quote(name)).append(": ").append(quote(value == null ? "" : value));
        json.append(comma ? ",\n" : "\n");
    }

    private static void field(StringBuilder json, String name, int value, boolean comma, int indent) {
        indent(json, indent).append(quote(name)).append(": ").append(value);
        json.append(comma ? ",\n" : "\n");
    }

    private static StringBuilder indent(StringBuilder json, int indent) {
        return json.append("  ".repeat(Math.max(0, indent)));
    }

    private static String quote(String value) {
        if (value == null) return "null";
        StringBuilder out = new StringBuilder(value.length() + 16);
        out.append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format(Locale.ROOT, "\\u%04x", (int)c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        return out.append('"').toString();
    }

    private static void usage() {
        System.out.println("Usage: TuxGuitarDatasetBuilder <source-root> <output-root> [per-format]");
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            usage();
            System.exit(2);
        }
        Path sourceRoot = Path.of(args[0]).toAbsolutePath().normalize();
        Path outputRoot = Path.of(args[1]).toAbsolutePath().normalize();
        int perFormat = args.length >= 3 ? Integer.parseInt(args[2]) : 10;
        TuxGuitarDatasetBuilder builder = new TuxGuitarDatasetBuilder(sourceRoot, outputRoot, perFormat);
        try {
            builder.run();
        } finally {
            builder.disconnectPlugins();
        }
    }

    private record SongLoad(TGSong song, TGSongManager manager) {}

    private static final class SongStats {
        int noteCount;
        int restVoiceCount;
        int chordBeatCount;
        int effectNoteCount;
        int multiDigitFretCount;
        int maxFret;
    }
}
