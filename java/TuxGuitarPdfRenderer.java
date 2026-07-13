import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

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
import app.tuxguitar.song.models.TGDuration;
import app.tuxguitar.song.models.TGMeasure;
import app.tuxguitar.song.models.TGSong;
import app.tuxguitar.song.models.TGTrack;
import app.tuxguitar.util.TGContext;
import app.tuxguitar.util.plugin.TGPlugin;

/** Headless score+TAB PDF renderer using the local TuxGuitar installation. */
public final class TuxGuitarPdfRenderer {
    private final TGContext context = new TGContext();
    private final List<TGPlugin> plugins = new ArrayList<>();

    private TuxGuitarPdfRenderer() throws Exception {
        this.plugins.add(new GP1InputStreamPlugin());
        this.plugins.add(new GP2InputStreamPlugin());
        this.plugins.add(new GP3InputStreamPlugin());
        this.plugins.add(new GP4InputStreamPlugin());
        this.plugins.add(new GP5InputStreamPlugin());
        this.plugins.add(new app.tuxguitar.io.gpx.v6.GPXInputStreamPlugin());
        this.plugins.add(new app.tuxguitar.io.gpx.v7.GPXInputStreamPlugin());
        this.plugins.add(new PDFSongWriterPlugin());
        for (TGPlugin plugin : this.plugins) plugin.connect(this.context);
    }

    private void close() {
        Collections.reverse(this.plugins);
        for (TGPlugin plugin : this.plugins) {
            try {
                plugin.disconnect(this.context);
            } catch (Throwable ignored) {
                // Best effort during process shutdown.
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: TuxGuitarPdfRenderer INPUT.gp OUTPUT.pdf");
        }
        Path input = Path.of(args[0]).toAbsolutePath();
        Path output = Path.of(args[1]).toAbsolutePath();
        if (!Files.isRegularFile(input)) throw new IllegalArgumentException("Missing input: " + input);
        Files.createDirectories(output.getParent());
        TuxGuitarPdfRenderer renderer = new TuxGuitarPdfRenderer();
        try {
            TGSongManager manager = new TGSongManager();
            TGSong song = renderer.read(input, manager);
            TGTrack track = chooseTrack(song);
            if (track == null) throw new IllegalStateException("No 4-8 string non-percussion track found");
            renderer.write(song, manager, track, output);
            System.out.println("OUTPUT_PDF=" + output);
            System.out.println("SONG=" + song.getName());
            System.out.println("TRACK_NUMBER=" + track.getNumber());
            System.out.println("TRACK_NAME=" + track.getName());
            System.out.println("STRINGS=" + track.stringCount());
            System.out.println("MEASURES=" + song.countMeasureHeaders());
            System.out.println("BYTES=" + Files.size(output));
        } finally {
            renderer.close();
        }
    }

    private TGSong read(Path input, TGSongManager manager) throws Exception {
        TGSongReaderHandle handle = new TGSongReaderHandle();
        handle.setFactory(manager.getFactory());
        handle.setInputStream(new FileInputStream(input.toFile()));
        handle.setContext(new TGSongStreamContext());
        handle.getContext().setAttribute(
                TGSongReaderHelper.ATTRIBUTE_FORMAT_CODE,
                TGFileFormatUtils.getFileFormatCode(input.toString()));
        TGFileFormatManager.getInstance(this.context).read(handle);
        TGSong song = handle.getSong();
        if (song == null || song.isEmpty()) throw new IllegalStateException("TuxGuitar read an empty song");
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

    private static TGTrack chooseTrack(TGSong song) {
        TGTrack fallback = null;
        Iterator<TGTrack> tracks = song.getTracks();
        while (tracks.hasNext()) {
            TGTrack track = tracks.next();
            if (!track.isPercussion() && track.stringCount() >= 4 && track.stringCount() <= 8) {
                if (track.stringCount() == 6) return track;
                if (fallback == null) fallback = track;
            }
        }
        return fallback;
    }

    private void write(TGSong song, TGSongManager manager, TGTrack track, Path output) throws Exception {
        TGPrintSettings settings = new TGPrintSettings();
        settings.setStyle(TGLayout.DISPLAY_COMPACT | TGLayout.DISPLAY_MODE_BLACK_WHITE
                | TGLayout.DISPLAY_SCORE | TGLayout.DISPLAY_TABLATURE);
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
        try (BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(output.toFile()))) {
            writer.setOutputStream(stream);
            TGFileFormatManager.getInstance(this.context).write(writer);
        }
        if (!Files.isRegularFile(output) || Files.size(output) == 0) {
            throw new IllegalStateException("TuxGuitar produced an empty PDF");
        }
    }
}
