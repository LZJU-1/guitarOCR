import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

import app.tuxguitar.awt.graphics.AWTPainter;
import app.tuxguitar.graphics.control.painters.TGClefPainter;
import app.tuxguitar.graphics.control.painters.TGKeySignaturePainter;
import app.tuxguitar.graphics.control.painters.TGNotePainter;
import app.tuxguitar.graphics.control.painters.TGNumberPainter;
import app.tuxguitar.graphics.control.painters.TGSilencePainter;
import app.tuxguitar.ui.resource.UIPainter;

/**
 * Renders atomic notation templates directly from TuxGuitar 2.0.1 painters.
 *
 * A filename has the form <semantic-class>__<visual-variant>.png.  Several
 * painter/font variants may map to the same semantic class in the Python
 * dataset builder.
 */
public final class TuxGuitarAtomicSymbolBuilder {
    private static final int CANVAS = 320;
    private static final int CROP_PADDING = 12;

    @FunctionalInterface
    private interface DrawAction {
        void draw(AWTPainter painter);
    }

    private final Path output;

    private TuxGuitarAtomicSymbolBuilder(Path output) {
        this.output = output;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: TuxGuitarAtomicSymbolBuilder <template-output-directory>");
            System.exit(2);
        }
        Path output = Path.of(args[0]).toAbsolutePath().normalize();
        Files.createDirectories(output);
        TuxGuitarAtomicSymbolBuilder builder = new TuxGuitarAtomicSymbolBuilder(output);
        builder.build();
        System.out.println("Atomic TuxGuitar templates written to " + output);
    }

    private void build() throws IOException {
        for (int digit = 0; digit <= 9; digit++) {
            final int value = digit;
            render("digit_" + digit + "__time", painter -> {
                painter.initPath(UIPainter.PATH_FILL);
                TGNumberPainter.paint(value, painter, 90f, 90f, 58f);
                painter.closePath();
            });
            render("digit_" + digit + "__tab", painter -> {
                painter.setFont(painter.createFont("Times New Roman", 64f, true, false));
                painter.drawString(Integer.toString(value), 95f, 85f);
            });
        }

        render("dead_x__tab", painter -> {
            painter.setFont(painter.createFont("Times New Roman", 64f, true, false));
            painter.drawString("X", 95f, 85f);
        });
        render("dead_x__score", painter -> {
            painter.setLineWidth(5f);
            painter.initPath(UIPainter.PATH_DRAW);
            TGNotePainter.paintXNote(painter, 100f, 100f, 70f);
            painter.closePath();
        });

        render("notehead_filled__score", painter -> {
            painter.initPath(UIPainter.PATH_FILL);
            TGNotePainter.paintNote(painter, 100f, 100f, 70f);
            painter.closePath();
        });
        render("notehead_open__score", painter -> {
            painter.setLineWidth(5f);
            painter.initPath(UIPainter.PATH_DRAW);
            TGNotePainter.paintNote(painter, 100f, 100f, 70f);
            painter.closePath();
        });
        render("notehead_harmonic__score", painter -> {
            painter.setLineWidth(5f);
            painter.initPath(UIPainter.PATH_DRAW);
            TGNotePainter.paintHarmonic(painter, 100f, 100f, 70f);
            painter.closePath();
        });

        render("rest_block__whole_or_half", painter -> {
            painter.initPath(UIPainter.PATH_FILL);
            TGSilencePainter.paintWhole(painter, 100f, 100f, 9f);
            painter.closePath();
        });
        render("rest_quarter__score", painter -> paintFilled(painter,
                () -> TGSilencePainter.paintQuarter(painter, 100f, 75f, 6f)));
        render("rest_eighth__score", painter -> paintFilled(painter,
                () -> TGSilencePainter.paintEighth(painter, 100f, 85f, 8f)));
        render("rest_sixteenth__score", painter -> paintFilled(painter,
                () -> TGSilencePainter.paintSixteenth(painter, 100f, 75f, 6f)));
        render("rest_thirty_second__score", painter -> paintFilled(painter,
                () -> TGSilencePainter.paintThirtySecond(painter, 100f, 70f, 5f)));
        render("rest_sixty_fourth__score", painter -> paintFilled(painter,
                () -> TGSilencePainter.paintSixtyFourth(painter, 100f, 60f, 4f)));

        render("accidental_sharp__score", painter -> paintFilled(painter,
                () -> TGKeySignaturePainter.paintSharp(painter, 100f, 80f, 30f)));
        render("accidental_flat__score", painter -> paintFilled(painter,
                () -> TGKeySignaturePainter.paintFlat(painter, 100f, 70f, 30f)));
        render("accidental_natural__score", painter -> paintFilled(painter,
                () -> TGKeySignaturePainter.paintNatural(painter, 100f, 70f, 30f)));

        render("clef_treble__score", painter -> paintFilled(painter,
                () -> TGClefPainter.paintTreble(painter, 100f, 60f, 35f)));
        render("clef_bass__score", painter -> paintFilled(painter,
                () -> TGClefPainter.paintBass(painter, 100f, 80f, 48f)));
        render("clef_c__alto_position", painter -> paintFilled(painter,
                () -> TGClefPainter.paintAlto(painter, 100f, 70f, 38f)));
        render("clef_c__tenor_position", painter -> paintFilled(painter,
                () -> TGClefPainter.paintTenor(painter, 100f, 70f, 38f)));
        render("clef_neutral__score", painter -> paintFilled(painter,
                () -> TGClefPainter.paintNeutral(painter, 100f, 70f, 40f)));

        render("dot__generic", painter -> {
            painter.initPath(UIPainter.PATH_FILL);
            painter.addCircle(140f, 140f, 34f);
            painter.closePath();
        });
    }

    private static void paintFilled(AWTPainter painter, Runnable paint) {
        painter.initPath(UIPainter.PATH_FILL);
        paint.run();
        painter.closePath();
    }

    private void render(String name, DrawAction action) throws IOException {
        BufferedImage image = new BufferedImage(CANVAS, CANVAS, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, CANVAS, CANVAS);
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.dispose();

        AWTPainter painter = new AWTPainter(image);
        painter.setForeground(painter.createColor(0, 0, 0));
        painter.setBackground(painter.createColor(0, 0, 0));
        painter.setAntialias(true);
        action.draw(painter);
        painter.dispose();

        BufferedImage cropped = cropInk(image, CROP_PADDING);
        ImageIO.write(cropped, "png", this.output.resolve(name + ".png").toFile());
    }

    private static BufferedImage cropInk(BufferedImage image, int padding) {
        int minX = image.getWidth();
        int minY = image.getHeight();
        int maxX = -1;
        int maxY = -1;
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >> 16) & 0xff;
                int green = (rgb >> 8) & 0xff;
                int blue = rgb & 0xff;
                if (red < 248 || green < 248 || blue < 248) {
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        if (maxX < minX || maxY < minY) {
            throw new IllegalStateException("Painter produced an empty template");
        }
        minX = Math.max(0, minX - padding);
        minY = Math.max(0, minY - padding);
        maxX = Math.min(image.getWidth() - 1, maxX + padding);
        maxY = Math.min(image.getHeight() - 1, maxY + padding);
        return image.getSubimage(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }
}
