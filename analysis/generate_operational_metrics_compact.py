from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path("/Users/siddhanthkalyanaraman/Desktop/nlp_project/analysis_outputs/operational_metrics_compact.png")

SYSTEMS = ["RARC", "Fixed", "No Retrieval"]
COLORS = ["#2C7FB8", "#D95F02", "#7F7F7F"]

METRICS = [
    ("LLM latency (s)", [27.6597, 10.9039, 16.5439], 33.0),
    ("GPU throughput (tok/s)", [9.8833, 10.3417, 11.4240], 13.5),
    ("GPU utilization (%)", [67.7977, 91.1469, 67.4054], 108.0),
    ("Peak GPU memory (GB)", [46.3864, 40.5305, 41.2329], 53.0),
]


def load_font(size: int, bold: bool = False):
    font_paths = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(46, bold=True)
PANEL_FONT = load_font(34, bold=True)
LABEL_FONT = load_font(26)
VALUE_FONT = load_font(28, bold=True)


def draw_centered(draw: ImageDraw.ImageDraw, xy, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    x = xy[0] - (bbox[2] - bbox[0]) / 2
    y = xy[1] - (bbox[3] - bbox[1]) / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_panel(draw, box, title, values, ymax):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=16, outline="#666666", width=2, fill="#FBFBFB")
    draw_centered(draw, ((x0 + x1) / 2, y0 + 26), title, PANEL_FONT, "#111111")

    left = x0 + 58
    right = x1 - 34
    top = y0 + 94
    bottom = y1 - 68

    draw.line((left, top, left, bottom), fill="#999999", width=2)
    draw.line((left, bottom, right, bottom), fill="#999999", width=2)

    grid_steps = 4
    for i in range(grid_steps + 1):
        y = top + (bottom - top) * i / grid_steps
        value = ymax * (grid_steps - i) / grid_steps
        draw.line((left, y, right, y), fill="#E3E3E3", width=1)
        label = f"{value:.0f}" if ymax >= 20 else f"{value:.1f}"
        draw.text((left - 52, y - 12), label, font=LABEL_FONT, fill="#666666")

    slot_width = (right - left) / len(values)
    bar_width = 84
    for idx, (label, value, color) in enumerate(zip(SYSTEMS, values, COLORS)):
        cx = left + slot_width * (idx + 0.5)
        bar_height = (value / ymax) * (bottom - top)
        bar_top = bottom - bar_height
        draw.rounded_rectangle(
            (cx - bar_width / 2, bar_top, cx + bar_width / 2, bottom),
            radius=8,
            fill=color,
            outline="#333333",
            width=1,
        )
        draw_centered(draw, (cx, bar_top - 20), f"{value:.1f}", VALUE_FONT, "#222222")
        draw_centered(draw, (cx, bottom + 30), label, LABEL_FONT, "#222222")


def main():
    width, height = 1880, 1280
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw_centered(draw, (width / 2, 58), "Operational Metrics Across Architectures", TITLE_FONT, "#111111")

    panel_w, panel_h = 800, 440
    margin_x = 110
    gap_x = 70
    start_y = 120
    gap_y = 70
    boxes = [
        (margin_x, start_y, margin_x + panel_w, start_y + panel_h),
        (margin_x + panel_w + gap_x, start_y, margin_x + 2 * panel_w + gap_x, start_y + panel_h),
        (margin_x, start_y + panel_h + gap_y, margin_x + panel_w, start_y + 2 * panel_h + gap_y),
        (margin_x + panel_w + gap_x, start_y + panel_h + gap_y, margin_x + 2 * panel_w + gap_x, start_y + 2 * panel_h + gap_y),
    ]

    for box, (title, values, ymax) in zip(boxes, METRICS):
        draw_panel(draw, box, title, values, ymax)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)


if __name__ == "__main__":
    main()
