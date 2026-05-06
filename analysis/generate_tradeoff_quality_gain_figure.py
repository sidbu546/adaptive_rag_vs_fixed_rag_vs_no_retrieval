from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/siddhanthkalyanaraman/Desktop/nlp_project")
CSV_PATH = ROOT / "analysis_outputs" / "configuration_level_deltas.csv"
OUT_PATH = ROOT / "analysis_outputs" / "tradeoff_quality_gain_vs_latency.png"


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


TITLE_FONT = load_font(30, bold=True)
AXIS_FONT = load_font(19)
TICK_FONT = load_font(16)
LABEL_FONT = load_font(16)
LEGEND_FONT = load_font(16)


COLORS = {
    "Llama 3.3 70B": "#2E6EAA",
    "Qwen 32B": "#D07428",
}


LABEL_OFFSETS = {
    "Llama\n4bit-medium": (8, -48),
    "Llama\n4bit-small": (18, -10),
    "Llama\n8bit-big": (18, 16),
    "Llama\n8bit-medium": (8, -34),
    "Llama\n8bit-small": (20, 2),
    "Llama\n4bit-big": (10, -38),
    "Qwen\n8bit-medium": (4, -38),
    "Qwen\n8bit-big": (4, -38),
    "Qwen\n8bit-small": (2, -2),
    "Qwen\n4bit-small": (2, -36),
    "Qwen\n4bit-medium": (16, -42),
    "Qwen\n4bit-big": (10, -38),
}


def compact_label(model_family: str, quantization: str, dataset_size: str) -> str:
    model = "Llama" if "Llama" in model_family else "Qwen"
    return f"{model}\n{quantization}-{dataset_size}"


def draw_centered(draw, x, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def draw_label(draw, px, py, label, dx, dy):
    tx = px + dx
    ty = py + dy
    lines = label.split("\n")
    widths = [draw.textbbox((0, 0), line, font=LABEL_FONT)[2] for line in lines]
    line_h = 18
    box_w = max(widths) + 14
    box_h = line_h * len(lines) + 10
    x0 = tx
    y0 = ty
    x1 = tx + box_w
    y1 = ty + box_h
    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill="white", outline="#8A8A8A", width=1)
    cy = y0 + 5
    for line in lines:
        draw.text((x0 + 7, cy), line, font=LABEL_FONT, fill="#2A2A2A")
        cy += line_h
    anchor_x = x0 if dx >= 0 else x1
    anchor_y = y0 + box_h / 2
    draw.line((px, py, anchor_x, anchor_y), fill="#7A7A7A", width=1)


def main():
    rows = []
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "model_family": row["model_family"],
                    "quantization": row["quantization"],
                    "dataset_size": row["dataset_size"],
                    "x": float(row["delta_response_time_s"]),
                    "y": float(row["delta_quality_index"]),
                }
            )

    width, height = 1200, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    left, top, right, bottom = 105, 95, 1080, 585
    x_min, x_max = -23.5, 10.5
    y_min, y_max = -0.08, 0.54

    def map_x(x):
        return left + (x - x_min) / (x_max - x_min) * (right - left)

    def map_y(y):
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    draw_centered(draw, width / 2, 36, "Tradeoff Plot: Quality Gain vs Response-Time Delta", TITLE_FONT, "#111111")

    for xv in [-20, -15, -10, -5, 0, 5, 10]:
        px = map_x(xv)
        draw.line((px, top, px, bottom), fill="#E6E6E6", width=1)
        draw.text((px - 10, bottom + 12), str(xv), font=TICK_FONT, fill="#555555")

    for yv in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        py = map_y(yv)
        draw.line((left, py, right, py), fill="#E6E6E6", width=1)
        label = f"{yv:.1f}"
        draw.text((left - 42, py - 8), label, font=TICK_FONT, fill="#555555")

    draw.rectangle((left, top, right, bottom), outline="#444444", width=2)
    draw.line((map_x(0), top, map_x(0), bottom), fill="#7A7A7A", width=2)
    draw.line((left, map_y(0), right, map_y(0)), fill="#7A7A7A", width=2)

    draw_centered(draw, width / 2, 632, "Response-time delta (Adaptive - Fixed)", AXIS_FONT, "#111111")

    y_label = "Quality-index gain (Adaptive - Fixed)"
    y_bbox = draw.textbbox((0, 0), y_label, font=AXIS_FONT)
    temp = Image.new("RGBA", (y_bbox[2] - y_bbox[0] + 20, y_bbox[3] - y_bbox[1] + 20), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((10, 10), y_label, font=AXIS_FONT, fill="#111111")
    rotated = temp.rotate(90, expand=True)
    img.paste(rotated, (18, int((top + bottom) / 2 - rotated.size[1] / 2)), rotated)

    legend_x, legend_y = 120, 108
    legend_w, legend_h = 330, 126
    draw.rounded_rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + legend_h), radius=8, fill="white", outline="#999999", width=1)
    legend_rows = [
        ("dot_blue", "Blue = Llama 3.3 70B"),
        ("dot_orange", "Orange = Qwen 2.5 32B"),
        ("circle", "Circle = 4-bit"),
        ("square", "Square = 8-bit"),
    ]
    row_y = legend_y + 18
    for kind, text in legend_rows:
        if kind == "dot_blue":
            draw.ellipse((legend_x + 14, row_y - 2, legend_x + 30, row_y + 14), fill=COLORS["Llama 3.3 70B"], outline="black")
        elif kind == "dot_orange":
            draw.ellipse((legend_x + 14, row_y - 2, legend_x + 30, row_y + 14), fill=COLORS["Qwen 32B"], outline="black")
        elif kind == "circle":
            draw.ellipse((legend_x + 14, row_y - 2, legend_x + 30, row_y + 14), outline="#666666", width=2)
        else:
            draw.rectangle((legend_x + 14, row_y - 2, legend_x + 30, row_y + 14), outline="#666666", width=2)
        draw.text((legend_x + 40, row_y - 6), text, font=LEGEND_FONT, fill="#222222")
        row_y += 28

    for row in rows:
        px, py = map_x(row["x"]), map_y(row["y"])
        color = COLORS[row["model_family"]]
        if row["quantization"] == "4bit":
            draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill=color, outline="black", width=1)
        else:
            draw.rectangle((px - 9, py - 9, px + 9, py + 9), fill=color, outline="black", width=1)

    for row in rows:
        px, py = map_x(row["x"]), map_y(row["y"])
        label = compact_label(row["model_family"], row["quantization"], row["dataset_size"])
        dx, dy = LABEL_OFFSETS.get(label, (10, -30))
        draw_label(draw, px, py, label, dx, dy)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH)


if __name__ == "__main__":
    main()
