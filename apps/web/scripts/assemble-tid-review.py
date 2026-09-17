"""Lay out Blender renders as a labeled review contact sheet."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[3]
letters = json.loads((root / 'tmp/tid/poses.json').read_text(encoding='utf-8'))
sheet = Image.new('RGB', (2400, 2640), '#111521')
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 26)
for i, letter in enumerate(letters):
    x, y = i % 5 * 480, i // 5 * 440
    sheet.paste(Image.open(root / f'tmp/tid/renders/{i+1:02d}.png').convert('RGB'), (x, y))
    draw.text((x+20, y+405), letter['ch'], font=font, fill='#d5e5f3')
sheet.save(root / 'output/models/tid-pose-review.png')
