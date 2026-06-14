import io
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CANVAS_WIDTH = 960
PADDING = 48
LINE_GAP = 8
PARAGRAPH_GAP = 18

BACKGROUND = '#f7f4ed'
TEXT = '#2b2b2b'
MUTED = '#646464'
ACCENT = '#6b5bd6'
RULE = '#ded8cc'

_FONT_CANDIDATES = [
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/msyh.ttf',
    'C:/Windows/Fonts/msyhbd.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/simsun.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
]

_GLYPH_MASK_CACHE: Dict[Tuple[int, str], Optional[Tuple[Tuple[int, int], bytes]]] = {}
_MISSING_GLYPH_CACHE: Dict[int, Optional[Tuple[Tuple[int, int], bytes]]] = {}


def render_markdown_to_png(markdown: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    fonts = {
        'h1': _load_font(ImageFont, 34),
        'h2': _load_font(ImageFont, 28),
        'h3': _load_font(ImageFont, 24),
        'body': _load_font(ImageFont, 22),
        'small': _load_font(ImageFont, 18),
    }
    scratch = Image.new('RGB', (CANVAS_WIDTH, 100), BACKGROUND)
    draw = ImageDraw.Draw(scratch)

    layout = list(_layout_markdown(draw, fonts, markdown))
    height = max(PADDING * 2 + 20, int(layout[-1][1] + layout[-1][4] + PADDING) if layout else 180)
    image = Image.new('RGB', (CANVAS_WIDTH, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    for kind, y, x, text, line_height, color in layout:
        if kind == 'rule':
            draw.line((PADDING, y + 8, CANVAS_WIDTH - PADDING, y + 8), fill=RULE, width=2)
            continue
        font = _font_for_line(fonts, kind)
        draw.text((x, y), _safe_text(text, draw, font), fill=color, font=font)

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def _layout_markdown(draw, fonts, markdown: str) -> Iterable[Tuple[str, int, int, str, int, str]]:
    y = PADDING
    in_code = False
    for raw_line in (markdown or '').splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith('```'):
            in_code = not in_code
            continue
        if not line.strip():
            y += PARAGRAPH_GAP
            continue

        kind, text, indent, color = _parse_line(line, in_code)
        if kind == 'rule':
            yield kind, y, PADDING, '', 18, color
            y += PARAGRAPH_GAP
            continue

        font = _font_for_line(fonts, kind)
        max_width = CANVAS_WIDTH - PADDING * 2 - indent
        wrapped = _wrap_text(draw, _strip_inline_markdown(text), font, max_width)
        line_height = _line_height(draw, font)
        x = PADDING + indent
        for wrapped_line in wrapped:
            yield kind, y, x, wrapped_line, line_height, color
            y += line_height + LINE_GAP
        y += _block_gap(kind)


def _parse_line(line: str, in_code: bool) -> Tuple[str, str, int, str]:
    if in_code:
        return 'small', line, 20, MUTED
    stripped = line.strip()
    if re.fullmatch(r'-{3,}|\*{3,}|_{3,}', stripped):
        return 'rule', '', 0, RULE
    if stripped.startswith('# '):
        return 'h1', stripped[2:].strip(), 0, ACCENT
    if stripped.startswith('## '):
        return 'h2', stripped[3:].strip(), 0, TEXT
    if stripped.startswith('### '):
        return 'h3', stripped[4:].strip(), 0, TEXT
    if stripped.startswith('>'):
        return 'body', stripped.lstrip('> ').strip(), 20, MUTED

    bullet = re.match(r'^[-*+]\s+(.+)$', stripped)
    if bullet:
        return 'body', f'• {bullet.group(1)}', 24, TEXT

    ordered = re.match(r'^(\d+)[.)]\s+(.+)$', stripped)
    if ordered:
        return 'body', f'{ordered.group(1)}. {ordered.group(2)}', 24, TEXT

    return 'body', stripped, 0, TEXT


def _strip_inline_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return text


def _wrap_text(draw, text: str, font, max_width: int) -> List[str]:
    lines = []
    current = ''
    for char in text:
        candidate = current + char
        if current and _text_width(draw, candidate, font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or ['']


def _text_width(draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), _safe_text(text, draw, font), font=font)
    return bbox[2] - bbox[0]


def _line_height(draw, font) -> int:
    bbox = draw.textbbox((0, 0), _safe_text('国Ag', draw, font), font=font)
    return bbox[3] - bbox[1] + 4


def _block_gap(kind: str) -> int:
    if kind == 'h1':
        return 18
    if kind in ('h2', 'h3'):
        return 12
    return 8


def _font_for_line(fonts, kind: str):
    if kind in fonts:
        return fonts[kind]
    return fonts['body']


def _load_font(image_font, size: int):
    for candidate in _font_candidates():
        if Path(candidate).exists():
            font = image_font.truetype(candidate, size=size)
            if _font_supports_cjk(font):
                return font
    raise RuntimeError('No CJK-capable font found for summary image rendering')


def _font_candidates() -> Iterable[str]:
    configured = os.getenv('PALLAS_SUMMARY_FONT')
    if configured:
        yield configured
    yield from _FONT_CANDIDATES


def _font_supports_cjk(font) -> bool:
    left = _glyph_mask(font, '测')
    right = _glyph_mask(font, '聊')
    return left is not None and right is not None and left != right


def _glyph_mask(font, char: str) -> Optional[Tuple[Tuple[int, int], bytes]]:
    key = (id(font), char)
    if key in _GLYPH_MASK_CACHE:
        return _GLYPH_MASK_CACHE[key]

    try:
        mask = font.getmask(char)
        data = mask.tobytes() if hasattr(mask, 'tobytes') else bytes(mask)
        result = (mask.size, data)
    except Exception:
        result = None

    _GLYPH_MASK_CACHE[key] = result
    return result


def _missing_glyph_mask(font) -> Optional[Tuple[Tuple[int, int], bytes]]:
    key = id(font)
    if key not in _MISSING_GLYPH_CACHE:
        _MISSING_GLYPH_CACHE[key] = _glyph_mask(font, '\U0010ffff')
    return _MISSING_GLYPH_CACHE[key]


def _safe_text(text: str, draw, font) -> str:
    text = _drop_missing_glyphs(text, font)
    try:
        draw.textbbox((0, 0), text, font=font)
    except UnicodeEncodeError:
        return text.encode('latin-1', 'replace').decode('latin-1')
    return text


def _drop_missing_glyphs(text: str, font) -> str:
    missing = _missing_glyph_mask(font)
    if missing is None:
        return text

    chars = []
    for char in text:
        if char.isspace() or _glyph_mask(font, char) != missing:
            chars.append(char)
    return ''.join(chars)
