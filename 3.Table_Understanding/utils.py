def merge_bboxes(bboxes):
    if not bboxes:
        return None
    xs = [b[0] for b in bboxes] + [b[2] for b in bboxes]
    ys = [b[1] for b in bboxes] + [b[3] for b in bboxes]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def expand_bbox(bbox, mx, my, iw, ih):
    return [
        max(0, bbox[0] - mx),
        max(0, bbox[1] - my),
        min(iw, bbox[2] + mx),
        min(ih, bbox[3] + my),
    ]


def split_by_columns(tokens, iw, gap_ratio=0.10):
    """Split tokens into column groups based on actual x-gaps between tokens."""
    if len(tokens) < 2:
        return [tokens]
    sorted_tokens = sorted(tokens, key=lambda t: (t["bbox"][0] + t["bbox"][2]) / 2)
    widths = [t["bbox"][2] - t["bbox"][0] for t in sorted_tokens]
    median_w = sorted(widths)[len(widths)//2] if widths else 50
    gap_threshold = max(median_w * 2, iw * 0.03)
    groups = [[sorted_tokens[0]]]
    for tok in sorted_tokens[1:]:
        prev_right = groups[-1][-1]["bbox"][2]
        curr_left = tok["bbox"][0]
        if curr_left - prev_right > gap_threshold:
            groups.append([tok])
        else:
            groups[-1].append(tok)
    return groups


def split_by_vertical_gap(tokens, ih, gap_multiplier=3.0):
    if len(tokens) < 2:
        return [tokens]
    tokens.sort(key=lambda t: (t["bbox"][1] + t["bbox"][3]) / 2)
    heights = [t["bbox"][3] - t["bbox"][1] for t in tokens]
    median_h = sorted(heights)[len(heights) // 2] if heights else 20
    max_gap = max(median_h * gap_multiplier, 20)
    rows = [[tokens[0]]]
    for tok in tokens[1:]:
        prev_y = (rows[-1][-1]["bbox"][1] + rows[-1][-1]["bbox"][3]) / 2
        curr_y = (tok["bbox"][1] + tok["bbox"][3]) / 2
        if curr_y - prev_y > max_gap:
            rows.append([tok])
        else:
            rows[-1].append(tok)
    return rows


def is_paragraph_cell(text):
    if not text:
        return False
    words = text.split()
    if len(words) < 15:
        return False
    num_count = sum(1 for w in words if any(c.isdigit() for c in w))
    comma_count = text.count(",") + text.count(".") + text.count(";")
    return num_count <= 2 and comma_count >= 2


def _clean_num_str(text):
    return text.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace("-", "").replace("VND", "").replace("%", "").replace(" ", "")


def is_numeric(text):
    if not text:
        return False
    # Split on spaces: if ANY token is numeric, consider the cell numeric
    tokens = text.split()
    for tok in tokens:
        if tok.count(".") > 1:
            continue
        cleaned = _clean_num_str(tok)
        if cleaned.isdigit() and len(cleaned) > 1:
            return True
    return False


def is_numeric_lenient(text):
    if not text:
        return False
    tokens = text.split()
    for tok in tokens:
        cleaned = _clean_num_str(tok)
        if cleaned and cleaned.isdigit():
            return True
    return False


def parse_number(text):
    v = text.strip()
    if not v:
        return None
    nv = v
    neg = False
    if nv.startswith("(") and nv.endswith(")"):
        neg, nv = True, nv[1:-1]
    if nv.startswith("-"):
        neg, nv = True, nv[1:]
    nv = nv.replace(",", "")
    try:
        val = float(nv)
        return -val if neg else val
    except:
        pass
    # Try each token in multi-value cells
    for token in v.split():
        token = token.strip()
        if not token:
            continue
        tn = token
        tneg = False
        if tn.startswith("(") and tn.endswith(")"):
            tneg, tn = True, tn[1:-1]
        if tn.startswith("-"):
            tneg, tn = True, tn[1:]
        tn = tn.replace(",", "").replace("%", "").replace("VND", "")
        try:
            val = float(tn)
            return -val if tneg else val
        except:
            continue
    return None
