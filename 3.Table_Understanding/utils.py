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
    """Split tokens into column groups based on x-gaps and left-edge clusters."""
    if len(tokens) < 2:
        return [tokens]

    # Strategy 1: gap-based splitting (original)
    sorted_tokens = sorted(tokens, key=lambda t: (t["bbox"][0] + t["bbox"][2]) / 2)
    widths = [t["bbox"][2] - t["bbox"][0] for t in sorted_tokens]
    median_w = sorted(widths)[len(widths)//2] if widths else 50
    gap_threshold = max(median_w * 2, iw * gap_ratio)
    groups = [[sorted_tokens[0]]]
    group_right = sorted_tokens[0]["bbox"][2]
    for tok in sorted_tokens[1:]:
        curr_left = tok["bbox"][0]
        if curr_left - group_right > gap_threshold:
            groups.append([tok])
            group_right = tok["bbox"][2]
        else:
            groups[-1].append(tok)
            group_right = max(group_right, tok["bbox"][2])

    return groups


def split_virtual_panels(tokens, iw, ih):
    """Split a landscape two-page spread at a real central whitespace gutter."""
    if len(tokens) < 2 or iw / max(1, ih) < 1.25:
        return [(0, tokens)]

    intervals = sorted((t["bbox"][0], t["bbox"][2]) for t in tokens)
    merged = [list(intervals[0])]
    for left, right in intervals[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])

    candidates = []
    for current, following in zip(merged, merged[1:]):
        gap = following[0] - current[1]
        midpoint = (following[0] + current[1]) / 2
        if gap >= iw * 0.04 and iw * 0.4 <= midpoint <= iw * 0.6:
            candidates.append((gap, midpoint))
    if not candidates:
        return [(0, tokens)]

    _, seam = max(candidates)
    left = [t for t in tokens if (t["bbox"][0] + t["bbox"][2]) / 2 < seam]
    right = [t for t in tokens if (t["bbox"][0] + t["bbox"][2]) / 2 >= seam]
    if min(len(left), len(right)) < 5:
        return [(0, tokens)]
    return [(0, left), (1, right)]


def merge_row_aligned_groups(groups):
    """Rejoin adjacent column groups that share a repeated row structure."""
    if len(groups) < 2:
        return groups

    merged = []
    current = groups[0]
    for candidate in groups[1:]:
        if _groups_share_rows(current, candidate):
            current = current + candidate
        else:
            merged.append(current)
            current = candidate
    merged.append(current)
    return merged


def _groups_share_rows(left, right):
    left_bbox = merge_bboxes([t["bbox"] for t in left])
    right_bbox = merge_bboxes([t["bbox"] for t in right])
    overlap = max(0, min(left_bbox[3], right_bbox[3]) - max(left_bbox[1], right_bbox[1]))
    shorter_span = max(1, min(left_bbox[3] - left_bbox[1], right_bbox[3] - right_bbox[1]))
    if overlap / shorter_span < 0.65:
        return False

    heights = [t["bbox"][3] - t["bbox"][1] for t in left + right]
    median_h = sorted(heights)[len(heights) // 2]
    tolerance = max(12, median_h * 0.75)
    left_y = sorted((t["bbox"][1] + t["bbox"][3]) / 2 for t in left)
    right_y = sorted((t["bbox"][1] + t["bbox"][3]) / 2 for t in right)

    matched = 0
    li = ri = 0
    while li < len(left_y) and ri < len(right_y):
        delta = left_y[li] - right_y[ri]
        if abs(delta) <= tolerance:
            matched += 1
            li += 1
            ri += 1
        elif delta < 0:
            li += 1
        else:
            ri += 1
    return matched >= 3


def trim_sparse_vertical_outliers(tokens):
    """Remove isolated top/bottom rows that are unlikely to belong to the table."""
    if len(tokens) < 6:
        return tokens

    heights = [t["bbox"][3] - t["bbox"][1] for t in tokens]
    median_h = sorted(heights)[len(heights) // 2]
    row_tolerance = max(10, median_h * 0.6)
    rows = []
    for token in sorted(tokens, key=lambda t: (t["bbox"][1] + t["bbox"][3]) / 2):
        center_y = (token["bbox"][1] + token["bbox"][3]) / 2
        if not rows or center_y - rows[-1]["center"] > row_tolerance:
            rows.append({"center": center_y, "tokens": [token]})
        else:
            rows[-1]["tokens"].append(token)
            rows[-1]["center"] = sum(
                (t["bbox"][1] + t["bbox"][3]) / 2 for t in rows[-1]["tokens"]
            ) / len(rows[-1]["tokens"])

    if len(rows) < 3:
        return tokens

    gaps = [rows[i + 1]["center"] - rows[i]["center"] for i in range(len(rows) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2]
    while len(rows) >= 3 and len(rows[-1]["tokens"]) == 1:
        last_gap = rows[-1]["center"] - rows[-2]["center"]
        token = rows[-1]["tokens"][0]
        full_bbox = merge_bboxes([t["bbox"] for row in rows[:-1] for t in row["tokens"]])
        token_width = token["bbox"][2] - token["bbox"][0]
        table_width = max(1, full_bbox[2] - full_bbox[0])
        sparse_edge_row = token_width / table_width < 0.35
        if last_gap <= max(median_gap * 1.15, median_h * 1.5) or not sparse_edge_row:
            break
        rows.pop()
    return [token for row in rows for token in row["tokens"]]


def split_by_vertical_gap(tokens, ih, gap_multiplier=3.0):
    if len(tokens) < 2:
        return [tokens]
    ordered = sorted(tokens, key=lambda t: (t["bbox"][1] + t["bbox"][3]) / 2)
    heights = [t["bbox"][3] - t["bbox"][1] for t in ordered]
    median_h = sorted(heights)[len(heights) // 2] if heights else 20
    max_gap = max(median_h * gap_multiplier, 20)
    rows = [[ordered[0]]]
    for tok in ordered[1:]:
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


def _to_standard_number(text):
    """Convert locale-specific number (European/Vietnamese) to float."""
    s = text.strip()
    if not s:
        return None
    s = s.replace("VND", "").replace("%", "").replace(" ", "")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.startswith("-"):
        neg, s = True, s[1:]
    if not s:
        return None
    has_comma = "," in s
    has_period = "." in s
    if has_comma and has_period:
        # Both separators: the LAST one is the decimal separator
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma:
        # Only comma: decimal if <=2 digits after, else thousands
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_period:
        # Only period: thousands if multiple periods, else keep decimal
        parts = s.split(".")
        if len(parts) > 2:
            s = s.replace(".", "")
    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return None


def is_numeric(text):
    if not text:
        return False
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
    result = _to_standard_number(v)
    if result is not None:
        return result
    for token in v.split():
        token = token.strip()
        if not token:
            continue
        result = _to_standard_number(token)
        if result is not None:
            return result
    return None
