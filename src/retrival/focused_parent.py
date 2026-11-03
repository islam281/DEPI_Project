"""Expand ranked children to parents and select smaller verbatim windows.

Only the selected context is shortened; Chroma and the source documents
remain unchanged. Exact child text is always retained in its parent window.
"""
from src.ingestion.chunking import _page

def _to_parents(ranked_children, parents, max_parents=5, max_context_chars=10500):
    selected, seen, chars = [], set(), 0
    for doc in ranked_children:
        
        pid = doc.metadata.get("parent_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        parent = parents[pid]
        length = len(parent.page_content)
        
        if chars + length > max_context_chars:
            continue
        selected.append(parent)
        chars += length
        if len(selected) >= max_parents:
            break
    return selected

def _strip_contextual_header(child):
    text = child.page_content
    if child.metadata.get("contextual_header", False):
        if "\n\n" not in text or not text.startswith("المصدر:"):
            raise ValueError(f"Unexpected contextual header in {child.metadata.get('child_id')}")
        text = text.split("\n\n", 1)[1]
    return text.strip()

def _child_position(parent, child):
    """Locate the *original* child passage in the parent; no fuzzy hallucination.

    RecursiveCharacterTextSplitter stores literal substrings (stripped ends).
    Refuse a missing anchor rather than creating a misleading focused passage.
    """
    body = _strip_contextual_header(child)
    if not body:
        raise ValueError("Empty child text")
    pos = parent.page_content.find(body)
    if pos < 0:
        raise ValueError(
            f"Child {child.metadata.get('child_id')} not an exact substring "
            f"of parent {parent.metadata.get('parent_id')}. "
            "Check source data and chunking before running E2E."
        )
    return pos, pos + len(body)

def _allocate_budget(parents, max_chars, min_chars):
    """Allocate to ranked parents, reassigning unused quota to longer parents.

    Small preference for higher ranked parents, but ALL parents retain a window.
    """
    n = len(parents)
    if not n:
        return []
    lengths = [len(p.page_content) for p in parents]
    if max_chars <= 0 or min_chars <= 0:
        raise ValueError("Budgets must be positive")
    if max_chars < sum(min(min_chars, l) for l in lengths):
        raise ValueError("Focus budget too small for min_chars per selected parent")
    budget = min(max_chars, sum(lengths))
    out = [min(min_chars, length) for length in lengths]
    remaining = budget - sum(out)
    # Redistribute one character at a time across at most ~5 x 2000 parents.
    weights = [n - (i * 0.15) for i in range(n)]
    while remaining > 0:
        candidates = [i for i in range(n) if out[i] < lengths[i]]
        if not candidates:
            break
        i = max(candidates, key=lambda j: (weights[j] / (out[j] + 1), -j))
        out[i] += 1
        remaining -= 1
    assert sum(out) <= max_chars
    return out

def _focused_window(parent, child, allocation, before_ratio=0.60):
    """Return a source-verbatim excerpt, always including the full best child."""
    if not 0 <= before_ratio <= 1:
        raise ValueError("before_ratio must be between 0 and 1")
    original = parent.page_content
    left, right = _child_position(parent, child)
    n = len(original)
    if allocation < (right - left):
        raise ValueError(
            f"Allocation {allocation} chars is smaller than retrieved child "
            f"{right-left} chars; increase focus budget/min_chars."
        )
    size = min(n, allocation)
    extra = size - (right - left)
    start = left - round(extra * before_ratio)
    start = max(0, min(start, n - size))
    end = start + size
    assert start <= left < right <= end
    text = original[start:end]
    assert original[left:right] in text
    return text, start, end

def select_focused_parents(ranked_children, selected_parents, *, focus_budget_chars=4500,
                           min_chars=450, before_ratio=0.60):
    """Exactly the same parents (same order/metadata) as full Parent method.

    Parent content is shortened only from its own source-verbatim text.
    User's documents, Chroma index and cached original docs are unchanged.
    """
    import copy

    if not selected_parents:
        return [], []
    by_parent = {}
    
    for child in ranked_children:
        pid = child.metadata.get("parent_id")
        if pid and pid not in by_parent:
            by_parent[pid] = child
    alloc = _allocate_budget(selected_parents, focus_budget_chars, min_chars)
    focused = []
    diagnostics = []
    for parent, allotment in zip(selected_parents, alloc):
        pid = parent.metadata["parent_id"]
        child = by_parent.get(pid)
        if child is None:
            raise ValueError(f"No child found for parent {pid}")
        excerpt, start, end = _focused_window(parent, child, allotment, before_ratio)
        clone = copy.copy(parent)
        clone.page_content = excerpt
        clone.metadata = {**parent.metadata, "focused": True,
                          "focus_start_char": start, "focus_end_char": end,
                          "focus_child_id": child.metadata.get("child_id")}
        focused.append(clone)
        diagnostics.append({
            "parent_id": pid, "page_number": _page(parent),
            "selected_child_id": child.metadata.get("child_id"),
            "parent_chars": len(parent.page_content), "focused_chars": len(excerpt),
            "start_char": start, "end_char": end,
            "child_text_present": _strip_contextual_header(child) in excerpt,
        })
    if [p.metadata["parent_id"] for p in focused] != [p.metadata["parent_id"] for p in selected_parents]:
        raise AssertionError("Focused context changed parent set or order")
    if sum(len(p.page_content) for p in focused) > focus_budget_chars:
        raise AssertionError("Focused context exceeded character budget")
    return focused, diagnostics
