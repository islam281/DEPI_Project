"""Create or safely reopen the contextual-child Chroma collection.

An on-disk manifest prevents accidentally reusing a stale index after
changing the source book, chunk sizes, or embedding configuration.
"""
import hashlib
import json
from pathlib import Path

def _fingerprint(children, config):
    h = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf8"))
    for child in children:
        h.update(child.metadata["child_id"].encode("utf8"))
        h.update(child.page_content.encode("utf8"))
    return h.hexdigest()

def open_or_create_child_chroma(children, embedding_model, *, index_dir, collection_name,
                                config, batch_size=32):
    """Reuse an index only when the stored content and configuration match."""
    from langchain_chroma import Chroma
    folder = Path(index_dir)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "nateq_phase6_manifest.json"
    expected = {"fingerprint": _fingerprint(children, config),
                "count": len(children), "config": config, "collection": collection_name}
    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(folder),
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    
    
    current_ids = store.get(limit=1)["ids"]
    if current_ids:
        if not manifest_path.exists():
            raise RuntimeError(f"Found existing Chroma without experiment manifest: {folder}. "
                               "Use a NEW index_dir; never blindly reuse it.")
        existing = json.loads(manifest_path.read_text(encoding="utf8"))
        if existing != expected:
            raise RuntimeError(f"Index parameters/content changed: {folder}. "
                               "Use a new index_dir for the changed experiment.")
        print(f"Reusing validated child index: {folder} ({len(children)} children)")
        return store
    if manifest_path.exists():
        raise RuntimeError(f"Manifest exists but collection is empty: {folder}. "
                           "Use a clean NEW index_dir to avoid stale state.")
    print(f"Indexing {len(children)} children in {collection_name} ...")
    
    for start in range(0, len(children), batch_size):
        batch = children[start:start+batch_size]
        store.add_documents(batch, ids=[doc.metadata["child_id"] for doc in batch])
        if (start // batch_size) % 10 == 0:
            print(f"  Indexed {min(start+batch_size,len(children))}/{len(children)}")
    manifest_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf8")
    return store
