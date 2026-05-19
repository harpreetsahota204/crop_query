"""CropQuery Plugin — embedding-based template matching for FiftyOne datasets.

Runs a sliding-window patch embedding approach using any FiftyOne zoo model:
- Extracts overlapping patches from each sample image
- Embeds patches and template images with the selected zoo model
- Computes cosine similarity between each patch and every template
- Produces a per-sample sparse heatmap (patches below the threshold are zero)
  and a per-sample score (raw max cosine similarity, for filtering/tagging)
"""

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

import os
import base64
import sys
import types as _types

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}

# Uploaded crops are saved here (outside the plugin dir to avoid cache invalidation).
UPLOAD_DIR = os.path.join(os.path.expanduser("~"), ".fiftyone", "cropquery_uploads")


# ---------------------------------------------------------------------------
# Model singleton (survives FiftyOne plugin reimports)
# See plugin-development/PERSISTENT-STATE-AND-IPC.md
# ---------------------------------------------------------------------------

_PERSIST_KEY = "cropquery__persist"
if _PERSIST_KEY not in sys.modules:
    _persist = _types.ModuleType(_PERSIST_KEY)
    _persist.model = None
    _persist.model_name = None
    _persist.cancel_requested = False
    sys.modules[_PERSIST_KEY] = _persist
else:
    _persist = sys.modules[_PERSIST_KEY]
    if not hasattr(_persist, "cancel_requested"):
        _persist.cancel_requested = False


def _get_model(model_name: str):
    """Load zoo model into process singleton; reuses if already loaded."""
    import fiftyone.zoo as foz

    if _persist.model is None or _persist.model_name != model_name:
        print(f"[CropQuery] Loading zoo model: {model_name}")
        _persist.model = foz.load_zoo_model(model_name)
        if not getattr(_persist.model, "has_embeddings", False):
            _persist.model = None
            _persist.model_name = None
            raise ValueError(
                f"Model '{model_name}' does not support embeddings. "
                "Choose a model that implements EmbeddingsMixin (e.g. clip-vit-base32-torch)."
            )
        _persist.model_name = model_name
        print(f"[CropQuery] Model ready: {model_name}")
    else:
        print(f"[CropQuery] Reusing cached model: {model_name}")

    return _persist.model


# ---------------------------------------------------------------------------
# Core embedding match function
# ---------------------------------------------------------------------------

def _embed_match_one(
    filepath, template_embeddings, model,
    n_cols, n_rows, overlap_pct, heatmap_threshold
):
    """Run embedding-based template matching on a single image.

    Divides the image into a grid of n_cols × n_rows patches (with optional
    overlap), embeds all patches, computes cosine similarity against each
    template embedding, and returns a sparse heatmap + score.

    Args:
        filepath: path to the sample image
        template_embeddings: list of L2-normalised (D,) float32 numpy vectors
        model: loaded FiftyOne zoo model with has_embeddings=True
        n_cols: number of patch columns that define patch width
        n_rows: number of patch rows that define patch height
        overlap_pct: percentage [0, 95] of overlap between adjacent patches
        heatmap_threshold: raw cosine similarity floor — patches below this
            value are set to 0 in the heatmap so only genuine matches show

    Returns:
        dict with:
          - ``heatmap``: sparse float32 (H, W) in [0, 1]; only regions whose
            cosine similarity exceeds heatmap_threshold are non-zero
          - ``score``: float, raw max cosine similarity across all patches
            (absolute, comparable across images — used for tagging/sorting)
        or None if the image cannot be read.
    """
    import cv2
    import numpy as np

    image_bgr = cv2.imread(filepath)
    if image_bgr is None:
        return None

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ---- derive patch dimensions from grid spec ---------------------------
    patch_w = max(1, W // n_cols)
    patch_h = max(1, H // n_rows)
    overlap_frac = max(0.0, min(0.95, overlap_pct / 100.0))
    stride_x = max(1, int(patch_w * (1.0 - overlap_frac)))
    stride_y = max(1, int(patch_h * (1.0 - overlap_frac)))

    # ---- sliding window ---------------------------------------------------
    # Patches are appended in row-major order so the flat index maps to the
    # (row, col) grid position directly via reshape — no positions list needed.
    ys = list(range(0, max(H - patch_h + 1, 1), stride_y))
    xs = list(range(0, max(W - patch_w + 1, 1), stride_x))
    grid_rows, grid_cols = len(ys), len(xs)

    patches = []
    for y in ys:
        for x in xs:
            y_end = min(y + patch_h, H)
            x_end = min(x + patch_w, W)
            patch = image_rgb[y:y_end, x:x_end]
            if patch.shape[0] < patch_h or patch.shape[1] < patch_w:
                patch = cv2.resize(patch, (patch_w, patch_h))
            patches.append(patch)

    # ---- batch embed patches ----------------------------------------------
    patch_embs = np.array(model.embed_all(patches), dtype=np.float32)  # (N, D)
    norms = np.linalg.norm(patch_embs, axis=1, keepdims=True)
    patch_embs /= np.maximum(norms, 1e-8)

    # ---- cosine similarity grid (max over templates) ----------------------
    # sims (N,) reshapes to (grid_rows, grid_cols) because patches are row-major.
    # np.maximum replaces the O(N) Python loop with a single vectorised op.
    sim_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for tmpl_emb in template_embeddings:
        sims = (patch_embs @ tmpl_emb).reshape(grid_rows, grid_cols)
        np.maximum(sim_grid, sims, out=sim_grid)

    # Raw score: peak cosine similarity — absolute, comparable across images
    raw_score = float(sim_grid.max())

    # ---- sparse heatmap: only show regions above heatmap_threshold --------
    # Patches below the threshold are 0 (no overlay). Patches above are
    # normalised within [heatmap_threshold, peak] so the best region is 1.0.
    if raw_score > heatmap_threshold:
        span = raw_score - heatmap_threshold
        display_grid = np.where(
            sim_grid >= heatmap_threshold,
            (sim_grid - heatmap_threshold) / span,
            0.0,
        ).astype(np.float32)
    else:
        display_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    # Upscale to quarter-image resolution with bilinear interpolation so the
    # heatmap looks smooth without storing a full-resolution buffer (which
    # triggers the detached-ArrayBuffer bug in FiftyOne's looker worker).
    out_w = max(grid_cols * 4, min(W // 4, 512))
    out_h = max(grid_rows * 4, min(H // 4, 512))
    heatmap = cv2.resize(display_grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return {"heatmap": heatmap, "score": raw_score}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class CropQueryPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="crop_query_panel",
            label="CropQuery",
            icon="image_search",
            surfaces="grid",
        )

    def on_load(self, ctx):
        print("[CropQuery] Panel loaded")
        print(
            f"[CropQuery] Dataset: "
            f"{ctx.dataset.name if ctx.dataset else 'None'}"
        )
        print(
            f"[CropQuery] Model in singleton: "
            f"{_persist.model_name or 'none'}"
        )

    def render(self, ctx):
        panel = types.Object()
        panel.btn(
            "cancel_run_btn",
            label="Cancel Run",
            icon="close",
            on_click=self.cancel_run,
            variant="outlined",
        )
        return types.Property(
            panel,
            view=types.View(
                component="CropQueryPanel",
                composite_view=True,
                list_templates=self.list_templates,
                browse_directory=self.browse_directory,
                load_model=self.load_model,
                upload_templates=self.upload_templates,
                cancel_run=self.cancel_run,
            ),
        )

    # -- panel methods (callable from React via usePanelEvent) --------------

    def load_model(self, ctx):
        """Load (or reuse) the zoo model in the process singleton.

        Blocks until the model is ready (~30 s first time; instant if already
        cached). Called from React via the Load Model button.

        Returns {"status": "ready", "model_name": ...} or {"error": "..."}.
        """
        model_name = ctx.params.get("model_name", "clip-vit-base32-torch")
        print(f"[CropQuery] load_model called, model='{model_name}'")
        try:
            _get_model(model_name)
            print(f"[CropQuery] load_model success: '{model_name}'")
            return {"status": "ready", "model_name": model_name}
        except Exception as exc:
            print(f"[CropQuery] load_model error: {exc}")
            return {"error": str(exc)}

    def browse_directory(self, ctx):
        """List directories and image files at the given path."""
        raw_path = ctx.params.get("path", "~")
        path = os.path.expanduser(raw_path)
        print(f"[CropQuery] browse_directory called, path='{path}'")

        if not os.path.isdir(path):
            print(f"[CropQuery] Not a directory: {path}")
            return {"error": f"Not a directory: {path}", "entries": [], "path": path}

        entries = []
        image_count = 0
        try:
            for name in sorted(os.listdir(path)):
                if name.startswith("."):
                    continue
                full = os.path.join(path, name)
                if os.path.isdir(full):
                    entries.append({"name": name, "path": full})
                else:
                    ext = os.path.splitext(name)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        image_count += 1
        except PermissionError:
            print(f"[CropQuery] Permission denied: {path}")
            return {
                "error": f"Permission denied: {path}",
                "entries": [],
                "path": path,
            }

        print(
            f"[CropQuery] browse: {len(entries)} dirs, "
            f"{image_count} images in {path}"
        )
        return {
            "entries": entries,
            "path": path,
            "parent": os.path.dirname(path),
            "image_count": image_count,
        }

    def _make_thumbnail(self, img, fname):
        """Generate a 128px base64 JPEG thumbnail from a BGR cv2 image.

        Returns a dict with ``thumbnail``, ``width``, ``height`` on success,
        or partial dict with only ``thumbnail: None`` on failure.
        """
        import cv2

        h, w = img.shape[:2]
        max_dim = 128
        thumb = img
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            thumb = cv2.resize(img, None, fx=scale, fy=scale)

        ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            b64 = base64.b64encode(buf.tobytes()).decode()
            print(f"[CropQuery]   • {fname}  {w}×{h}  thumb={len(b64)} chars")
            return {
                "thumbnail": f"data:image/jpeg;base64,{b64}",
                "width": w,
                "height": h,
            }

        print(f"[CropQuery]   • {fname}  (encode failed)")
        return {"thumbnail": None}

    def list_templates(self, ctx):
        """Read template directory and return file list with thumbnails."""
        import cv2

        template_dir = ctx.params.get("template_dir", "")
        print(f"[CropQuery] list_templates called, dir='{template_dir}'")

        if not template_dir:
            print("[CropQuery] No directory provided")
            return {"error": "No directory path provided", "templates": []}

        template_dir = os.path.expanduser(template_dir)

        if not os.path.isdir(template_dir):
            print(f"[CropQuery] Not a directory: {template_dir}")
            return {"error": f"Directory not found: {template_dir}", "templates": []}

        templates = []
        for fname in sorted(os.listdir(template_dir)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            fpath = os.path.join(template_dir, fname)
            entry = {"filename": fname, "filepath": fpath}
            img = cv2.imread(fpath)
            if img is not None:
                entry.update(self._make_thumbnail(img, fname))
            else:
                print(f"[CropQuery]   • {fname}  (could not read)")
                entry["thumbnail"] = None

            templates.append(entry)

        print(f"[CropQuery] Found {len(templates)} template image(s)")
        return {"templates": templates, "count": len(templates)}

    def upload_templates(self, ctx):
        """Receive base64-encoded image files from the browser, save them to
        UPLOAD_DIR, and return thumbnails in the same format as list_templates.

        Clears UPLOAD_DIR at the start of each call so there is no stale
        accumulation from previous sessions.

        Expects ctx.params["files"] = [{name, content (base64 without prefix)}].
        Returns {"templates": [...], "count": N, "upload_dir": UPLOAD_DIR}
        or {"error": "..."}.
        """
        import cv2

        files = ctx.params.get("files", [])
        print(f"[CropQuery] upload_templates called, {len(files)} file(s)")

        if not files:
            return {"error": "No files received", "templates": []}

        # Clear and recreate the upload directory
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        for existing in os.listdir(UPLOAD_DIR):
            try:
                os.remove(os.path.join(UPLOAD_DIR, existing))
            except OSError:
                pass

        templates = []
        for f in files:
            fname = f.get("name", "")
            content_b64 = f.get("content", "")

            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                print(f"[CropQuery] Skipping non-image file: {fname}")
                continue

            try:
                img_bytes = base64.b64decode(content_b64)
            except Exception as exc:
                print(f"[CropQuery] Could not decode {fname}: {exc}")
                continue

            fpath = os.path.join(UPLOAD_DIR, fname)
            with open(fpath, "wb") as fp:
                fp.write(img_bytes)

            img = cv2.imread(fpath)
            if img is None:
                print(f"[CropQuery]   • {fname}  (cv2 could not read after save)")
                continue

            entry = {"filename": fname, "filepath": fpath}
            entry.update(self._make_thumbnail(img, fname))
            templates.append(entry)

        print(f"[CropQuery] Saved {len(templates)} template(s) to {UPLOAD_DIR}")
        return {
            "templates": templates,
            "count": len(templates),
            "upload_dir": UPLOAD_DIR,
        }

    def cancel_run(self, ctx):
        """Signal the running operator to stop after the current sample.

        Sets a flag on the process singleton that RunCropQuery.execute()
        checks at the top of each loop iteration.
        """
        _persist.cancel_requested = True
        print("[CropQuery] cancel_run called — cancellation flag set")
        return {"cancelled": True}


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class RunCropQuery(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="run_crop_query",
            label="Run Template Match",
            description=(
                "Run embedding-based template matching across the dataset. "
                "Extracts overlapping image patches, embeds them with a zoo "
                "model, and computes cosine similarity against template embeddings. "
                "Produces per-sample heatmaps, scores, and tags."
            ),
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "template_dir",
            label="Template Directory",
            required=True,
            description="Path to directory containing cropped template images",
        )
        inputs.str(
            "model_name",
            label="Embedding Model",
            default="clip-vit-base32-torch",
            description=(
                "Any FiftyOne zoo model that supports embeddings "
                "(implements EmbeddingsMixin). E.g. clip-vit-base32-torch, "
                "clip-vit-large14-torch, resnet50-imagenet-torch."
            ),
        )
        inputs.int(
            "n_cols",
            label="Patches per Row",
            default=10,
            min=2,
            max=32,
            description=(
                "How many columns to divide each image into. "
                "Set roughly to image_width / expected_object_width. "
                "E.g. if the target spans ~1/10 of the image width, use 10."
            ),
        )
        inputs.int(
            "n_rows",
            label="Patches per Column",
            default=7,
            min=2,
            max=32,
            description=(
                "How many rows to divide each image into. "
                "Set roughly to image_height / expected_object_height."
            ),
        )
        inputs.int(
            "overlap_pct",
            label="Patch Overlap (%)",
            default=50,
            min=0,
            max=75,
            description=(
                "How much adjacent patches overlap. "
                "50% overlap means each patch shares half its area with its neighbour. "
                "Higher overlap = denser coverage but more patches to embed (slower)."
            ),
        )
        inputs.str(
            "score_field",
            label="Score Field",
            default="template_score",
            description="Dataset field for the raw max cosine similarity score (float).",
        )
        inputs.float(
            "tag_threshold",
            label="Tag Threshold",
            default=0.7,
            description=(
                "Samples with score >= this value get tagged. "
                "Applies to raw cosine similarity (roughly 0–1)."
            ),
        )
        inputs.float(
            "heatmap_threshold",
            label="Heatmap Cutoff",
            default=0.5,
            description=(
                "Minimum cosine similarity for a patch to appear in the heatmap. "
                "Patches below this value are shown as 0 (dark/transparent). "
                "Increase to show only the strongest matches; decrease to show more context."
            ),
        )
        inputs.str(
            "tag_name",
            label="Tag Name",
            default="potential_match",
        )
        inputs.view_target(ctx)
        return types.Property(inputs)

    def execute(self, ctx):
        import cv2
        import numpy as np

        # ---- params -------------------------------------------------------
        template_dir      = os.path.expanduser(ctx.params["template_dir"])
        model_name        = ctx.params.get("model_name", "clip-vit-base32-torch")
        n_cols            = int(ctx.params.get("n_cols", 10))
        n_rows            = int(ctx.params.get("n_rows", 7))
        overlap_pct       = int(ctx.params.get("overlap_pct", 50))
        score_field       = ctx.params.get("score_field", "template_score")
        tag_threshold     = float(ctx.params.get("tag_threshold", 0.7))
        heatmap_threshold = float(ctx.params.get("heatmap_threshold", 0.5))
        tag_name          = ctx.params.get("tag_name", "potential_match")

        print("[CropQuery] ══════════════════════════════════════════")
        print("[CropQuery] Starting crop query")
        print(f"[CropQuery]   template_dir      : {template_dir}")
        print(f"[CropQuery]   model_name        : {model_name}")
        print(f"[CropQuery]   grid              : {n_cols} cols × {n_rows} rows")
        print(f"[CropQuery]   overlap_pct       : {overlap_pct}%")
        print(f"[CropQuery]   score_field       : {score_field}")
        print(f"[CropQuery]   tag_threshold     : {tag_threshold}")
        print(f"[CropQuery]   heatmap_threshold : {heatmap_threshold}")
        print(f"[CropQuery]   tag_name          : {tag_name}")
        print(f"[CropQuery]   view_target param : {ctx.params.get('view_target', 'DATASET')}")
        print(f"[CropQuery]   ctx.dataset count : {len(ctx.dataset)}")
        print(f"[CropQuery]   ctx.view count    : {len(ctx.view)}")
        print("[CropQuery] ══════════════════════════════════════════")

        # ---- validate directory -------------------------------------------
        if not os.path.isdir(template_dir):
            msg = f"Template directory not found: {template_dir}"
            print(f"[CropQuery] ERROR: {msg}")
            yield {"error": msg}
            return

        # ---- load model ---------------------------------------------------
        print(f"[CropQuery] Loading model '{model_name}' …")
        try:
            model = _get_model(model_name)
        except Exception as exc:
            msg = str(exc)
            print(f"[CropQuery] ERROR loading model: {msg}")
            yield {"error": msg}
            return

        # ---- load and embed templates -------------------------------------
        # template_files (explicit paths from the panel UI) takes priority over
        # scanning template_dir. This lets the user remove individual crops from
        # the grid without touching disk — only the remaining crops get embedded.
        template_files_param = ctx.params.get("template_files", None)
        print("[CropQuery] Embedding template images …")
        template_embeddings = []

        if template_files_param:
            # Explicit list from the React panel — respects per-image removals.
            candidate_paths = [
                (os.path.basename(p), p)
                for p in template_files_param
                if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
            ]
            print(
                f"[CropQuery] Using {len(candidate_paths)} explicit template "
                f"path(s) from panel"
            )
        else:
            # Fallback: scan the directory (called without panel params).
            candidate_paths = [
                (fname, os.path.join(template_dir, fname))
                for fname in sorted(os.listdir(template_dir))
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
            ]
            print(
                f"[CropQuery] Scanning template_dir: {len(candidate_paths)} "
                f"image(s) found"
            )

        for fname, fpath in candidate_paths:
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                print(f"[CropQuery] WARN: could not read {fname}, skipping")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            emb = np.array(model.embed(img_rgb), dtype=np.float32)
            emb /= np.maximum(np.linalg.norm(emb), 1e-8)
            template_embeddings.append(emb)
            print(
                f"[CropQuery]   • {fname}  "
                f"shape={img_rgb.shape}  emb_dim={emb.shape[0]}"
            )

        if not template_embeddings:
            msg = "No valid template images found"
            print(f"[CropQuery] ERROR: {msg}")
            yield {"error": msg}
            return

        print(f"[CropQuery] {len(template_embeddings)} template embedding(s) ready")

        # ---- resolve target view -----------------------------------------
        # ctx.target_view() reads the view_target param FiftyOne passes through
        # its operator execution pipeline, which correctly serialises the
        # active app filters. Fallback values: DATASET → full dataset,
        # CURRENT_VIEW / unset → ctx.view (the filtered view in the app).
        view = ctx.target_view()
        print(
            f"[CropQuery] Target view: {len(view)} samples "
            f"(dataset has {len(ctx.dataset)})"
        )

        # ---- collect sample info -----------------------------------------
        # view.values() issues a single batched query instead of loading
        # every sample document individually — significantly faster on large views.
        print("[CropQuery] Collecting sample filepaths …")
        ids, filepaths = view.values(["id", "filepath"])
        sample_infos = list(zip(ids, filepaths))
        total = len(sample_infos)

        if total == 0:
            print("[CropQuery] WARN: view is empty, nothing to process")
            yield {"error": "No samples to process"}
            return

        heatmap_field = f"{score_field}_heatmap"
        tagged_count = 0

        # Estimate patch count for a typical 1920×1080 image
        _pw = max(1, 1920 // n_cols)
        _ph = max(1, 1080 // n_rows)
        _sx = max(1, int(_pw * (1.0 - overlap_pct / 100.0)))
        _sy = max(1, int(_ph * (1.0 - overlap_pct / 100.0)))
        _nc = (1920 - _pw) // _sx + 1
        _nr = (1080 - _ph) // _sy + 1
        patches_per_image_est = _nc * _nr

        print(
            f"[CropQuery] Output fields: "
            f"'{score_field}' (float), '{heatmap_field}' (Heatmap)"
        )
        print(
            f"[CropQuery] Estimated patches per image (1080p): "
            f"~{patches_per_image_est}  ({_nc} cols × {_nr} rows, "
            f"patch {_pw}×{_ph}px, stride {_sx}×{_sy}px)"
        )
        print(f"[CropQuery] Processing {total} samples sequentially …")

        # Reset any stale cancellation flag from a previous run
        _persist.cancel_requested = False
        cancelled = False

        # ---- sequential embedding loop -----------------------------------
        for i, (sid, fpath) in enumerate(sample_infos):
            if _persist.cancel_requested:
                print(f"[CropQuery] Cancellation requested — stopping after {i} samples")
                cancelled = True
                break

            try:
                result = _embed_match_one(
                    fpath, template_embeddings, model,
                    n_cols, n_rows, overlap_pct, heatmap_threshold
                )
            except Exception as exc:
                print(f"[CropQuery] ERROR on {sid}: {exc}")
                result = None

            if result is not None:
                sample = ctx.dataset[sid]
                sample[heatmap_field] = fo.Heatmap(map=result["heatmap"])
                sample[score_field] = result["score"]

                if result["score"] >= tag_threshold:
                    if tag_name not in sample.tags:
                        sample.tags.append(tag_name)
                    tagged_count += 1

                sample.save()

            completed = i + 1
            if completed % max(1, total // 20) == 0 or completed == total:
                print(
                    f"[CropQuery] {completed}/{total}  "
                    f"tagged_so_far={tagged_count}"
                )

            yield ctx.trigger(
                "set_progress",
                {
                    "progress": completed / total,
                    "label": f"Processed {completed}/{total} samples",
                },
            )

        if cancelled:
            print("[CropQuery] ══════════════════════════════════════════")
            print(
                f"[CropQuery] CANCELLED — {i} processed, {tagged_count} tagged"
            )
            print("[CropQuery] ══════════════════════════════════════════")
        else:
            print("[CropQuery] ══════════════════════════════════════════")
            print(
                f"[CropQuery] DONE — {total} processed, {tagged_count} tagged"
            )
            print("[CropQuery] ══════════════════════════════════════════")

        # ---- create index on score field for fast sorting/filtering ------
        try:
            ctx.dataset.create_index(score_field)
            print(f"[CropQuery] Index created on '{score_field}'")
        except Exception as exc:
            print(f"[CropQuery] WARN: could not create index on '{score_field}': {exc}")

        processed_count = i + 1 if cancelled else total
        yield {
            "success": True,
            "cancelled": cancelled,
            "processed": processed_count,
            "tagged": tagged_count,
            "tag_name": tag_name,
        }

        # ---- clear CUDA cache ----------------------------------------
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[CropQuery] CUDA cache cleared")
            else:
                print("[CropQuery] CUDA not available, skipping cache clear")
        except ImportError:
            print("[CropQuery] torch not importable, skipping CUDA cache clear")

        # ---- reload dataset in the UI --------------------------------
        yield ctx.trigger("reload_dataset")
        print("[CropQuery] Dataset reload triggered")

    def resolve_output(self, ctx):
        outputs = types.Object()
        result = ctx.results or {}

        if "error" in result:
            outputs.str("error", label="Error", default=result["error"])
        elif result.get("cancelled"):
            outputs.str(
                "summary",
                label="Summary",
                view=types.MarkdownView(),
                default=(
                    f"**Run cancelled**\n\n"
                    f"- Processed: {result.get('processed', 0)} samples before cancellation\n"
                    f"- Tagged `{result.get('tag_name', 'potential_match')}`: "
                    f"{result.get('tagged', 0)} samples\n"
                ),
            )
        else:
            outputs.str(
                "summary",
                label="Summary",
                view=types.MarkdownView(),
                default=(
                    f"**Embedding template match complete**\n\n"
                    f"- Processed: {result.get('processed', 0)} samples\n"
                    f"- Tagged `{result.get('tag_name', 'potential_match')}`: "
                    f"{result.get('tagged', 0)} samples\n"
                ),
            )

        return types.Property(outputs)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(p):
    p.register(CropQueryPanel)
    p.register(RunCropQuery)
