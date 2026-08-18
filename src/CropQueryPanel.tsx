import React, { useState, useCallback, useEffect } from "react";
import {
  usePanelEvent,
  useOperatorExecutor,
} from "@fiftyone/operators";

// ---------------------------------------------------------------------------
// Module-level persistence cache
// The bundle stays loaded for the entire browser session, so this object
// survives panel unmount/remount (clicking away and back). It resets only
// on a full page refresh, which is acceptable for a triage session.
// ---------------------------------------------------------------------------
const _cache: Record<string, any> = {};

function usePersistentState<T>(key: string, defaultValue: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() =>
    key in _cache ? _cache[key] : defaultValue
  );
  const setPersistent = useCallback(
    (next: T) => {
      _cache[key] = next;
      setValue(next);
    },
    [key]
  );
  return [value, setPersistent];
}

/* eslint-disable no-console */

const PLUGIN_NAME = "@harpreetsahota/cropquery";
const PANEL_NAME  = "crop_query_panel";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TemplateInfo {
  filename: string;
  filepath: string;
  thumbnail: string | null;
  width?: number;
  height?: number;
}

interface BrowseEntry {
  name: string;
  path: string;
}

type ModelStatus = "idle" | "loading" | "ready" | "error";

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function CropQueryPanel() {

  // ---- persistent template state (survive panel unmount/remount) --------
  const [templateDir, setTemplateDir]           = usePersistentState<string>("templateDir", "");
  const [templates, setTemplates]               = usePersistentState<TemplateInfo[]>("templates", []);
  const [templatesLoaded, setTemplatesLoaded]   = usePersistentState<boolean>("templatesLoaded", false);

  // ---- transient template state (fine to reset on remount) --------------
  const [templatesError, setTemplatesError]     = useState<string | null>(null);

  // ---- transient upload state -------------------------------------------
  const [isDragOver, setIsDragOver]             = useState(false);
  const [isUploading, setIsUploading]           = useState(false);
  const [uploadError, setUploadError]           = useState<string | null>(null);

  // ---- transient URL-add state -------------------------------------------
  const [templateUrl, setTemplateUrl]           = useState("");
  const [isUrlLoading, setIsUrlLoading]         = useState(false);
  const [urlError, setUrlError]                 = useState<string | null>(null);

  // ---- transient browse state -------------------------------------------
  const [browsing, setBrowsing]                 = useState(false);
  const [browseEntries, setBrowseEntries]       = useState<BrowseEntry[]>([]);
  const [browsePath, setBrowsePath]             = useState("");
  const [browseParent, setBrowseParent]         = useState("");
  const [browseImageCount, setBrowseImageCount] = useState(0);
  const [browseError, setBrowseError]           = useState<string | null>(null);

  // ---- persistent model state -------------------------------------------
  const [modelName, setModelName]               = usePersistentState<string>("modelName", "clip-vit-base32-torch");
  const [modelStatus, setModelStatus]           = usePersistentState<ModelStatus>("modelStatus", "idle");
  const [loadedModelName, setLoadedModelName]   = usePersistentState<string>("loadedModelName", "");

  // ---- transient model error (fine to reset) ----------------------------
  const [modelError, setModelError]             = useState<string | null>(null);

  // ---- persistent settings ----------------------------------------------
  const [nCols, setNCols]                       = usePersistentState<number>("nCols", 10);
  const [nRows, setNRows]                       = usePersistentState<number>("nRows", 7);
  const [overlapPct, setOverlapPct]             = usePersistentState<number>("overlapPct", 50);
  const [scoreField, setScoreField]             = usePersistentState<string>("scoreField", "template_score");
  const [tagThreshold, setTagThreshold]         = usePersistentState<number>("tagThreshold", 0.7);
  const [heatmapThreshold, setHeatmapThreshold] = usePersistentState<number>("heatmapThreshold", 0.5);
  const [tagName, setTagName]                   = usePersistentState<string>("tagName", "potential_match");
  // "DATASET" / "CURRENT_VIEW" are FiftyOne's canonical view_target values
  const [target, setTarget]                     = usePersistentState<string>("target", "DATASET");

  // ---- transient run state (fine to reset) ------------------------------
  const [runComplete, setRunComplete]           = useState(false);
  const [runResult, setRunResult]               = useState<any>(null);
  const [cancelling, setCancelling]             = useState(false);

  // ---- hooks ------------------------------------------------------------
  const handleEvent = usePanelEvent();
  const executor    = useOperatorExecutor(`${PLUGIN_NAME}/run_crop_query`);

  // ---------------------------------------------------------------------------
  // Load Model
  // ---------------------------------------------------------------------------

  const loadModel = useCallback(() => {
    console.log("[CropQuery] loadModel called, modelName:", modelName);
    setModelStatus("loading");
    setModelError(null);

    handleEvent("load_model", {
      operator: `${PLUGIN_NAME}/${PANEL_NAME}#load_model`,
      params: { model_name: modelName },
      callback: (result: any) => {
        const p = result?.result;

        if (!p) {
          setModelStatus("error");
          setModelError("No response from server");
          return;
        }
        if (p.error) {
          console.error("[CropQuery] load_model error:", p.error);
          setModelStatus("error");
          setModelError(p.error);
          return;
        }
        setModelStatus("ready");
        setLoadedModelName(p.model_name ?? modelName);
      },
    });
  }, [modelName, handleEvent]);

  // ---------------------------------------------------------------------------
  // Browse directory
  // ---------------------------------------------------------------------------

  const doBrowse = useCallback(
    (path: string) => {
      console.log("[CropQuery] doBrowse, path:", path);
      setBrowseError(null);

      handleEvent("browse_directory", {
        operator: `${PLUGIN_NAME}/${PANEL_NAME}#browse_directory`,
        params: { path: path || "~" },
        callback: (result: any) => {
          const p = result?.result;
          if (!p) {
            setBrowseError("No response from server");
            return;
          }
          if (p.error) {
            console.warn("[CropQuery] browse_directory error:", p.error);
            setBrowseError(p.error);
            return;
          }
          setBrowseEntries(p.entries || []);
          setBrowsePath(p.path || "");
          setBrowseParent(p.parent || "");
          setBrowseImageCount(p.image_count || 0);
          setTemplateDir(p.path || "");
        },
      });
    },
    [handleEvent]
  );

  // Shared result handler for list_templates and upload_templates callbacks.
  // Declared before openBrowser/selectBrowseDir which both reference it.
  const handleTemplateListResult = useCallback((result: any) => {
    const payload = result?.result;
    if (!payload) {
      setTemplatesError("No response from server");
      return;
    }
    if (payload.error) {
      console.error("[CropQuery] list_templates error:", payload.error);
      setTemplatesError(payload.error);
      setTemplatesLoaded(true);
      return;
    }
    setTemplates(payload.templates || []);
    setTemplatesLoaded(true);
  }, []);

  // ---------------------------------------------------------------------------
  // File upload (drag-and-drop / click-to-browse)
  // ---------------------------------------------------------------------------

  const handleFiles = useCallback((fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;

    const imageFiles = Array.from(fileList).filter((f) =>
      f.type.startsWith("image/") || /\.(jpe?g|png|webp|tiff?|bmp)$/i.test(f.name)
    );

    if (imageFiles.length === 0) {
      setUploadError("No image files found in selection");
      return;
    }

    console.log(`[CropQuery] Uploading ${imageFiles.length} file(s)`);
    setIsUploading(true);
    setUploadError(null);
    setTemplatesError(null);
    setTemplatesLoaded(false);
    setTemplates([]);

    // Read all files as base64 via FileReader, then send to Python in one call
    const reads = imageFiles.map(
      (file) =>
        new Promise<{ name: string; content: string }>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            const dataUrl = reader.result as string;
            // Strip the "data:image/...;base64," prefix — Python only needs the raw b64
            resolve({ name: file.name, content: dataUrl.split(",")[1] });
          };
          reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
          reader.readAsDataURL(file);
        })
    );

    Promise.all(reads)
      .then((files) => {
        handleEvent("upload_templates", {
          operator: `${PLUGIN_NAME}/${PANEL_NAME}#upload_templates`,
          params: { files },
          callback: (result: any) => {
            setIsUploading(false);
            const payload = result?.result;
            if (payload?.upload_dir) {
              setTemplateDir(payload.upload_dir);
            }
            if (payload?.error) {
              setUploadError(payload.error);
            }
            handleTemplateListResult(result);
          },
        });
      })
      .catch((err) => {
        setIsUploading(false);
        setUploadError(err.message ?? "Failed to read files");
        console.error("[CropQuery] FileReader error:", err);
      });
  }, [handleEvent, handleTemplateListResult]);

  // ---------------------------------------------------------------------------
  // Add template from URL
  // ---------------------------------------------------------------------------

  const loadTemplateFromUrl = useCallback(() => {
    const url = templateUrl.trim();
    if (!url) return;

    console.log("[CropQuery] loadTemplateFromUrl called, url:", url);
    setIsUrlLoading(true);
    setUrlError(null);

    handleEvent("load_template_from_url", {
      operator: `${PLUGIN_NAME}/${PANEL_NAME}#load_template_from_url`,
      params: { url },
      callback: (result: any) => {
        setIsUrlLoading(false);
        const payload = result?.result;
        if (!payload) {
          setUrlError("No response from server");
          return;
        }
        if (payload.error) {
          console.error("[CropQuery] load_template_from_url error:", payload.error);
          setUrlError(payload.error);
          return;
        }
        if (payload.upload_dir) {
          setTemplateDir(payload.upload_dir);
        }
        setTemplates([...templates, ...(payload.templates || [])]);
        setTemplatesLoaded(true);
        setTemplateUrl("");
      },
    });
  }, [templateUrl, templates, handleEvent, setTemplateDir, setTemplates, setTemplatesLoaded]);

  const openBrowser = useCallback(() => {
    setBrowsing(true);
    doBrowse(templateDir || "~");
  }, [templateDir, doBrowse]);

  const selectBrowseDir = useCallback(() => {
    setTemplateDir(browsePath);
    setBrowsing(false);
    setTemplatesError(null);
    setTemplatesLoaded(false);
    setTemplates([]);

    handleEvent("list_templates", {
      operator: `${PLUGIN_NAME}/${PANEL_NAME}#list_templates`,
      params: { template_dir: browsePath },
      callback: handleTemplateListResult,
    });
  }, [browsePath, handleEvent, handleTemplateListResult]);

  // ---------------------------------------------------------------------------
  // Load templates
  // ---------------------------------------------------------------------------

  const loadTemplates = useCallback(() => {
    setTemplatesError(null);
    setTemplatesLoaded(false);
    setTemplates([]);

    handleEvent("list_templates", {
      operator: `${PLUGIN_NAME}/${PANEL_NAME}#list_templates`,
      params: { template_dir: templateDir },
      callback: handleTemplateListResult,
    });
  }, [templateDir, handleEvent, handleTemplateListResult]);

  // ---------------------------------------------------------------------------
  // Run matching
  // ---------------------------------------------------------------------------

  const runMatching = useCallback(async () => {
    // template_files is the current visible set — respects per-image removals
    // without touching disk. Falls back to scanning template_dir on the Python side.
    const params = {
      template_dir:       templateDir,
      template_files:     templates.map((t) => t.filepath),
      model_name:         modelName,
      n_cols:             nCols,
      n_rows:             nRows,
      overlap_pct:        overlapPct,
      score_field:        scoreField,
      tag_threshold:      tagThreshold,
      heatmap_threshold:  heatmapThreshold,
      tag_name:           tagName,
      view_target:        target,
    };
    setRunComplete(false);
    setRunResult(null);

    try {
      const result = await executor.execute(params);
      setRunComplete(true);
      setRunResult(result);
    } catch (e: any) {
      console.error("[CropQuery] run error:", e);
      setRunComplete(true);
      setRunResult({ error: e.message ?? "Unknown error" });
    }
  }, [
    templateDir, templates, modelName, nCols, nRows, overlapPct,
    scoreField, tagThreshold, heatmapThreshold, tagName, target, executor,
  ]);

  const isRunning = executor.isExecuting;
  const canRun    =
    templatesLoaded &&
    templates.length > 0 &&
    modelStatus === "ready" &&
    !isRunning;

  // ---------------------------------------------------------------------------
  // Cancel run
  // ---------------------------------------------------------------------------

  const cancelRun = useCallback(() => {
    if (!isRunning || cancelling) return;
    console.log("[CropQuery] cancelRun called");
    setCancelling(true);

    handleEvent("cancel_run", {
      operator: `${PLUGIN_NAME}/${PANEL_NAME}#cancel_run`,
      params: {},
      callback: () => {
        // cancelling badge stays until isRunning goes false
      },
    });
  }, [isRunning, cancelling, handleEvent]);

  // Reset cancelling badge once the run finishes
  useEffect(() => {
    if (!isRunning) setCancelling(false);
  }, [isRunning]);  // isRunning = executor.isExecuting (FiftyOne global state)

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  const modelStatusBadge = () => {
    if (modelStatus === "idle")
      return <span style={S.badgeIdle}>Not loaded</span>;
    if (modelStatus === "loading")
      return <span style={S.badgeLoading}>Loading…</span>;
    if (modelStatus === "ready")
      return (
        <span style={S.badgeReady}>
          Ready — {loadedModelName}
        </span>
      );
    return <span style={S.badgeError}>Error</span>;
  };

  // ---------------------------------------------------------------------------
  // JSX
  // ---------------------------------------------------------------------------

  return (
    <div style={S.root}>
      <h2 style={S.title}>CropQuery</h2>

      {/* ═══════ Reference Templates ═══════ */}
      <section style={S.section}>
        <div style={S.sectionLabel}>Reference Templates</div>

        <div style={S.row}>
          <input
            style={S.input}
            type="text"
            placeholder="Type a path or click Browse"
            value={templateDir}
            onChange={(e) => {
              setTemplateDir(e.target.value);
              setTemplatesLoaded(false);
              setTemplatesError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") loadTemplates();
            }}
          />
          <button style={S.btnSecondary} onClick={openBrowser}>
            Browse
          </button>
          <button style={S.btnAccent} onClick={loadTemplates}>
            Load
          </button>
        </div>

        {/* ---- divider + dropzone — hidden while browse panel is open ---- */}
        {!browsing && (
        <>
        <div style={S.orDivider}>
          <div style={S.orDividerLine} />
          <span style={S.orText}>or</span>
          <div style={S.orDividerLine} />
        </div>

        <div
          style={{
            ...S.dropzone,
            ...(isDragOver ? S.dropzoneActive : {}),
            ...(isUploading ? S.dropzoneUploading : {}),
          }}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => {
            if (!isUploading) {
              const input = document.createElement("input");
              input.type = "file";
              input.multiple = true;
              input.accept = "image/*";
              input.onchange = () => handleFiles(input.files);
              input.click();
            }
          }}
        >
          {isUploading ? (
            <span style={S.dropzoneText}>Uploading…</span>
          ) : (
            <>
              <span style={S.dropzoneIcon}>📂</span>
              <span style={S.dropzoneText}>
                Drop images here, or click to select
              </span>
              <span style={S.dropzoneHint}>JPG · PNG · WEBP · TIFF</span>
            </>
          )}
        </div>

        {uploadError && <div style={S.error}>{uploadError}</div>}

        <div style={S.orDivider}>
          <div style={S.orDividerLine} />
          <span style={S.orText}>or</span>
          <div style={S.orDividerLine} />
        </div>

        <div style={S.row}>
          <input
            style={S.input}
            type="text"
            placeholder="Paste an image URL"
            value={templateUrl}
            onChange={(e) => {
              setTemplateUrl(e.target.value);
              setUrlError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") loadTemplateFromUrl();
            }}
          />
          <button
            style={{
              ...S.btnAccent,
              ...(isUrlLoading ? S.btnDisabledStyle : {}),
            }}
            onClick={loadTemplateFromUrl}
            disabled={isUrlLoading || !templateUrl.trim()}
          >
            {isUrlLoading ? "Adding…" : "Add"}
          </button>
        </div>

        {urlError && <div style={S.error}>{urlError}</div>}
        </>
        )}

        {/* ---- directory browser ---- */}
        {browsing && (
          <div style={S.browsePanel}>
            <div style={S.browseHeader}>
              <span style={S.browsePathText} title={browsePath}>
                {browsePath || "~"}
              </span>
              <button style={S.btnSmall} onClick={() => setBrowsing(false)}>
                ✕
              </button>
            </div>

            {browseError && <div style={S.error}>{browseError}</div>}

            <div style={S.browseList}>
              {browseParent && browseParent !== browsePath && (
                <div style={S.browseItem} onClick={() => doBrowse(browseParent)}>
                  📁 ..
                </div>
              )}
              {browseEntries.map((entry, i) => (
                <div
                  key={i}
                  style={S.browseItem}
                  onClick={() => doBrowse(entry.path)}
                >
                  📁 {entry.name}
                </div>
              ))}
              {browseEntries.length === 0 && !browseError && (
                <div style={S.muted}>No subdirectories</div>
              )}
            </div>

            {browseImageCount > 0 && (
              <div style={S.browseFooter}>
                {browseImageCount} image(s) in this directory
              </div>
            )}

            <button
              style={{ ...S.btnAccent, width: "100%", marginTop: 8 }}
              onClick={selectBrowseDir}
            >
              Select this directory
            </button>
          </div>
        )}

        {templatesError && <div style={S.error}>{templatesError}</div>}

        {templatesLoaded && templates.length > 0 && (
          <div style={S.templateGrid}>
            {templates.map((t, i) => (
              <div key={i} style={S.templateCard}>
                <div style={S.templateRemoveRow}>
                  <button
                    style={S.templateRemoveBtn}
                    title={`Remove ${t.filename}`}
                    onClick={() => {
                      const next = templates.filter((_, j) => j !== i);
                      setTemplates(next);
                      if (next.length === 0) setTemplatesLoaded(false);
                    }}
                  >
                    ✕
                  </button>
                </div>
                {t.thumbnail ? (
                  <img
                    src={t.thumbnail}
                    alt={t.filename}
                    style={S.templateThumb}
                  />
                ) : (
                  <div style={S.templateNoThumb}>?</div>
                )}
                <div style={S.templateName} title={t.filename}>
                  {t.filename}
                </div>
                {t.width && t.height && (
                  <div style={S.templateDims}>
                    {t.width}×{t.height}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {templatesLoaded && templates.length === 0 && !templatesError && (
          <div style={S.warn}>No image files found in directory</div>
        )}
      </section>

      {/* ═══════ Model ═══════ */}
      <section style={S.section}>
        <div style={S.sectionLabel}>Embedding Model</div>

        <div style={S.row}>
          <input
            style={S.input}
            type="text"
            placeholder="e.g. clip-vit-base32-torch"
            value={modelName}
            onChange={(e) => {
              setModelName(e.target.value);
              setModelStatus("idle");
              setModelError(null);
            }}
          />
          <button
            style={{
              ...S.btnAccent,
              ...(modelStatus === "loading" ? S.btnDisabledStyle : {}),
            }}
            onClick={loadModel}
            disabled={modelStatus === "loading" || !modelName.trim()}
          >
            {modelStatus === "loading" ? "Loading…" : "Load Model"}
          </button>
        </div>

        <div style={S.modelStatusRow}>
          {modelStatusBadge()}
          {modelError && <span style={S.modelErrorText}>{modelError}</span>}
        </div>

        <div style={S.hint}>
          Any FiftyOne zoo model that supports embeddings. First load downloads
          weights (~seconds to minutes). Subsequent loads are instant.
        </div>
      </section>

      {/* ═══════ Settings ═══════ */}
      <section style={S.section}>
        <div style={S.sectionLabel}>Settings</div>

        <Field label="Run against">
          <Radio
            name="target"
            value={target}
            onChange={setTarget}
            options={[
              { value: "DATASET",       label: "Entire dataset" },
              { value: "CURRENT_VIEW",  label: "Current view" },
            ]}
          />
        </Field>

        {/* ---- Grid spec ---- */}
        <Slider
          label="Patches per row"
          value={nCols}
          onChange={(v) => setNCols(Math.round(v))}
          min={2}
          max={32}
          step={1}
          display={(v) => `${Math.round(v)}`}
        />

        <Slider
          label="Patches per column"
          value={nRows}
          onChange={(v) => setNRows(Math.round(v))}
          min={2}
          max={32}
          step={1}
          display={(v) => `${Math.round(v)}`}
        />

        <Slider
          label="Overlap %"
          value={overlapPct}
          onChange={(v) => setOverlapPct(Math.round(v))}
          min={0}
          max={75}
          step={5}
          display={(v) => `${Math.round(v)}%`}
        />

        <PatchEstimate nCols={nCols} nRows={nRows} overlapPct={overlapPct} />

        {/* ---- Output fields ---- */}
        <Field label="Score field">
          <input
            style={S.inputSm}
            value={scoreField}
            onChange={(e) => setScoreField(e.target.value)}
          />
        </Field>

        <Slider
          label="Tag above (score threshold)"
          value={tagThreshold}
          onChange={setTagThreshold}
          min={0}
          max={1}
          step={0.01}
          display={(v) => v.toFixed(2)}
        />

        <Slider
          label="Heatmap cutoff"
          value={heatmapThreshold}
          onChange={setHeatmapThreshold}
          min={0}
          max={0.95}
          step={0.01}
          display={(v) => v.toFixed(2)}
        />
        <div style={S.strideHint}>
          Patches below this cosine similarity show as dark/transparent in the heatmap
        </div>

        <Field label="Tag name">
          <input
            style={S.inputSm}
            value={tagName}
            onChange={(e) => setTagName(e.target.value)}
          />
        </Field>
      </section>

      {/* ═══════ Run ═══════ */}
      <section style={S.section}>
        {!templatesLoaded || templates.length === 0 ? (
          <div style={S.runBlockedMsg}>
            Load template images above before running.
          </div>
        ) : modelStatus !== "ready" ? (
          <div style={S.runBlockedMsg}>
            Load a model above before running.
          </div>
        ) : null}

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button
            style={{
              ...S.runBtn,
              ...(canRun ? {} : S.runBtnDisabled),
            }}
          onClick={runMatching}
          disabled={!canRun}
          >
            {isRunning ? "Running…" : "Run CropQuery"}
          </button>

          {isRunning && (
            <button
              style={{
                ...S.btnSecondary,
                opacity: cancelling ? 0.5 : 1,
                cursor: cancelling ? "default" : "pointer",
              }}
              onClick={cancelRun}
              disabled={cancelling}
            >
              {cancelling ? "Cancelling…" : "Cancel"}
            </button>
          )}
        </div>

        {isRunning && !cancelling && (
          <div style={S.statusMsg}>
            Processing — check the Runs panel for live progress.
          </div>
        )}

        {isRunning && cancelling && (
          <div style={S.statusMsg}>
            Cancellation requested — finishing current sample…
          </div>
        )}

        {runComplete && (
          <div style={runResult?.error ? S.error : S.successMsg}>
            {runResult?.error
              ? `Error: ${runResult.error}`
              : runResult?.cancelled
              ? "Cancelled — partial results saved. Check the Runs panel for details."
              : "Done — check the Runs panel for details"}
          </div>
        )}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper components
// ---------------------------------------------------------------------------

function PatchEstimate({
  nCols,
  nRows,
  overlapPct,
}: {
  nCols: number;
  nRows: number;
  overlapPct: number;
}) {
  // Estimate effective patch count for a typical 1920×1080 image.
  // overlapPct is capped at 75 by the slider so overlapFrac < 1 always holds.
  const overlapFrac = overlapPct / 100;
  const effectiveCols = Math.floor((nCols - 1) / (1 - overlapFrac)) + 1;
  const effectiveRows = Math.floor((nRows - 1) / (1 - overlapFrac)) + 1;
  const total = effectiveCols * effectiveRows;

  const patchW = Math.round(1920 / nCols);
  const patchH = Math.round(1080 / nRows);

  return (
    <div style={S.patchEstimate}>
      ~{total} patches per image ({effectiveCols}c × {effectiveRows}r)
      &nbsp;·&nbsp;{patchW}×{patchH}px each
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div style={S.field}>
      <label style={S.label}>{label}</label>
      {children}
    </div>
  );
}

function Radio({
  name,
  value,
  onChange,
  options,
}: {
  name: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div style={S.radioGroup}>
      {options.map((o) => (
        <label key={o.value} style={S.radioLabel}>
          <input
            type="radio"
            name={name}
            value={o.value}
            checked={value === o.value}
            onChange={() => onChange(o.value)}
          />
          {o.label}
        </label>
      ))}
    </div>
  );
}

function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  display,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  display: (v: number) => string;
}) {
  return (
    <div style={S.field}>
      <label style={S.label}>{label}</label>
      <div style={S.sliderRow}>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          style={S.slider}
        />
        <span style={S.sliderVal}>{display(value)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const S: Record<string, React.CSSProperties> = {
  root: {
    padding: "16px 20px",
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    color: "#e0e0e0",
    height: "100%",
    overflowY: "auto",
  },
  title: {
    fontSize: 18,
    fontWeight: 600,
    marginBottom: 16,
    color: "#fff",
  },
  section: {
    marginBottom: 20,
    borderBottom: "1px solid #333",
    paddingBottom: 16,
  },
  sectionLabel: {
    fontSize: 12,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    color: "#888",
    marginBottom: 8,
  },

  /* inputs */
  row:    { display: "flex", gap: 8, alignItems: "center" },
  input: {
    flex: 1,
    padding: "8px 12px",
    borderRadius: 4,
    border: "1px solid #444",
    background: "#1e1e1e",
    color: "#e0e0e0",
    fontSize: 13,
    outline: "none",
  },
  inputSm: {
    padding: "6px 10px",
    borderRadius: 4,
    border: "1px solid #444",
    background: "#1e1e1e",
    color: "#e0e0e0",
    fontSize: 13,
    outline: "none",
    width: 200,
  },

  /* buttons */
  btnAccent: {
    padding: "8px 16px",
    borderRadius: 4,
    border: "none",
    background: "#ff6d04",
    color: "#fff",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  btnSecondary: {
    padding: "8px 16px",
    borderRadius: 4,
    border: "1px solid #555",
    background: "#2a2a2a",
    color: "#ccc",
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  btnSmall: {
    padding: "2px 8px",
    borderRadius: 3,
    border: "1px solid #555",
    background: "transparent",
    color: "#aaa",
    fontSize: 13,
    cursor: "pointer",
  },
  btnDisabledStyle: {
    background: "#555",
    cursor: "not-allowed",
    color: "#999",
  },
  runBtn: {
    padding: "10px 24px",
    borderRadius: 4,
    border: "none",
    background: "#ff6d04",
    color: "#fff",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
    width: "100%",
  },
  runBtnDisabled: {
    background: "#555",
    cursor: "not-allowed",
    color: "#999",
  },

  /* form fields */
  field:      { marginBottom: 12 },
  label: {
    fontSize: 13,
    color: "#aaa",
    marginBottom: 4,
    display: "block",
  },
  radioGroup: { display: "flex", gap: 16 },
  radioLabel: {
    fontSize: 13,
    color: "#ccc",
    display: "flex",
    alignItems: "center",
    gap: 4,
    cursor: "pointer",
  },
  sliderRow: { display: "flex", alignItems: "center", gap: 12 },
  slider:    { flex: 1, accentColor: "#ff6d04" },
  sliderVal: {
    fontSize: 13,
    color: "#ccc",
    minWidth: 50,
    textAlign: "right",
  },
  strideHint: {
    fontSize: 11,
    color: "#666",
    marginTop: -8,
    marginBottom: 12,
    marginLeft: 2,
  },
  patchEstimate: {
    fontSize: 11,
    color: "#ff6d04",
    background: "#1a0d00",
    border: "1px solid #3a1a00",
    borderRadius: 4,
    padding: "4px 8px",
    marginTop: -6,
    marginBottom: 12,
    fontVariantNumeric: "tabular-nums",
  },

  /* model status */
  modelStatusRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginTop: 6,
  },
  badgeIdle: {
    fontSize: 11,
    color: "#888",
    background: "#222",
    border: "1px solid #444",
    borderRadius: 10,
    padding: "2px 8px",
  },
  badgeLoading: {
    fontSize: 11,
    color: "#ffaa00",
    background: "#2a2200",
    border: "1px solid #664400",
    borderRadius: 10,
    padding: "2px 8px",
  },
  badgeReady: {
    fontSize: 11,
    color: "#44cc44",
    background: "#0a2a0a",
    border: "1px solid #226622",
    borderRadius: 10,
    padding: "2px 8px",
  },
  badgeError: {
    fontSize: 11,
    color: "#ff4444",
    background: "#2a0000",
    border: "1px solid #660000",
    borderRadius: 10,
    padding: "2px 8px",
  },
  modelErrorText: {
    fontSize: 11,
    color: "#ff4444",
    flex: 1,
  },
  hint: {
    fontSize: 11,
    color: "#666",
    marginTop: 6,
    lineHeight: 1.4,
  },

  /* upload dropzone */
  orDivider: {
    display: "flex",
    alignItems: "center",
    margin: "10px 0",
  },
  orDividerLine: {
    flex: 1,
    height: 1,
    background: "#333",
  },
  orText: {
    fontSize: 11,
    color: "#555",
    padding: "0 10px",
    whiteSpace: "nowrap" as const,
  },
  dropzone: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    gap: 4,
    padding: "14px 12px",
    border: "1px dashed #444",
    borderRadius: 4,
    cursor: "pointer",
    transition: "border-color 0.15s, background 0.15s",
    marginBottom: 8,
    minHeight: 72,
  },
  dropzoneActive: {
    borderColor: "#ff6d04",
    background: "#1a0d00",
  },
  dropzoneUploading: {
    borderColor: "#555",
    cursor: "not-allowed",
    opacity: 0.7,
  },
  dropzoneIcon: { fontSize: 20 },
  dropzoneText: { fontSize: 12, color: "#ccc" },
  dropzoneHint: { fontSize: 10, color: "#555" },

  /* browse panel */
  browsePanel: {
    marginTop: 8,
    padding: 10,
    background: "#1a1a1a",
    border: "1px solid #444",
    borderRadius: 4,
  },
  browseHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  browsePathText: {
    fontSize: 12,
    fontFamily: "monospace",
    color: "#ff6d04",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    flex: 1,
    marginRight: 8,
  },
  browseList: {
    maxHeight: 200,
    overflowY: "auto",
    borderTop: "1px solid #333",
    borderBottom: "1px solid #333",
    padding: "4px 0",
  },
  browseItem: {
    fontSize: 13,
    color: "#ccc",
    padding: "4px 8px",
    cursor: "pointer",
    borderRadius: 2,
  },
  browseFooter: {
    fontSize: 12,
    color: "#888",
    marginTop: 6,
  },

  /* template previews */
  templateGrid: {
    marginTop: 10,
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(100px, 1fr))",
    gap: 8,
  },
  templateCard: {
    background: "#1a1a1a",
    borderRadius: 4,
    overflow: "hidden",
    border: "1px solid #333",
    textAlign: "center",
    position: "relative",
  },
  templateRemoveRow: {
    display: "flex",
    justifyContent: "flex-end",
    padding: "2px 2px 0 0",
  },
  templateRemoveBtn: {
    background: "transparent",
    border: "none",
    color: "#666",
    fontSize: 10,
    cursor: "pointer",
    padding: "0 2px",
    lineHeight: 1,
  },
  templateThumb: {
    width: "100%",
    height: 80,
    objectFit: "cover",
    display: "block",
  },
  templateNoThumb: {
    width: "100%",
    height: 80,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#222",
    color: "#666",
    fontSize: 24,
  },
  templateName: {
    fontSize: 10,
    color: "#ccc",
    padding: "4px 4px 0 4px",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  templateDims: {
    fontSize: 9,
    color: "#666",
    padding: "0 4px 4px 4px",
  },

  /* messages */
  muted:          { fontSize: 12, color: "#888", padding: "4px 8px" },
  error:          { marginTop: 8, color: "#ff4444", fontSize: 12 },
  warn:           { marginTop: 8, color: "#ffaa00", fontSize: 12 },
  statusMsg:      { marginTop: 8, color: "#ffaa00", fontSize: 13 },
  successMsg:     { marginTop: 8, color: "#44ff44", fontSize: 13 },
  runBlockedMsg:  { fontSize: 12, color: "#888", marginBottom: 8 },
};
