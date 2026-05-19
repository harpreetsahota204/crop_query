# Crop Query

**Few-shot annotation triage for FiftyOne.** Find images containing a specific object in a large unlabeled dataset using only a few example crops. No training, no fine-tuning, no GPU required.

---

## Installation

```
fiftyone plugins download https://github.com/harpreetsahota204/crop_query
```

## What it does

You have thousands of unlabeled images and a handful of cropped examples of an object you care about (a bird's nest, a defect on a product, a vehicle type, a sponsor logo). Crop Query finds which images in your dataset likely contain that object and marks where in each image it appears.

The output is:

- **A ranked annotation queue**: images sorted by how closely they match your reference crops, with a configurable tag applied to the top candidates

- **Spatial heatmaps**: overlaid on each image in the FiftyOne modal, showing which region matched the templates most strongly

---

## Use cases

- **Industrial inspection**: defects, anomalies, or specific component types in unlabeled inspection imagery

- **Infrastructure monitoring**: bird's nests on power lines, debris on rail tracks, signs of wear in drone footage

- **Wildlife and ecology**: filtering camera trap frames for a specific species

- **Sports media and broadcast analytics**: sponsor logos, team logos, recurring visual elements across long footage archives

- **Retail and inventory**: locating specific products in shelf imagery

- **Aerial and remote sensing**: aircraft, ships, or vehicle types in satellite or drone scenes

- **Manufacturing QA**: scratches, dents, contamination on parts

---

## How it works

### Reference templates

You provide a directory of pre-cropped image examples, the "templates." These are simply image files you've already prepared: crop out one or more instances of the object you're looking for. More templates covering different angles, lighting conditions, or distances will give broader coverage.

### Embedding-based patch comparison

Rather than comparing raw pixels (like classical template matching), the plugin uses a pretrained vision model from the [FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/index.html) to convert images into semantic embedding vectors. Images that look conceptually similar end up with similar vectors, even under different lighting or slight scale changes.

The comparison works in three steps:

**1. Embed the templates.** Each reference crop is passed through the model once, producing a compact numeric representation of its appearance.

**2. Divide and embed each dataset image.** Each image is cut into a grid of overlapping patches. Every patch is embedded with the same model. Because patches are roughly the same scale as your reference crops, the comparison is fair: you're asking "does this local region look like my template?", not "does this entire image look like my template?" Overlap ensures an object that straddles two patch boundaries still falls squarely inside at least one patch.

**3. Compute similarity.** The cosine similarity between each patch and each template is computed. The highest similarity found across all patches and all templates becomes the image's **score** — a single float you can sort and filter on. A spatial map of those per-patch similarities becomes the **heatmap**, with patches below a configurable threshold zeroed out so only genuine hits show up.

Samples whose score meets your threshold are **tagged**, giving you an annotation queue in one click.

---

## Getting started

### 1. Prepare your templates

Crop out examples of the object you're looking for and save them as image files (JPG, PNG, WEBP, TIFF) in a single directory. A few good examples are enough to start. You can always add more.

Tips:
- Use crops that represent the range of appearances you expect (different distances, angles, lighting)
- Crops don't need to be tightly cropped; some background context is fine
- 3 to 10 templates is usually enough; beyond that, returns diminish quickly

### 2. Open the panel

Open your dataset in FiftyOne and launch the **Crop Query** panel from the panel menu. It appears in the grid view.

### 3. Load your templates

Enter the path to your template directory (or use the Browse button to navigate there). Click **Load** to preview the templates as thumbnails.

### 4. Load a model

Enter a FiftyOne zoo model name and click **Load Model**. The first time you load a model, weights are downloaded and initialized. This can take 30 seconds to several minutes depending on the model and your connection. Subsequent loads in the same session are instant.

**Recommended starting model:** `clip-vit-base32-torch`

CLIP (Contrastive Language-Image Pretraining) is a strong general-purpose vision model that understands appearance, texture, and shape. It works well across a wide range of object types without any task-specific tuning.

### 5. Configure the grid

The grid settings control how each image is divided into patches for comparison:

| Setting | What it controls |
|---|---|
| **Patches per row** | How many columns to cut each image into |
| **Patches per column** | How many rows |
| **Overlap %** | How much adjacent patches share |

**Rule of thumb:** set the grid so that one patch roughly covers the area your target object would occupy in a typical image. If the object spans about 1/10th of the image width, use 10 patches per row.

The live estimate below the sliders shows the expected patch count and patch size in pixels. Use this to sanity-check your settings before running.

### 6. Configure outputs

| Setting | What it controls |
|---|---|
| **Score field** | Name of the float field written to each sample |
| **Tag above** | Samples with score ≥ this value get tagged |
| **Heatmap cutoff** | Patches below this similarity are hidden in the heatmap |
| **Tag name** | The tag applied to high-scoring samples |

### 7. Choose the target

- **Entire dataset**: processes every sample
- **Current view**: processes only the samples currently visible in the grid (respects any active filters or saved views)

Running on a filtered view first is a good way to test your settings on a small subset before committing to the full dataset.

### 8. Run

Click **Run Crop Query**. Progress is tracked in the FiftyOne Runs panel. When complete, the dataset reloads automatically and the new fields appear in the sidebar.

---

## Understanding the outputs

### Score field (`template_score` by default)

A float between roughly 0 and 1 representing how strongly the best-matching patch in that image resembled any of your templates. This is a raw cosine similarity value. **Scores are comparable across all images within a single run using the same templates and model.** Different templates or models produce a new score scale, so don't compare scores across runs.

Use this field to:
- Sort the dataset by score (descending) to see the strongest matches first
- Filter to samples above a threshold: `dataset.filter_samples(F("template_score") > 0.75)`
- Build a quick annotation queue: `dataset.match_tags("potential_match")`

### Heatmap field (`template_score_heatmap` by default)

A spatial overlay visible in the FiftyOne sample modal. The bright regions are where the model found the strongest similarity to your templates. Dark regions scored below the heatmap cutoff threshold and are shown as transparent.

The heatmap's spatial resolution depends on your grid settings. Finer grids produce more localized hot spots. The hot spot is roughly the size of one patch and is centered on the best-matching region.

### Tags

Samples whose score meets or exceeds the **Tag above** threshold are tagged with your chosen tag name. This is the fastest path to an annotation queue:

```python
import fiftyone as fo

dataset = fo.load_dataset("your_dataset")
annotation_queue = dataset.match_tags("potential_match")
```

---

## Best practices

### Calibrate the thresholds before tagging the full dataset

Run on a small view (20 to 50 samples) first. After it completes, sort by `template_score` descending and manually check:
- What score do genuine matches get?
- What score do non-matches get?
- Is there a clear gap between the two groups?

Set your **Tag above** threshold at the valley between those two groups, then re-run on the full dataset.

### Match patch size to object size

The single most important setting. If your object occupies roughly:
- 1/5 of the image: use 5 to 6 patches per row
- 1/10 of the image: use 10 to 12 patches per row
- 1/20 of the image: use 15 to 20 patches per row

Patches that are too large capture too much background and dilute the similarity. Patches that are too small may not capture enough of the object for a meaningful comparison.

### Start with lower overlap, increase if needed

0% overlap is fastest. 50% overlap gives better coverage and avoids missing objects that fall on patch boundaries. 75% is rarely necessary and significantly increases compute time.

### Use the heatmap threshold to sharpen the overlay

The default heatmap cutoff (0.5) shows any patch above that raw similarity. If the heatmap looks noisy or spreads across the whole image, raise the cutoff to 0.6 or 0.7 to keep only the strongest hits visible.

### High scores everywhere means the threshold needs calibration

CLIP cosine similarities for natural images cluster in a relatively narrow range. Unrelated patches can still score 0.5 to 0.6. If every image is getting tagged, raise the **Tag above** threshold (try 0.80 to 0.85) and look at the actual score distribution in the sidebar to find where the real matches separate from the background.

### Multiple templates beat one perfect template

A single template anchors the comparison to one specific appearance. Two or three templates covering different distances, angles, or lighting conditions will catch more true positives without requiring retraining.

---

## Performance

Embedding-based matching is more compute-intensive than classical pixel matching. Rough estimates on CPU with CLIP ViT-B/32:

| Grid (cols × rows) | Overlap | ~Patches/image | ~Time/image |
|---|---|---|---|
| 10 × 7 | 0% | 70 | ~0.5s |
| 10 × 7 | 50% | 190 | ~1.5s |
| 16 × 10 | 50% | 480 | ~3.5s |

For large datasets, use **Delegated Execution** (available via FiftyOne's operator settings) to run the job as a background process without blocking the app.

A GPU is not required but provides a 5 to 20× speedup depending on hardware and model.

---

## Limitations

- Produces heatmaps and scores, not bounding boxes. A human annotator still draws the boxes.
- Spatial resolution of the heatmap is limited by patch size. Very small objects require fine grids and longer runtimes.
- Similarity scores are relative to the provided templates. A low score means "not like your crops," not "object absent."

---

## When to reach for something else

Crop Query is a triage and discovery tool, not a detector. If you need:

- **Tight bounding boxes** rather than heatmaps: use an open-vocabulary detector like OWLv2 or Grounding DINO with image prompts
- **Precise instance counting**: use a few-shot counting model
- **Production inference on GPU at scale**: train a dedicated detector after using Crop Query to source the initial annotation set

---

## Related concepts

Crop Query implements a workflow that goes by several names in the literature:

- **Few-Shot Object Localization (FSOL)** ([arxiv 2403.12466](https://arxiv.org/abs/2403.12466)): using a small number of labeled exemplars to find positional information about matching objects in unlabeled images

- **Image-conditioned / visual prompt detection**: using images as detection queries instead of text (related: MQ-Det, OWLv2 image-conditioned mode, Visual Textualization)

- **Query by example / reference-based retrieval**: finding images similar to a small reference set

- **Embedding-based sample selection**: a common pre-annotation pattern in data-centric ML pipelines

- **Similarity-based annotation**: extrapolating labels from a small annotated set to unannotated regions using embedding-space proximity (related: SAFE framework in medical CV)

What's distinctive about Crop Query is the packaging: a single FiftyOne panel with template directory input, sliders that map to object scale, threshold calibration, and a heatmap output, runnable on a laptop CPU.