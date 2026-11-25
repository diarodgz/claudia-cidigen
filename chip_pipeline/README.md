# ____ (project title)

> **How to read this document.** Anything marked `____` was not provided and is left for you to fill in.
> Values marked *(default in code)* are the defaults written in the notebook code; the **Value used** column is for the values you actually ran with.

**Summary.** ____ 

This repository/notebook turns a very large RGBA orthomosaic plus a DSM and a DEM into a labeled dataset of 224 × 224 px image chips of elevated structures (roofs, trees and other objects taller than 2 m), ready for training or evaluating CLIP-style models. It is designed to be memory-efficient: the full orthomosaic, DSM and DEM are never loaded at full resolution, only one small window at a time.

## Contents

0. [Pipeline](#0-pipeline-at-a-glance)
1. [Requirements and environment](#1-requirements-and-environment)
2. [Files and formats](#2-files-used-and-file-formats)
3. [Variables](#3-variable-definitions)
4. [Tiling and identification of candidate structures](#4-tiling-and-identification-of-candidate-structures)
5. [Object segmentation (SAM) and chip creation](#5-object-segmentation-sam-and-chip-creation)
6. [Chip labeling tool](#6-chip-labeling-tool)
7. [Export for training](#7-export-for-training)
8. [Outputs](#8-outputs-of-each-process)
9. [Running and resuming](#9-running-and-resuming)
10. [Known limitations](#10-known-limitations)
11. [Troubleshooting](#11-troubleshooting)
12. [References, license and contact](#12-references-license-and-contact)

---

## 0. Pipeline at a glance

```
Orthomosaic (RGBA GeoTIFF) ─┐
DSM (GeoTIFF) ──────────────┼─► [4] Tile grid (500 x 500 px) + nDSM per tile
DEM (GeoTIFF) ──────────────┘        + structure detection (nDSM > 2 m) + grouping
                                      │
                                      ▼  grid.parquet (tile table)
                          [5] SAM ViT-B on each "structures" tile
                                      + one 224 x 224 chip per object
                                      │
                                      ▼  chips/*.png + chips.csv
                          [6] Labeling tool (5 classes)
                                      │
                                      ▼  labels.csv
                          [7] Spatial train/val/test export
                                      │
                                      ▼  export/{train,val,test}.csv + classes.json
```

| Stage | Notebook section / cell | Needs GPU |
|---|---|---|
| Tile grid, nDSM statistics, grouping | ____ | No |
| SAM object finding and chip creation | ____ | **Yes** |
| Labeling tool | ____ | No |
| Export | ____ | No |

---

## 1. Requirements and environment

### 1.1 Runtime

| Item | Value |
|---|---|
| Platform | Google Colab Pro |
| Runtime type | Python 3.13.16 |
| Hardware accelerator | **GPU** (required for the SAM step; the notebook asserts `cuda`). GPU model: `____` |
| High-RAM runtime | `____` |
| System RAM / disk shown by Colab | `____` / `____` |
| Date of the runs | `____` |

The tiling and nDSM steps run on CPU. The SAM step (section 5) needs a GPU runtime; with no GPU, loading SAM fails (see [Troubleshooting](#11-troubleshooting)).

### 1.2 Packages

| Package | Used for | Version |
|---|---|---|
| Python | interpreter | ____ |
| numpy | arrays (version 2.x or later: `ndarray.ptp` no longer exists, so the code uses `np.ptp`) | ____ |
| pandas | tables, CSV reading and writing | ____ |
| geopandas | tile table with geometries, plotting, parquet | ____ |
| shapely | `box`, `shape`, `unary_union` | ____ |
| rasterio | raster I/O, windows, `WarpedVRT`, `features.shapes` | ____ |
| affine | pixel-to-map transforms (`Affine.scale`) | ____ |
| scipy | `scipy.ndimage`: labeling, distance transform, morphology | ____ |
| scikit-image | `remove_small_objects`, `h_maxima`, `watershed` | ____ |
| matplotlib | all plots | ____ |
| Pillow | chip crop, resize and PNG writing | ____ |
| tqdm | progress bars | ____ |
| ipywidgets | labeling tool | ____ |
| IPython | `display` | ____ |
| pyarrow | `grid.parquet` (GeoDataFrame to/from parquet) | ____ |
| torch (CUDA build) | running SAM | ____ |
| torchvision | SAM dependency | ____ |
| segment-anything (Meta AI) | SAM model code | ____ |
| open_clip_torch (optional) | only if you train or evaluate with open_clip / RemoteCLIP | ____ |

Standard library modules also used: `pathlib`, `csv`, `json`, `time`.

### 1.3 Installation (Colab)

```python
from google.colab import drive
drive.mount("/content/drive")

!pip install -q git+https://github.com/facebookresearch/segment-anything.git
!pip install -q rasterio geopandas scikit-image ipywidgets tqdm pyarrow
```

Do not upgrade `torch` in Colab; the preinstalled build is the CUDA-enabled one.

### 1.4 Record the exact versions of your run

Run this cell and paste the output into the table above:

```python
import importlib.metadata as md, torch
for p in ["numpy", "pandas", "geopandas", "shapely", "rasterio", "affine", "scipy", "scikit-image",
          "matplotlib", "pillow", "tqdm", "ipywidgets", "pyarrow", "torch", "torchvision",
          "segment_anything", "open_clip_torch"]:
    try:
        print(f"{p}: {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p}: not installed")
print("CUDA build:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```

```python
!python --version
!nvidia-smi
```

---

## 2. Files used and file formats

### 2.1 Inputs

| Role | Variable | Format | Details |
|---|---|---|---|
| Orthomosaic | `dom_path` | GeoTIFF, 4 bands (R, G, B, alpha), `uint8`, nodata = 0 | Width 24,508 px × height 26,629 px. Pixel size about 0.0542 m (5.42 cm/px), derived from the bounds below; confirm with `src.res`. CRS: UTM-type coordinates, zone/EPSG `____`. Bounds: left 263910.9354684872, bottom 6341422.044243328, right 265239.2690684872, top 6342865.336043328 (about 1,328 × 1,443 m). The valid footprint is a rotated rectangle: about 43% of the bounding box has data (alpha > 0). Acquisition date `____`, sensor/platform `____`, file size `____`, internal tiling/compression `____` |
| DSM | `DSM_PATH` | GeoTIFF | Bands `____`, dtype `____`, resolution `____`, CRS `____`, nodata value `____`, vertical reference `____`, source/software `____` |
| DEM | `DEM_PATH` | GeoTIFF | Bands `____`, dtype `____`, resolution `____`, CRS `____`, nodata value `____`, vertical reference `____`, how it was produced `____` |
| SAM checkpoint | `CKPT` | PyTorch checkpoint (`.pth`) | `/content/drive/MyDrive/sam/sam_vit_b_01ec64.pth`, Meta AI Segment Anything **ViT-B** (base) weights. Downloaded from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` |
| Area of interest (optional) | `kml_path` | KML (WGS84, EPSG:4326) | Not used by the current pipeline. Path `____`; description `____` |

Where the files live while the notebook runs: `____` (recommended: copy the three rasters from Drive to `/content` before running, because window reads over the Drive mount are slow).

### 2.2 Requirements on the inputs

- The DSM and DEM are resampled **on the fly** to the orthomosaic's pixel grid (bilinear), so they do not need the same resolution or CRS as the orthomosaic. They must overlap it and use the same vertical reference.
- nDSM = DSM − DEM, in the units of the DSM/DEM (assumed to be meters).
- Pixels outside the orthomosaic footprint (alpha = 0) are set to nodata in the nDSM.

### 2.3 Files written by the pipeline

All under `OUT = /content/drive/MyDrive/roof_chips` (change if needed: `____`).

```
roof_chips/
  grid.parquet        tile table (geometries + per-tile statistics + groups)
  chips/              224 x 224 RGB PNG chips, one per object
  chips.csv           chip manifest (one row per chip)
  done_tiles.txt      ids of tiles already processed by SAM (for resuming)
  labels.csv          labels assigned in the labeling tool
  export/             train.csv, val.csv, test.csv, classes.json
```

Details are in [section 8](#8-outputs-of-each-process).

---

## 3. Variable definitions

### 3.1 Parameters of the tiling and structure detection (section 4)

| Variable | Meaning | Default in code | Value used |
|---|---|---|---|
| `TILE` | Tile side in pixels (500 px is about 27 m) | 500 | ____ |
| `OVERLAP` | Pixels of overlap between neighboring tiles; step = `TILE − OVERLAP` | 0 | ____ |
| `EDGE` | Edge handling. `"shift"`: the last row/column of tiles moves inward so every tile is exactly `TILE` × `TILE` (it overlaps its neighbor). `"partial"`: no overlap, the last row/column is smaller (8 px wide here) | `"shift"` | ____ |
| `MIN_VALID` | A tile is "no data" when its share of valid pixels in the overview is ≤ this value (0.0 = only completely empty tiles) | 0.0 | ____ |
| `F` | Overview factor: the overview is `1/F` of the full resolution (about 0.54 m/px for `F = 10`) | 10 | ____ |
| `HEIGHT_THR` | A structure is anything with nDSM above this height (m) | 2.0 | ____ |
| `MIN_STRUCT_M2` | Smallest structure that counts (m²) | 2.0 | ____ |
| `NDSM_RANGE` | nDSM values outside this range (m) are treated as nodata | (−20, 250) | ____ |

### 3.2 Derived raster variables

| Variable | Meaning |
|---|---|
| `H`, `W` | Orthomosaic height and width in pixels (26,629 and 24,508) |
| `tf` | Affine transform of the orthomosaic (pixel to map coordinates) |
| `crs` | CRS of the orthomosaic |
| `bounds` | Bounding box of the orthomosaic (`left, bottom, right, top`) |
| `res` | Pixel size in meters (`src.res[0]`) |
| `step` | `TILE − OVERLAP` |
| `ov` | RGBA overview, shape `(4, H // F, W // F)`, average resampling |
| `valid_ov` | Boolean overview of the valid footprint (`ov[3] > 127`) |
| `fy`, `fx` | Full-resolution pixels per overview pixel along rows and columns |
| `yy`, `xx` | Overview pixel centers expressed in full-resolution pixels |
| `footprint` | Polygon of the valid footprint (from `valid_ov`, simplified) |
| `min_px` | `int(MIN_STRUCT_M2 / res²)`, the minimum structure size in pixels (about 680 px with the defaults) |

### 3.3 Functions and classes

| Name | Purpose |
|---|---|
| `TileReader` | Opens the three rasters lazily and reads one window at a time: RGB, validity mask and nDSM. DSM and DEM are read through `WarpedVRT` on the orthomosaic grid |
| `starts(total, tile, step)` | Start offsets of the tiles along one axis, following `EDGE` |
| `ov_slice(...)` | Overview pixels whose centers fall inside a full-resolution window |
| `clean_struct(mask)` | 3 × 3 binary opening, then removal of objects smaller than `min_px` |
| `show_tile(tile_id)` | Plots RGB, nDSM and the structure mask of one tile |
| `TileDataset` | Lazy dataset of tiles (image, validity mask, nDSM, structure mask) |
| `seed_regions(struct)` | Splits the structure mask into candidate regions (section 5) |
| `find_objects(win)` | Runs SAM on a window and returns object masks (section 5) |
| `process_tile(t)` | Finds the objects of one tile, creates the chips and returns manifest rows |

### 3.4 Tile table `grid` (GeoDataFrame, saved as `grid.parquet`)

| Column | Meaning |
|---|---|
| `tile_id` | Unique tile id (also the index) |
| `ri`, `ci` | Row and column index of the tile in the grid |
| `row_off`, `col_off` | Top-left corner of the tile in full-resolution pixels |
| `width`, `height` | Tile size in pixels |
| `geometry` | Footprint polygon of the tile in map coordinates (`crs`) |
| `valid_frac_ov` | Valid-pixel share estimated from the overview |
| `has_data` | `valid_frac_ov > MIN_VALID` |
| `valid_frac` | Exact valid-pixel share at full resolution (tiles with data only) |
| `ndsm_cover` | Share of valid pixels that have an nDSM value |
| `struct_px` | Pixels belonging to structures (cleaned mask of nDSM > `HEIGHT_THR`) |
| `struct_frac` | `struct_px` divided by the valid pixels of the tile |
| `ndsm_max` | Maximum nDSM value in the tile (m) |
| `group` | `"no data"`, `"ground only"` or `"structures"` |
| `cluster` | Id of the cluster of adjacent structure tiles (0 = not a structure tile) |
| `own_r0`, `own_r1`, `own_c0`, `own_c1` | Rows and columns "owned" by the tile cell (the nominal grid cell). An object belongs to the tile whose owned area contains its centroid, which avoids duplicates in overlap strips |

### 3.5 Parameters of the SAM and chip step (section 5)

| Variable | Meaning | Default in code | Value used |
|---|---|---|---|
| `OUT` | Output folder (Drive) | `/content/drive/MyDrive/roof_chips` | ____ |
| `CKPT` | SAM ViT-B checkpoint | `/content/drive/MyDrive/sam/sam_vit_b_01ec64.pth` | ____ |
| `HALO` | Pixels of context added around each tile cell when running SAM, so objects crossing a tile border stay whole | 64 | ____ |
| `CHIP` | Chip side in pixels | 224 | ____ |
| `MARGIN` | Context around the object, as a fraction of its size on each side | 0.25 | ____ |
| `MIN_SIDE` | Smallest source window for a chip (px) | 96 | ____ |
| `BG_FADE` | 0 = keep the surroundings; 1 = paint everything outside the object mid-gray | 0.0 | ____ |
| `SPLIT` | Split touching structures into separate objects | `True` | ____ |
| `H_M` | Depth of the "neck" (m) between touching structures required to split them; larger = fewer splits | 0.7 | ____ |
| `PAD_PX` | Padding added to each object's box prompt (px) | 8 | ____ |
| `DILATE_PX` | How far a SAM mask may extend beyond the nDSM structure mask (px) | 12 | ____ |
| `MIN_INSIDE` | Minimum share of a SAM mask that must lie on the nDSM structure mask; otherwise the nDSM object is used | 0.6 | ____ |
| `chunk` (in `find_objects`) | Prompts sent to SAM per batch | 32 | ____ |
| `iou_thr` (in `find_objects`) | Masks with a higher IoU than this are considered duplicates | 0.6 | ____ |
| `TIME_BUDGET_MIN` | Stop cleanly after this many minutes (`None` = no limit) | `None` | ____ |

### 3.6 Columns of `chips.csv`

| Column | Meaning |
|---|---|
| `chip_id` | Unique id, `t{tile_id:04d}_o{k:03d}` |
| `filepath` | Chip path relative to `OUT` (`chips/<chip_id>.png`) |
| `tile_id`, `ri`, `ci`, `cluster` | Tile the object belongs to, its grid position and cluster |
| `center_e`, `center_n` | Object centroid in map coordinates (easting, northing) |
| `col_px`, `row_px` | Object centroid in full-resolution orthomosaic pixels |
| `area_m2` | Object area |
| `height_mean`, `height_max` | Mean and maximum nDSM value inside the object (m) |
| `sam_score` | SAM's predicted mask quality score |
| `source` | `sam` (SAM mask used) or `ndsm` (SAM mask rejected, the nDSM object was used) |
| `chip_gsd_cm` | Effective pixel size of the chip in cm (`res × 100 × side / CHIP`), since each chip is resized from a window of a different size |

### 3.7 Labeling and export variables

| Variable | Meaning |
|---|---|
| `LABELS` | The five classes: `light metal sheet`, `oxidized metal sheet`, `precarious/mixed`, `vegetation`, `other` (their order defines `label_id` 0 to 4) |
| `manifest` | `chips.csv` loaded, duplicates removed (`keep="last"`) |
| `LABEL_FILE` | `OUT / "labels.csv"` |
| `labels` | Dictionary `chip_id → label` held in memory and saved on every click |
| `CAPTIONS` | Text caption per label, used as the `title` column of the exports |
| `val_frac`, `test_frac`, `block`, `seed` | Export parameters: validation and test fractions (default 0.15 each), block size in tiles (default 4), random seed (default 0). Values used: `____` |

---

## 4. Tiling and identification of candidate structures

The goal of this stage is to divide the orthomosaic into 500 × 500 px tiles that cover all valid data, to compute the nDSM of each tile, and to decide which tiles contain structures taller than 2 m. Only the overview is read at once; everything else is read tile by tile.

**Step 4.1: Create the tile reader.** `TileReader` opens the orthomosaic, DSM and DEM lazily (so it also works after a fork, for example in DataLoader workers). The DSM and DEM are wrapped in `WarpedVRT` objects that present them on the orthomosaic's pixel grid with bilinear resampling. For a window, `read` returns:
- `valid`: `alpha > 0` of the orthomosaic,
- `ndsm`: `DSM − DEM` as `float32`, with `NaN` where either raster has no data, where the orthomosaic is invalid, or where the value falls outside `NDSM_RANGE`,
- `image` (optional): bands 1 to 3 as `uint8`.

**Step 4.2: Read a small overview.** The orthomosaic is read once at `1/F` of the full resolution (average resampling, all four bands). `valid_ov = ov[3] > 127` is the valid footprint. `fx`, `fy`, `yy` and `xx` convert between overview and full-resolution pixels.

**Step 4.3: Build the tile grid.** `starts()` generates tile start offsets along each axis with step `TILE − OVERLAP`. With `EDGE = "shift"`, the last start moves inward so the tile ends exactly at the image edge. For each (row, column) a `Window` is created, its footprint polygon is computed from the transform, and `valid_frac_ov` is estimated from the overview. With the defaults, the grid has **50 columns × 54 rows = 2,700 tiles** (derived from the raster size; actual: `____`). A tile has `has_data = True` when `valid_frac_ov > MIN_VALID`; tiles with data: `____`.

**Step 4.4: Build the footprint polygon and check the coverage.** `rasterio.features.shapes` vectorizes `valid_ov`, the pieces are merged and simplified into `footprint`. Two checks are printed:
- area of the raster bounding box not covered by the grid (should be about 0),
- number of valid overview pixels not covered by any tile with data (should be 0).

**Step 4.5: Plot the grid.** The whole grid is drawn in cyan over the orthomosaic footprint filled with gray (with the raster bounding box as a dashed white line), with Easting/Northing axes.

**Step 4.6: Per-tile nDSM statistics.** For every tile with data, one window is read (no RGB), and the structure mask is computed:

1. `ndsm > HEIGHT_THR` (NaN counts as background),
2. 3 × 3 binary opening (removes thin noise),
3. removal of connected regions smaller than `min_px` (`MIN_STRUCT_M2`).

Only a few numbers per tile are kept (`valid_frac`, `ndsm_cover`, `struct_px`, `struct_frac`, `ndsm_max`); the arrays are discarded. This is the slowest part of the stage, because it reads the DSM and DEM for each tile (time: `____`).

**Step 4.7: Group the tiles.**
- `group = "no data"` if `has_data` is false,
- `group = "structures"` if the tile has at least one structure pixel (`struct_px > 0`),
- `group = "ground only"` otherwise.

Adjacent structure tiles are linked into clusters: the tile grid is treated as a 2D boolean array of structure tiles and labeled with 8-connectivity (`scipy.ndimage.label`). `cluster = 0` for tiles that are not structure tiles. Counts: structure tiles `____`, ground-only tiles `____`, clusters `____`.

**Step 4.8: Inspect.** A map colors tiles by group (structure tiles red, ground-only blue, cluster outlines yellow). `show_tile(tile_id)` plots RGB, nDSM and the structure mask of one tile with Easting/Northing axes.

**Step 4.9: Add ownership columns and save the tile table.** `own_r0`, `own_r1`, `own_c0`, `own_c1` define the nominal grid cell of each tile (the last row and column extend to the image edge). The table is saved with `grid.to_parquet(OUT / "grid.parquet")` so a new Colab session can skip steps 4.1 to 4.8.

**Outputs of stage 4:** the `grid` table (in memory and as `grid.parquet`), the footprint polygon, the coverage check printout and the plots. No image data is written.

---

## 5. Object segmentation (SAM) and chip creation

The objects are the structures found in section 4, segmented with Meta AI's SAM ViT-B. SAM is prompted with the nDSM structures, so "object" means anything above `HEIGHT_THR`: roofs, but also trees, walls and vehicles. Each object gets one 224 × 224 chip.

**Step 5.1: Load SAM.** `sam_model_registry["vit_b"](checkpoint=CKPT)` is moved to the GPU and wrapped in `SamPredictor`.

**Step 5.2: Select the tiles to process.** Tiles with `group == "structures"` whose id is not listed in `done_tiles.txt`.

**Step 5.3: Define the processing window.** For each tile, the window is its owned cell expanded by `HALO` pixels on each side (clipped to the image). The halo keeps objects that cross a tile border whole. An object is kept only if its centroid falls inside the owned cell, so no object is created twice.

**Step 5.4: Read the window and build the structure mask.** RGB, validity and nDSM are read with `TileReader`; `clean_struct(ndsm > HEIGHT_THR)` gives the structure mask.

**Step 5.5: Find candidate regions (seeds).** `seed_regions`:
1. labels the connected components of the mask,
2. computes the distance transform of the mask,
3. if `SPLIT`, takes the h-maxima of the distance map (depth `H_M / res` px) as markers, so touching structures are separated at narrow necks without over-splitting long structures,
4. gives each component without a marker one marker at its distance-map maximum,
5. runs a watershed on the negative distance map, restricted to the mask, which yields one region per candidate object.

**Step 5.6: Build the prompts.** For each region: its bounding box padded by `PAD_PX`, plus a positive point at the region's maximum of the distance map (a point well inside the object).

**Step 5.7: Run SAM.** `predictor.set_image(window)` computes the image embedding once per window. The box and point prompts are sent in chunks of 32 with `multimask_output=False` (one mask per prompt).

**Step 5.8: Post-process the masks.**
1. Each SAM mask is restricted to the structure mask dilated by `DILATE_PX` pixels (SAM cannot leak onto the ground).
2. If a mask is smaller than `min_px`, or less than `MIN_INSIDE` of it lies on the structure mask, it is replaced by the nDSM region (`source = ndsm`); otherwise `source = sam`.
3. Near-duplicate masks (IoU above 0.6, computed on masks downsampled by 4) are removed, keeping the highest SAM score.

**Step 5.9: Keep the objects this tile owns.** The centroid is converted to full-resolution pixel coordinates; objects whose centroid lies outside the tile's owned cell are skipped (they belong to a neighboring cell), as are objects smaller than `min_px`.

**Step 5.10: Create the chip.**
1. The source window is a square centered on the object's centroid, with side `max(bbox height, bbox width) × (1 + 2 × MARGIN)` px, but not less than `MIN_SIDE`.
2. The window is cropped from the orthomosaic window (areas outside the image are filled black) and resized to `CHIP × CHIP` with Lanczos resampling.
3. If `BG_FADE > 0`, pixels outside the object mask are blended toward mid-gray by that fraction.
4. The chip is saved as `chips/t{tile_id:04d}_o{k:03d}.png`.
5. A manifest row is built (see section 3.6), including the centroid in map coordinates.

**Step 5.11: Write the manifest and mark the tile as done.** The tile's rows are appended to `chips.csv`, and only then is the tile id appended to `done_tiles.txt`. If the run is interrupted, unfinished tiles are processed again the next time.

**Step 5.12: Check the result.** The notebook prints the number of chips and tiles, the share of `sam` versus `ndsm` sources and summary statistics. It also shows a sample of 16 chips and a map of object centers over the footprint.

**Outputs of stage 5:** `chips/*.png`, `chips.csv`, `done_tiles.txt`. Number of chips: `____`; tiles processed: `____`; processing time: `____`.

---

## 6. Chip labeling tool

An `ipywidgets` interface that runs in the notebook (Colab or Jupyter).

**Interface**

| Element | Behavior |
|---|---|
| Image | The current chip, shown at 448 × 448 px (the file stays 224 × 224) |
| Info line | Chip id, position in the current list, area (m²), mean and maximum height (m), source (`sam`/`ndsm`) and the current label |
| Counter line | Number labeled out of the total, and the count per class |
| 5 label buttons | `light metal sheet`, `oxidized metal sheet`, `precarious/mixed`, `vegetation`, `other`. Clicking one saves the label and moves to the next chip |
| `Prev` / `Next (skip)` | Move backwards or forwards without labeling |
| `Clear label` | Removes the label of the current chip |
| `Order` dropdown | `as extracted`, `random`, or `unlabeled only` (the list of unlabeled chips at the moment of selection) |

There are no keyboard shortcuts; labeling is done by clicking.

**Behavior**
- Labels are held in the dictionary `labels` and written to `labels.csv` (columns `chip_id`, `label`) after every click, using a temporary file that then replaces the old one, so a disconnect loses at most the last click.
- At startup, existing `labels.csv` is loaded, so labeling can continue after a Colab restart.
- The manifest (`chips.csv`) is never modified by the labeling tool.

**Label definitions.** Fill in the criteria you use when labeling:

| Label | Definition / criteria used | Example chips |
|---|---|---|
| light metal sheet | ____ | ____ |
| oxidized metal sheet | ____ | ____ |
| precarious/mixed | ____ | ____ |
| vegetation | ____ | ____ |
| other | ____ | ____ |

Labeled chips: `____`; labeled by: `____`; labeling period: `____`.

---

## 7. Export for training

`export_dataset()` joins `chips.csv` with `labels.csv` (labeled chips only) and writes the files in `OUT/export/`.

1. `label_id` is the position of the label in `LABELS` (0 to 4) and `title` is the caption from `CAPTIONS` for that label.
2. `filepath` is made absolute (`OUT/...`). If you move the data, update the paths.
3. **Spatial split.** Chips are grouped into blocks of `block × block` tiles (default 4 × 4 tiles, about 108 m). Whole blocks are randomly assigned to test, validation or train (`seed`), so neighboring chips never end up in different splits.
4. One tab-separated file per split is written (`train.csv`, `val.csv`, `test.csv`) with the columns `filepath`, `title`, `label`, `label_id`, `chip_id`, `tile_id`, `center_e`, `center_n`, `area_m2`, plus `classes.json` with the list of labels. Tab separation and the `filepath`/`title` column names are the defaults of open_clip's CSV loader.

A small PyTorch dataset (`ChipDataset`) reads these CSV files; pass it the preprocessing transform returned by `open_clip.create_model_and_transforms(...)`. For the RemoteCLIP checkpoint and loading code, follow the RemoteCLIP repository: `____`. Model and checkpoint used: `____`.

Class distribution in the export (fill in after running):

| Split | light metal sheet | oxidized metal sheet | precarious/mixed | vegetation | other | Total |
|---|---|---|---|---|---|---|
| train | ____ | ____ | ____ | ____ | ____ | ____ |
| val | ____ | ____ | ____ | ____ | ____ | ____ |
| test | ____ | ____ | ____ | ____ | ____ | ____ |

---

## 8. Outputs of each process

| Process | Output | Location | Format | Contents |
|---|---|---|---|---|
| 4. Tile grid and nDSM statistics | `grid` (tile table) | memory, `OUT/grid.parquet` | GeoParquet | One row per tile: position, window, footprint, validity, nDSM structure statistics, `group`, `cluster`, ownership bounds (section 3.4) |
| 4. Tile grid | Footprint polygon, coverage printout, plots | memory / notebook | `footprint` (shapely polygon); figures | Grid in cyan over the gray footprint; group map |
| 4. Tile inspection | `show_tile`, `TileDataset` | notebook | figure / lazy dataset | RGB, nDSM and structure mask of a tile, read on demand |
| 5. SAM objects | Object masks | memory only (one tile at a time) | boolean arrays | Not stored; their statistics go into the manifest |
| 5. Chips | `chips/<chip_id>.png` | `OUT/chips/` | PNG, RGB, 224 × 224 | One image per object |
| 5. Manifest | `chips.csv` | `OUT/` | CSV (comma-separated) | One row per chip (section 3.6) |
| 5. Progress | `done_tiles.txt` | `OUT/` | text, one tile id per line | Tiles already processed |
| 6. Labeling | `labels.csv` | `OUT/` | CSV | `chip_id`, `label` |
| 7. Export | `train.csv`, `val.csv`, `test.csv` | `OUT/export/` | tab-separated | `filepath`, `title`, `label`, `label_id`, ids, coordinates, area |
| 7. Export | `classes.json` | `OUT/export/` | JSON | List of the five labels in `label_id` order |

Summary of the numbers of your run (fill in): tiles in grid `____`, tiles with data `____`, structure tiles `____`, clusters `____`, chips `____`, labeled chips `____`, train/val/test chips `____` / `____` / `____`.

---

## 9. Running and resuming

**Order of execution**

1. Mount Drive, install packages, copy the rasters to `/content`, set `dom_path`, `DSM_PATH`, `DEM_PATH`.
2. Define the parameters and `TileReader`; build the grid; check the coverage; plot the grid (section 4.1 to 4.5).
3. Run the nDSM statistics pass and the grouping (4.6 to 4.8); add the ownership columns and save `grid.parquet` (4.9).
4. Switch to a GPU runtime if needed; load SAM; run the chip creation loop (section 5).
5. Check the chips; run the labeling tool (section 6).
6. Run the export (section 7).

**Resuming after a Colab disconnect**

1. Remount Drive, reinstall the packages, copy the rasters again.
2. Rerun the cells that define the parameters, `TileReader`, `clean_struct` and the SAM functions.
3. Reload the tile table: `grid = gpd.read_parquet(OUT / "grid.parquet")`.
4. Rerun the chip creation loop; tiles already in `done_tiles.txt` are skipped.
5. Rerun the labeling tool cell; existing labels in `labels.csv` are reloaded.

If you change `BG_FADE`, `MARGIN`, `MIN_SIDE` or any SAM parameter and want to regenerate the chips, delete `done_tiles.txt`, `chips.csv` and the `chips/` folder first (and `labels.csv` if the old labels no longer apply).

Total runtime of each stage in your run: tiling `____`, nDSM pass `____`, SAM and chips `____`.

---

## 10. Known limitations

- **What counts as a structure:** everything above `HEIGHT_THR` (2 m), including trees, walls and tall vehicles. Roofs lower than 2 m (sheds, lean-tos) are not detected. The label classes `vegetation` and `other` absorb the non-roof objects.
- **Merged roofs:** adjacent roofs that touch above 2 m form one nDSM object. The watershed split (`SPLIT`, `H_M`) helps for narrow necks but not for long rows of houses of equal height. A high share of `source = ndsm` in the manifest indicates SAM disagreed with the nDSM objects.
- **Context in the chips:** with `BG_FADE = 0` the chip includes neighboring roofs, which can influence material labels.
- **Variable scale:** every chip is resized from a window of a different size, so the effective pixel size differs (`chip_gsd_cm`). Large objects are downsampled and small objects are upsampled.
- **Validity estimate:** `valid_frac_ov` comes from an overview, so it is approximate; `valid_frac` (exact) is only computed for tiles with data.
- **Overlap strips:** with `EDGE = "shift"`, the last tile in each row/column overlaps its neighbor. Per-tile statistics (such as `struct_px`) count pixels in the overlap twice when summed over tiles.
- **nDSM quality:** photogrammetric DSMs smooth building edges, and the DEM may be interpolated under buildings. Check `ndsm_cover` and inspect tiles with `show_tile`.
- **Objects at the image border:** chips can include black padding where the source window extends beyond the orthomosaic.
- **Accuracy of the labels:** `____` (annotator agreement, review process, etc.).

---

## 11. Troubleshooting

| Error | Cause and fix |
|---|---|
| `AssertionError: Torch not compiled with CUDA enabled`, or `GPU: False` | The runtime has no GPU, or torch is a CPU build. Runtime → Change runtime type → GPU, then rerun the setup cells. If `torch.version.cuda` is `None`, reinstall a CUDA build of torch |
| `AttributeError: 'numpy.ndarray' object has no attribute 'ptp'` | NumPy 2 removed `ndarray.ptp`. Use `np.ptp(array)` (already done in the final code) |
| `FileNotFoundError` for the checkpoint or rasters | Drive not mounted, or a wrong path. Check with `!ls` |
| `ModuleNotFoundError: segment_anything` | Colab installs are lost on restart; rerun the `pip install` cell |
| `CUDA out of memory` | Restart the session; reduce `chunk` in `find_objects` |
| Reading the rasters is very slow | The files are read over the Drive mount. Copy them to `/content` |
| Duplicate rows in `chips.csv` | A tile was interrupted mid-write. Load with `drop_duplicates("chip_id", keep="last")` (already done in the labeling tool) |
| Other | `____` |

---

## 12. References, license and contact

- Kirillov et al., *Segment Anything* (Meta AI, 2023). Code: https://github.com/facebookresearch/segment-anything
- RemoteCLIP: `____` (paper and checkpoint reference)
- open_clip: `____`
- Data source and acknowledgements: `____`

**Authors:** `____`
**Contact:** `____`
**License (code):** `____`
**License / terms of the data:** `____`
**How to cite this work:** `____`
