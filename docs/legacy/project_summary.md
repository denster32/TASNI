# TASNI Project Summary

## What Was Done

### 1. Script Cleanup
- Production scripts: 18 (kept for production pipeline)
- Archived scripts: 25 (moved to src/tasni/legacy/)
- Removed duplicates: filter_anomalies.py, crossmatch.py, download_wise.py, etc.

### 2. Testing Infrastructure
- 9 tests passing covering:
  - Crossmatch functions
  - Filter quality functions
  - Thermal profile computation
  - Weirdness scoring
  - Integration tests (golden targets, tier5, tier4)
  - Schema validation

### 3. Documentation
- SCHEMA.md - Complete schema for all output files
- README.md - Quick start guide for output directory
- manifest.json - Pipeline parameters for reproducibility
- PIPELINE_STATUS.md - Current state of the pipeline

### 4. Pipeline Status
| Phase | Sources | Status |
|-------|---------|--------|
| WISE Download | 747M | Complete |
| Gaia Download | 28GB | Complete |
| Crossmatch | 406M orphans | Complete |
| Quality Filter | 2.37M | Complete |
| Multi-Wavelength Veto | 39,188 | Complete |
| Radio Filter | ~4K | Complete |
| Golden Targets | 100 | Complete |

### 5. Output Files
- golden_targets.csv - 100 ranked candidates
- tier5_radio_silent.parquet - Radio quiet sources
- tier4_final.parquet - Final candidate catalog
- image_clusters.csv - ML clustering results

## How to Use

### Run Tests


### View Results
Version: ImageMagick 7.1.2-9 Q16-HDRI aarch64 23451 https://imagemagick.org
Copyright: (C) 1999 ImageMagick Studio LLC
License: https://imagemagick.org/script/license.php
Features: Cipher DPC HDRI Modules OpenMP
Delegates (built-in): bzlib fontconfig freetype heic jng jp2 jpeg jxl lcms lqr ltdl lzma openexr png raw tiff uhdr webp xml zip zlib zstd
Compiler: clang (17.0.0)
Usage: import [options ...] [ file ]

Image Settings:
  -adjoin              join images into a single multi-image file
  -border              include window border in the output image
  -channel type        apply option to select image channels
  -colorspace type     alternate image colorspace
  -comment string      annotate image with comment
  -compress type       type of pixel compression when writing the image
  -define format:option
                       define one or more image format options
  -density geometry    horizontal and vertical density of the image
  -depth value         image depth
  -descend             obtain image by descending window hierarchy
  -display server      X server to contact
  -dispose method      layer disposal method
  -dither method       apply error diffusion to image
  -delay value         display the next image after pausing
  -encipher filename   convert plain pixels to cipher pixels
  -endian type         endianness (MSB or LSB) of the image
  -encoding type       text encoding type
  -filter type         use this filter when resizing an image
  -format "string"     output formatted image characteristics
  -frame               include window manager frame
  -gravity direction   which direction to gravitate towards
  -identify            identify the format and characteristics of the image
  -interlace type      None, Line, Plane, or Partition
  -interpolate method  pixel color interpolation method
  -label string        assign a label to an image
  -limit type value    Area, Disk, Map, or Memory resource limit
  -monitor             monitor progress
  -page geometry       size and location of an image canvas
  -pause seconds       seconds delay between snapshots
  -pointsize value     font point size
  -quality value       JPEG/MIFF/PNG compression level
  -quiet               suppress all warning messages
  -regard-warnings     pay attention to warning messages
  -repage geometry     size and location of an image canvas
  -respect-parentheses settings remain in effect until parenthesis boundary
  -sampling-factor geometry
                       horizontal and vertical sampling factor
  -scene value         image scene number
  -screen              select image from root window
  -seed value          seed a new sequence of pseudo-random numbers
  -set property value  set an image property
  -silent              operate silently, i.e. don't ring any bells
  -snaps value         number of screen snapshots
  -support factor      resize support: > 1.0 is blurry, < 1.0 is sharp
  -synchronize         synchronize image to storage device
  -taint               declare the image as modified
  -transparent-color color
                       transparent color
  -treedepth value     color tree depth
  -verbose             print detailed information about the image
  -virtual-pixel method
                       Constant, Edge, Mirror, or Tile
  -window id           select window with this id or name
                       root selects whole screen

Image Operators:
  -annotate geometry text
                       annotate the image with text
  -colors value        preferred number of colors in the image
  -crop geometry       preferred size and location of the cropped image
  -encipher filename   convert plain pixels to cipher pixels
  -extent geometry     set the image size
  -geometry geometry   preferred size or location of the image
  -help                print program options
  -monochrome          transform image to black and white
  -negate              replace every pixel with its complementary color
  -quantize colorspace reduce colors in this colorspace
  -resize geometry     resize the image
  -rotate degrees      apply Paeth rotation to the image
  -strip               strip image of all profiles and comments
  -thumbnail geometry  create a thumbnail of the image
  -transparent color   make this color transparent within the image
  -trim                trim image edges
  -type type           image type

Miscellaneous Options:
  -debug events        display copious debugging information
  -help                print program options
  -list type           print a list of supported option arguments
  -log format          format of debugging information
  -version             print version information

By default, 'file' is written in the MIFF image format.  To
specify a particular image format, precede the filename with an image
format name and a colon (i.e. ps:image) or specify the image type as
the filename suffix (i.e. image.ps).  Specify 'file' as '-' for
standard input or output.

### Re-run Pipeline


## Key Files

| File | Purpose |
|------|---------|
| src/tasni/run_pipeline.sh | Pipeline runner |
| src/tasni/config.py | Configuration |
| src/tasni/filter_anomalies_full.py | Filtering |
| src/tasni/generate_golden_list.py | Golden targets |
| output/SCHEMA.md | Output documentation |
| tests/ | Test suite |

## Next Steps

1. Publication - Write paper on golden targets
2. Follow-up - GBT observations
3. Community - Share and get feedback
4. Phase 5 - Distributed network (future)

---

The forest was never dark.
