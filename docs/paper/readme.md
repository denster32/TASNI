# TASNI Paper Draft

## Files

- `draft.md` - Full paper outline (for arXiv submission)
- `paper.tex` - AASTeX LaTeX template (in progress)
- `preprint.md` - Markdown version for arXiv
- `figures/` - Place for publication figures

## Paper Outline (draft.md Sections)

1. **Abstract** - 200-250 words, motivation/methods/results/implications
2. **Introduction** - Silent civilization problem, detection gap, our approach
3. **Methodology** - Data sources, cross-match pipeline, anomaly scoring, validation
4. **Results** - Filtering statistics, golden target properties, UMAP clustering
5. **Discussion** - Natural explanations, Warm Dark hypothesis, limitations, future work
6. **Conclusion** - Summary, reproducibility statement
7. **Acknowledgments** - Data sources, archives
8. **References** - Citations

## Building

### Convert Markdown to LaTeX (for arXiv)

```bash
pip install pandoc
pandoc docs/paper/draft.md -o docs/paper/manuscript.tex
```

### Convert to HTML (for review)

```bash
pandoc docs/paper/draft.md -o docs/paper/manuscript.html
```

## Submission Checklist

- [ ] Fill in actual numbers from pipeline results
- [ ] Add actual references (ADS BibTeX)
- [ ] Replace [REPOSITORY] placeholder with GitHub URL
- [ ] Verify all figures are 300+ DPI
- [ ] Check figure captions reference correctly
- [ ] Anonymize for submission (if required)
- [ ] Generate PDF via LaTeX
- [ ] Upload to arXiv

## Existing Resources

See also: `paper.tex` (AASTeX format) and `preprint.md` (alternate version) in this directory.

For full writing guidelines, see internal documentation above.
