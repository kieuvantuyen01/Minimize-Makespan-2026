# Build standalone figure PDFs for the SALBP-PPC manuscript.
# Run from the figures/ directory:  cd figures && make

# The RQ1 and cactus plots are generated as inline TikZ/PGFPlots code from
# normalized CSV data by analysis/analyze_manuscript_results.py. Historical
# standalone cactus files are archived under ../../tmp/legacy_cactus_figures/
# and are not built here because they can become stale after data exclusions.
FIGS = fig_framework fig_example
PDFS = $(addsuffix .pdf, $(FIGS))
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean

all: $(PDFS)

# Prefer latexmk if available, fall back to two pdflatex passes
%.pdf: %.tex
	@if command -v latexmk >/dev/null 2>&1; then \
	  $(LATEXMK) $<; \
	else \
	  $(PDFLATEX) $< && $(PDFLATEX) $<; \
	fi

clean:
	rm -f *.aux *.log *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PDFS)
