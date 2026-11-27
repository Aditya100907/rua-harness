# Build the paper. Usage: make (pdf), make tables, make plots, make clean.
# Uses latexmk if present, else pdflatex + bibtex, else tectonic.
PYTHON ?= python
MAIN   := main

LATEXMK  := $(shell command -v latexmk 2>/dev/null)
PDFLATEX := $(shell command -v pdflatex 2>/dev/null)
BIBTEX   := $(shell command -v bibtex 2>/dev/null)
TECTONIC := $(shell command -v tectonic 2>/dev/null)

.PHONY: all pdf tables plots clean

all: pdf

tables:
	$(PYTHON) ../scripts/make_tables.py

plots:
	$(PYTHON) ../scripts/make_plots.py

pdf: tables
ifneq ($(LATEXMK),)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex
else ifneq ($(and $(PDFLATEX),$(BIBTEX)),)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
else ifneq ($(TECTONIC),)
	tectonic --keep-logs $(MAIN).tex
else
	@echo "ERROR: no TeX engine found (latexmk, pdflatex+bibtex, or tectonic)."
	@echo "Install one, e.g. 'brew install tectonic' on a Mac. See paper/README.md."
	@exit 1
endif

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,fls,fdb_latexmk,synctex.gz} $(MAIN).pdf
