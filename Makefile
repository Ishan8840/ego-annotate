.DEFAULT_GOAL := help

# Full build. Every target writes into artifacts/ and reads the previous stage.
PY ?= python3
EGO = $(PY) -m egoannot

.PHONY: all test selftest quality events features segments spans caption score \
        gold clean-artifacts paths

all: quality events features spans caption score

paths:      ; $(EGO) paths
test:       ; $(PY) -m pytest tests -q
selftest:   ; $(EGO) quality selftest && $(EGO) lint

quality:    ; $(EGO) quality measure
events:     ; $(EGO) events measure
features:   ; $(EGO) features fit && $(EGO) features build
segments:   ; $(EGO) segments render
spans:      ; $(EGO) spans build
caption:    ; $(EGO) caption run --backend $(BACKEND)
score:      ; $(EGO) score
gold:       ; $(EGO) gold

BACKEND ?= stub

clean-artifacts:
	rm -rf artifacts/quality artifacts/events artifacts/spans artifacts/captions

.PHONY: help doctor
help:
	@echo "Video workflows: $(PY) ego.py --help"
	@echo "Setup check:     make doctor PY=.venv/bin/python"
	@echo "Tests:           make test PY=.venv/bin/python"
	@echo "Legacy corpus:   make all (see docs/pipeline.md)"

doctor:
	$(PY) ego.py doctor
