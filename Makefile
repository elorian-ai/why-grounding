# make tables: Tables 1-4 and 6 and Figure 3 from $(GR_RESULTS) and BrickVision; make paper: also the figures that
# draw dataset images from $(GR_DATA). Output in $(GR_RESULTS)/paper (CPU, minutes).
GR_DATA ?= data
GR_RESULTS ?= results
export GR_DATA GR_RESULTS
PYTHON ?= uv run --all-extras python
FIGURES = teaser synth_tasks brickvision_examples brickvision_construction format_ablation vgr probe_sft
PROBES = wire_tracing prompted_counting arrow_following

.PHONY: tables paper
tables:
	mkdir -p $(GR_RESULTS)/paper
	if [ -d $(GR_RESULTS)/decode_randomization/rollouts ]; then $(PYTHON) -m grounded_reasoning.randomize report; fi
	cp $(GR_RESULTS)/decode_randomization/report/table_decode_randomization.tex $(GR_RESULTS)/paper/  # Table 1
	$(PYTHON) paper/tables.py                                                    # Tables 2, 3, 6
	$(PYTHON) -m brickvision.evaluate tables > $(GR_RESULTS)/paper/table_brickvision.md  # Table 4
	for t in $(PROBES); do $(PYTHON) -m grounded_reasoning.probes.$$t summarize > /dev/null || exit 1; done
	$(PYTHON) paper/figures/train_rand.py                                        # Figure 3

paper: tables
	for f in $(FIGURES); do $(PYTHON) paper/figures/$$f.py || exit 1; done
