PYTHON ?= python3

.PHONY: assets

# Regenerate every paper figure and table from the aggregate CSVs in data/ and
# ../reports/robustness/w1_axis/. No experiment is rerun. The first three
# builders write intermediate row tables in tables/; build_v4_assets.py applies
# the paper's display names and writes figures/v4/ and tables/v4/;
# build_v5_figures.py draws the explanatory Figures 1-3 in figures/v5/.
assets:
	$(PYTHON) build_assets.py
	$(PYTHON) build_profiles.py
	$(PYTHON) build_w1_robustness.py
	$(PYTHON) build_v4_assets.py
	$(PYTHON) build_v5_figures.py
