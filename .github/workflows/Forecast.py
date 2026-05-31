name: GEM Forecast Maps

on:
  # Run daily at 14:00 UTC (after 12Z model run is available)
  schedule:
    - cron: '0 14 * * *'
  # Allow manual trigger from Actions tab
  workflow_dispatch:
    inputs:
      forecast_days:
        description: 'Forecast days (GDPS, default 6)'
        required: false
        default: '6'

jobs:
  generate-maps:
    runs-on: ubuntu-latest
    timeout-minutes: 90

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      # ── System deps (eccodes for GRIB2) ─────────────────────────────────
      - name: Install system dependencies
        run: |
          sudo apt-get update -qq
          sudo apt-get install -y -q libeccodes-dev libeccodes-tools

      # ── Python ──────────────────────────────────────────────────────────
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install Python dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # ── Run ─────────────────────────────────────────────────────────────
      - name: Generate forecast maps
        env:
          OUTPUT_DIR: output
          GDPS_FORECAST_DAYS: ${{ github.event.inputs.forecast_days || '6' }}
        run: |
          mkdir -p output
          python forecast_gem.py

      # ── Upload HTML outputs as artifact ─────────────────────────────────
      - name: Upload maps
        uses: actions/upload-artifact@v4
        with:
          name: forecast-maps-${{ github.run_number }}
          path: output/
          retention-days: 7

      # ── Optional: deploy to GitHub Pages ────────────────────────────────
      # Uncomment the block below and enable Pages (Settings → Pages → GitHub Actions)
      # to publish the maps at https://<username>.github.io/<repo>/
      #
      # - name: Deploy to GitHub Pages
      #   uses: peaceiris/actions-gh-pages@v3
      #   if: github.ref == 'refs/heads/main'
      #   with:
      #     github_token: ${{ secrets.GITHUB_TOKEN }}
      #     publish_dir: ./output
      #     destination_dir: latest
