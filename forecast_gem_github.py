name: Generate Forecast Maps

on:
  schedule:
    - cron: '0 14 * * *'   # 14Z daily, after 12Z model data arrives
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install system dependencies
        run: |
          sudo apt-get update -qq
          sudo apt-get install -y -q libeccodes-dev

      - name: Install Python dependencies
        run: |
          pip install \
            requests \
            scipy \
            matplotlib \
            numpy \
            pykrige \
            folium \
            shapely \
            branca \
            pandas \
            cfgrib \
            eccodes \
            xarray \
            metpy

      - name: Run forecast script
        run: python forecast_gem_github.py

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./output
          publish_branch: gh-pages
          keep_files: true
