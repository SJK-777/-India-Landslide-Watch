# India Landslide Watch — multi-location dashboard

1. Add `app.py`, `requirements.txt`, and `INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv` to the GitHub repository root.
2. Deploy/redeploy on Streamlit Community Cloud with `app.py` as the main file.
3. Optional site CSV columns: `site`, `latitude`, `longitude`, plus predictor columns matching the training CSV names.
4. Use the sidebar's Refresh forecast data button to clear cached data and request forecasts again.

## Important
This is a research decision-support dashboard, not an authorized warning system. It does not issue evacuation orders. The classifier is refit from the supplied training CSV; it does not load the prior exact fitted research model. Coordinates do not automatically fetch terrain/land-cover rasters. If site predictor fields are absent, training medians/modes are substituted and the app labels that fact. Rainfall comes from a forecast API, not rain-gauge observations, and the displayed rainfall bands are provisional and unvalidated. Do not use alone for life-safety decisions.
