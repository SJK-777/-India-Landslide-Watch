# Simple India Landslide Watch UI

Upload `app.py`, `requirements.txt`, and `INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv` to the GitHub repo root and redeploy Streamlit Community Cloud.

The visible UI contains only location inputs, environmental predictor inputs, susceptibility output, rainfall forecast and plain-language interpretation. Dataset preview and dataset upload controls are removed from the interface. The training CSV must still be present beside app.py so the app can train its model.

Important: this is not an authorized public warning system. Coordinate inputs do not automatically fetch local raster predictors. Default predictor values are training medians/modes. Rainfall is forecast data, not rain-gauge observations, and screening thresholds are not validated for evacuation decisions.
