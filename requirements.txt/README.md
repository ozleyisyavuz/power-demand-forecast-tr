# Power Demand Forecast TR

Electricity demand forecasting baseline (time-series) + FastAPI inference service.

## What it does
- Generates (demo) hourly demand dataset
- Trains a baseline model and reports MAPE
- Serves predictions via FastAPI `/predict`
- Includes tests + GitHub Actions CI

## Roadmap
- Replace synthetic data with real load data (TEIAS/ENTSO-E)
- Add drift monitoring and dashboards
