# Hispanic Heritage App 2.0

## Run locally
Use Python 3.10+.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## GitHub + Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Open Streamlit Community Cloud and connect the repository.
4. Select `app.py` as the main file and deploy.
5. After deployment, test the historical API fetch, custom functions, calculations, and downloads.

## Render
Create a **Python Web Service** connected to the GitHub repository.

Build command:
```bash
pip install -r requirements.txt
```

Start command (one line):
```bash
python -m python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Verification checklist
- Try `x**2 - 4*x + 3`, `100*exp(-0.18*x)`, `12 + 8*sin(2*pi*x/24)`, and `(x+1)/(x-2)`.
- Test malformed input, `log(x)` for negative x, `sqrt(x)` for negative x, and rational discontinuities.
- Change the current point and interval endpoints; confirm displayed values, tangent, secant, and simulation indicator update together.
- Change course mode and show/hide each graph curve.
- Fetch historical data, fit different polynomial degrees, inspect residuals and in-sample metrics, and compare a chronological holdout score when enough data exist.
- Try a degree that exceeds the number of distinct years, add duplicate years in the editor, and check validation messages.
- Check single-year and batch estimates agree for the same year. Test year bounds 1950 and 2100, invalid custom years, and range steps 1 and 10.
- Edit the working data, apply it, and reset to the downloaded observations.
- Download and open CSV, Excel, and PDF; verify they match the current model and calculated results.
- Test on the deployed app and review platform logs if API requests or package installs fail.

## Important limitations
This is an educational prototype. Polynomial extrapolation is not clipped and may be implausible. The app does not provide validated 95% prediction intervals. Numerical calculus root-finding is approximate and can miss difficult cases. Medical, weather, economic, and other context examples are not professional advice or validated forecasts.
