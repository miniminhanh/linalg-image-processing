# linalg-image-processing

## Filter UI

Upload your own image and apply any of the filters in this repo from your browser:

```
python app.py
```

Then open http://localhost:8000. The filters themselves live in `filters.py`; add a function there and register it in `FILTERS` to make it show up in the UI.

### Deploying to a server

Install the server packages with `pip install -r requirements.txt` (headless OpenCV, no desktop windows), then use `deploy/image-filter.service` to run `app.py` as a systemd service. Instructions are in the comments at the top of that file. Settings such as `HOST`, `PORT` and `MAX_SIDE` are described at the top of `app.py`.
