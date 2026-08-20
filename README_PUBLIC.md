# Personal Cinema Archive — public release

This is a personal movie-matching tool built from 148 viewing records. Visitors submit a plot summary, themes, or viewing intent; the app compares it with the author's highly rated semantic archive and returns a match level with related titles.

## Privacy boundary

The publishable version contains only:

- Film titles and ratings
- The normalized vector artifact: `model/embeddings.npy`
- The recommendation app and its dependencies

It does **not** include raw reviews, cleaned review text, viewing dates, film metadata, or the local model cache.

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app\app.py
```

On a fresh machine, the first request downloads the public Sentence-BERT model. Later requests reuse the cache.

## Deploy to Streamlit Community Cloud

1. Create a new public GitHub repository and commit only files not ignored by `.gitignore`.
2. Go to `https://share.streamlit.io`, sign in, and connect GitHub.
3. Select the repository, set the entrypoint to `app/app.py`, and choose Python 3.12.
4. Select **Deploy** and wait for the initial model download.

The deployed project will receive a shareable `*.streamlit.app` URL.
