# Running the notebooks in uv

To run the notebooks in uv, use the following command:
```bash
uv run --extra=nb python -m ipykernel install --user --name alpasim --display-name "Python (alpasim)"
uv run --extra=nb jupyter notebook notebooks/<notebook>.ipynb
```
