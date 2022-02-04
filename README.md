This package requires jax, tensorflow, and numpy. Either tensorflow or
scikit-learn can be used for loading data.

To run in a nix-shell with required packages (at specific versions used

``` bash
nix-shell
```

Results are generated from main.py, running with arguments required, e.g.
`python main.py --lr <learning rate> --width <width>`. The results as described
in the paper are in csv files in the `results` subfolder.

Figures in the paper can be reproduced by running `analysis.py`. To generate
plots with the bounds and errors using the same scale (as described in the
appendix), set the variable `BOUND_SCALE_AXIS` in this file to False.
