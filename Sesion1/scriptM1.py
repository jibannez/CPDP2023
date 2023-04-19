# Script con distintos ejemplos de librerías para computación científica en python
# adaptados de los ejemplos oficiales de cada proyecto para mostrar sus aplicaciones.
# Jorge Ibáñez

def run_interpolation_example():
    # Author: Mathieu Blondel
    #         Jake Vanderplas
    #         Christian Lorentzen
    #         Malte Londschien
    # License: BSD 3 clause
    
    #ADAPTED FROM:
    #https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
    from sklearn.pipeline import make_pipeline


    # %%
    # We start by defining a function that we intend to approximate and prepare
    # plotting it.


    def f(x):
        """Function to be approximated by polynomial interpolation."""
        return x * np.sin(x)

    # whole range we want to plot
    x_plot = np.linspace(-1, 11, 100)

    # %%
    # To make it interesting, we only give a small subset of points to train on.

    x_train = np.linspace(0, 10, 100)
    rng = np.random.RandomState(0)
    x_train = np.sort(rng.choice(x_train, size=20, replace=False))
    y_train = f(x_train)

    # create 2D-array versions of these arrays to feed to transformers
    X_train = x_train[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    # %%
    # Now we are ready to create polynomial features and splines, fit on the
    # training points and show how well they interpolate.

    # plot function
    lw = 2
    fig, ax = plt.subplots()
    ax.set_prop_cycle(
        color=["black", "teal", "yellowgreen", "gold", "darkorange", "tomato"]
    )
    ax.plot(x_plot, f(x_plot), linewidth=lw, label="Curva real")

    # plot training points
    ax.scatter(x_train, y_train, label="Datos observados")

    # polynomial features
    for degree in [3, 4, 5]:
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
        model.fit(X_train, y_train)
        y_plot = model.predict(X_plot)
        ax.plot(x_plot, y_plot, label=f"Interpolación de grado {degree}")


    ax.legend(loc="lower center")
    ax.set_ylim(-20, 10)
    plt.show()

    


def run_faces_example():
    """
    ============================
    Faces dataset decompositions
    ============================

    This example applies to :ref:`olivetti_faces_dataset` different unsupervised
    matrix decomposition (dimension reduction) methods from the module
    :py:mod:`sklearn.decomposition` (see the documentation chapter
    :ref:`decompositions`).


    - Authors: Vlad Niculae, Alexandre Gramfort
    - License: BSD 3 clause
    
    ADAPTED FROM:
    https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
    """

    # %%
    # Dataset preparation
    # -------------------
    #
    # Loading and preprocessing the Olivetti faces dataset.

    import logging

    from numpy.random import RandomState
    import matplotlib.pyplot as plt

    from sklearn.datasets import fetch_olivetti_faces
    from sklearn import cluster
    from sklearn import decomposition

    rng = RandomState(0)

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    n_samples, n_features = faces.shape

    # Global centering (focus on one feature, centering all samples)
    faces_centered = faces - faces.mean(axis=0)

    # Local centering (focus on one sample, centering all features)
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

    print("Dataset consists of %d faces" % n_samples)

    # %%
    # Define a base function to plot the gallery of faces.

    n_row, n_col = 2, 3
    n_components = n_row * n_col
    image_shape = (64, 64)


    def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
        fig, axs = plt.subplots(
            nrows=n_row,
            ncols=n_col,
            figsize=(2.0 * n_col, 2.3 * n_row),
            facecolor="white",
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
        fig.set_edgecolor("black")
        fig.suptitle(title, size=16)
        for ax, vec in zip(axs.flat, images):
            vmax = max(vec.max(), -vec.min())
            im = ax.imshow(
                vec.reshape(image_shape),
                cmap=cmap,
                interpolation="nearest",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.axis("off")

        fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
        plt.show()
    
    
    # %%
    # Let's take a look at our data. Gray color indicates negative values,
    # white indicates positive values.

    plot_gallery("Faces from dataset", faces_centered[:n_components])

    # %%
    # Decomposition
    # -------------
    #
    # Initialise different estimators for decomposition and fit each
    # of them on all images and plot some results. Each estimator extracts
    # 6 components as vectors :math:`h \in \mathbb{R}^{4096}`.
    # We just displayed these vectors in human-friendly visualisation as 64x64 pixel images.
    #
    # Read more in the :ref:`User Guide <decompositions>`.

    # %%
    # Eigenfaces - PCA using randomized SVD
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Linear dimensionality reduction using Singular Value Decomposition (SVD) of the data
    # to project it to a lower dimensional space.
    #
    #
    # .. note::
    #
    #     The Eigenfaces estimator, via the :py:mod:`sklearn.decomposition.PCA`,
    #     also provides a scalar `noise_variance_` (the mean of pixelwise variance)
    #     that cannot be displayed as an image.

    # %%
    pca_estimator = decomposition.PCA(
        n_components=n_components, svd_solver="randomized", whiten=True
    )
    pca_estimator.fit(faces_centered)
    plot_gallery(
        "Eigenfaces - PCA using randomized SVD", pca_estimator.components_[:n_components]
    )

    # %%
    # Non-negative components - NMF
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # Estimate non-negative original data as production of two non-negative matrices.

    # %%
    nmf_estimator = decomposition.NMF(n_components=n_components, tol=5e-3)
    nmf_estimator.fit(faces)  # original non- negative dataset
    plot_gallery("Non-negative components - NMF", nmf_estimator.components_[:n_components])

    
def run_fourier_example():
    from scipy.fft import fft, fftfreq
    from scipy.signal import blackman
    import numpy as np
    import matplotlib.pyplot as plt

    # Number of sample points
    N = 600

    # sample spacing
    T = 1.0 / 800.0

    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = fft(y)
    w = blackman(N)
    ywf = fft(y*w)
    xf = fftfreq(N, T)[:N//2]

    plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')
    plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
    plt.legend(['FFT', 'FFT w. window'])
    plt.grid()
    plt.show()    