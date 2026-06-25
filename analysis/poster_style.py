"""Shared poster plot style: Lato (the Open Athena blog font) + cream block-colored background.

Lato TTFs ship with texlive; we register them with matplotlib so the plots match the poster text.
The figure/axes background is set to the block-body cream so plots blend into the poster blocks."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

_LATO_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/truetype/typoland/lato"
for _f in ("Lato-Regular.ttf", "Lato-Bold.ttf", "Lato-Italic.ttf"):
    fm.fontManager.addfont(f"{_LATO_DIR}/{_f}")

BOX = "#F2EDE6"  # poster block-body cream


def apply():
    plt.rcParams.update({
        "font.family": "Lato",
        "mathtext.fontset": "dejavusans",
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.facecolor": BOX,
        "axes.facecolor": BOX,
        "savefig.facecolor": BOX,
        "savefig.edgecolor": BOX,
    })
