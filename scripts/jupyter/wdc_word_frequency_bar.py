import matplotlib.pyplot as plt
import pandas as pd

from main import PLOT_FIG_SIZE


def show_bar(data: pd.DataFrame, page: int = 1, items: int = 10, bar_width: float = 0.4) -> pd.DataFrame:
    if page < 1:
        raise Exception('page must be >= 1')
    chunk = data.iloc[(page - 1) * items:page * items]
    plt.subplots(figsize=PLOT_FIG_SIZE)
    plt.bar(list(chunk['word']), list(chunk['count']), width=bar_width)
    plt.show()

    return chunk
