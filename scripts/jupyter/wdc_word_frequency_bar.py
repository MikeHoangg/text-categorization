import matplotlib.pyplot as plt
import pandas as pd

from main import PLOT_FIG_SIZE


def show_frequency_bar(data: pd.DataFrame, page: int = 1, items: int = 10, bar_width: float = 0.4) -> pd.DataFrame:
    if page < 1:
        raise Exception('page must be >= 1')

    chunk = data.iloc[(page - 1) * items:page * items]

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
    ax.bar(list(chunk['word']), list(chunk['count']), width=bar_width)
    ax.set_title(f'Frequency table')
    plt.show()

    return chunk
