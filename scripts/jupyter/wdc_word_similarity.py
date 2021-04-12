import matplotlib.pyplot as plt
import pandas as pd

from main import PLOT_FIG_SIZE


def show_similarity_bar(data: pd.DataFrame, word: str, page: int = 1, items: int = 10,
                        bar_width: float = 0.4) -> pd.DataFrame:
    if page < 1:
        raise Exception('page must be >= 1')

    data = data[(data['word1'] == word) | (data['word2'] == word)]
    chunk = data.iloc[(page - 1) * items:page * items]

    res_words = list(
        chunk[['word1', 'word2']].apply(
            lambda x: next(s for s in x.astype(str) if s != word),
            axis=1)
    )
    res_percents = list(chunk['percent'])

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
    ax.bar(res_words, res_percents, width=bar_width)
    ax.set_title(f'Similarity table for word - {word}')
    plt.show()

    return pd.DataFrame(zip(res_words, res_percents), columns=['word', 'percent'])
