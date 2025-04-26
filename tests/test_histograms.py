import numpy as np
import pandas as pd

from tableone import TableOne
from tableone.formatting import generate_histograms


def test_generate_histograms_simple():
    # Simple case: clean data
    x = np.linspace(0, 10, 100)
    hist = generate_histograms(x)
    assert isinstance(hist, str)
    assert all(c in '▁▂▃▄▅▆▇█' for c in hist)


def test_generate_histograms_empty_array():
    # Edge case: empty array
    x = np.array([])
    hist = generate_histograms(x)
    assert isinstance(hist, str)
    assert hist == ''


def test_clip_histogram_behavior():

    # Create toy data: mostly normal values, plus strong outliers
    rng = np.random.default_rng(seed=42)
    normal_data = rng.normal(loc=50, scale=5, size=95)
    outliers = np.array([300, 400, 500, 600, 1000])  # Big outliers
    all_data = np.concatenate([normal_data, outliers])

    df = pd.DataFrame({
        'group': ['A'] * 50 + ['B'] * 50,
        'value': all_data
    })

    # No clipping
    t1_noclip = TableOne(df, columns=['value'], groupby='group', continuous=['value'],
                         show_histograms=True, clip_histograms=None)

    # With clipping
    t1_clip = TableOne(df, columns=['value'], groupby='group', continuous=['value'],
                       show_histograms=True, clip_histograms=(5, 95))

    # Find the index for the summary row
    main_row_idx = None
    for idx in t1_noclip.tableone.index:
        if idx[0].startswith('value') and idx[1] == '':
            main_row_idx = idx
            break

    assert main_row_idx is not None, "Could not find main summary row for 'value'."

    # Extract histograms
    no_clip_hist = t1_noclip.tableone.loc[main_row_idx, ('Grouped by group', 'Overall Histogram')]
    clip_hist = t1_clip.tableone.loc[main_row_idx, ('Grouped by group', 'Overall Histogram')]

    # They should be different
    assert no_clip_hist != clip_hist

    # Histograms should not be empty
    assert isinstance(no_clip_hist, str) and len(no_clip_hist) > 0
    assert isinstance(clip_hist, str) and len(clip_hist) > 0


def test_histogram_unicode_characters_only():
    # Check that only expected unicode block characters are used
    data = np.random.randn(100)
    hist = generate_histograms(data)
    block_chars = set('▁▂▃▄▅▆▇█')
    assert set(hist).issubset(block_chars)
