import warnings

import numpy as np
import pandas as pd


def docstring_copier(*sub):
    """
    Wrap the TableOne docstring (not ideal :/)
    """
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


def set_display_options(max_rows=None,
                        max_columns=None,
                        width=None,
                        max_colwidth=None):
    """
    Set pandas display options. Display all rows and columns by default.
    """
    display_options = {'display.max_rows': max_rows,
                       'display.max_columns': max_columns,
                       'display.width': width,
                       'display.max_colwidth': max_colwidth}

    for k in display_options:
        try:
            pd.set_option(k, display_options[k])
        except ValueError:
            msg = """Newer version of Pandas required to set the '{}'
                        option.""".format(k)
            warnings.warn(msg)


def format_pvalues(table, pval, pval_adjust, pval_threshold):
    """
    Formats the p value columns, applying rounding rules and adding
    significance markers based on defined thresholds.
    """
    # round pval column and convert to string
    if pval and pval_adjust:
        table['P-Value (adjusted)'] = table['P-Value (adjusted)'].apply('{:.3f}'.format).astype(str)
        table.loc[table['P-Value (adjusted)'] == '0.000',
                  'P-Value (adjusted)'] = '<0.001'

        if pval_threshold:
            asterisk_mask = table['P-Value (adjusted)'] < pval_threshold
            table.loc[asterisk_mask, 'P-Value (adjusted)'] = (
                table['P-Value (adjusted)'][asterisk_mask].astype(str)+"*"  # type: ignore
            )

    elif pval:
        table['P-Value'] = table['P-Value'].apply('{:.3f}'.format).astype(str)
        table.loc[table['P-Value'] == '0.000', 'P-Value'] = '<0.001'

        if pval_threshold:
            asterisk_mask = table['P-Value'] < pval_threshold
            table.loc[asterisk_mask, 'P-Value'] = (
                table['P-Value'][asterisk_mask].astype(str)+"*"  # type: ignore
            )

    return table


def format_smd_columns(table, smd, smd_table):
    """
    Formats the SMD (Standardized Mean Differences) columns. Rounds the SMD values
    and ensures they are presented as strings.
    """
    # round smd columns and convert to string
    if smd and smd_table is not None:
        for c in list(smd_table.columns):
            table[c] = table[c].apply('{:.3f}'.format).astype(str)
            table.loc[table[c] == '0.000', c] = '<0.001'

    return table


def apply_limits(table, data, limits, categorical, order):
    """
    Applies limits to the number of categories shown for each categorical variable
    in the DataFrame, based on specified requirements.
    """
    # set the limit on the number of categorical variables
    if limits:
        levelcounts = data[categorical].nunique()

        for k, _ in levelcounts.items():
            # set the limit for the variable
            if (isinstance(limits, int)
                    and levelcounts[k] >= limits):
                limit = limits
            elif isinstance(limits, dict) and k in limits:
                limit = limits[k]
            else:
                continue

            if not order or (order and k not in order):
                # re-order the variables by frequency
                count = data[k].value_counts().sort_values(ascending=False)
                new_idx = [(k, '{}'.format(i)) for i in count.index]
            else:
                # apply order
                all_var = table.loc[k].index.unique(level='value')
                new_idx = [(k, '{}'.format(v)) for v in order[k]]
                new_idx += [(k, '{}'.format(v)) for v in all_var
                            if v not in order[k]]

            # restructure to match the original idx
            new_idx_array = np.empty((len(new_idx),), dtype=object)
            new_idx_array[:] = [tuple(i) for i in new_idx]
            orig_idx = table.index.values.copy()
            orig_idx[table.index.get_loc(k)] = new_idx_array
            table = table.reindex(orig_idx)

            # drop the rows > the limit
            table = table.drop(new_idx_array[limit:])  # type: ignore

    return table


def sort_and_reindex(table, smd, smd_table, sort, columns):
    """
    Sorts and reindexes the table to meet requirements.
    """
    # sort the table rows
    sort_columns = ['Missing', 'P-Value', 'P-Value (adjusted)', 'Test']

    if smd and smd_table is not None:
        sort_columns = sort_columns + list(smd_table.columns)

    if sort and isinstance(sort, bool):
        new_index = sorted(table.index.values, key=lambda x: x[0].lower())
    elif sort and isinstance(sort, str) and (sort in sort_columns):
        try:
            new_index = table.sort_values(sort).index
        except KeyError:
            new_index = sorted(table.index.values,
                               key=lambda x: columns.index(x[0]))
            warnings.warn(f'Sort variable not found: {sort}')
    elif sort and isinstance(sort, str) and (sort not in sort_columns):
        new_index = sorted(table.index.values,
                           key=lambda x: columns.index(x[0]))
        warnings.warn(f'Sort must be in the following list: {sort}')
    else:
        # sort by the columns argument
        new_index = sorted(table.index.values,
                           key=lambda x: columns.index(x[0]))
    table = table.reindex(new_index)

    return table


def apply_order(table, order, groupby):
    """
    Applies a predefined order to rows based on specified requirements.
    May include reordering based on categorical group levels or other criteria.
    """
    # if an order is specified, apply it
    if order:
        for k in order:
            # Skip if the variable isn't present
            try:
                all_var = table.loc[k].index.unique(level='value')
            except KeyError:
                if k not in groupby:  # type: ignore
                    warnings.warn(f"Order variable not found: {k}")
                continue

            # Remove value from order if it is not present
            if [i for i in order[k] if i not in all_var]:
                rm_var = [i for i in order[k] if i not in all_var]
                order[k] = [i for i in order[k] if i in all_var]
                warnings.warn(f'Order value not found: "{k}: {rm_var}"')

            new_seq = [(k, '{}'.format(v)) for v in order[k]]
            new_seq += [(k, '{}'.format(v)) for v in all_var
                        if v not in order[k]]

            # restructure to match the original idx
            new_idx_array = np.empty((len(new_seq),), dtype=object)
            new_idx_array[:] = [tuple(i) for i in new_seq]
            orig_idx = table.index.values.copy()
            orig_idx[table.index.get_loc(k)] = new_idx_array
            table = table.reindex(orig_idx)

    return table


def mask_duplicate_values(table, optional_columns, smd, smd_table):
    """
    Masks duplicate values, ensuring that repeated values (e.g. counts of
    missing values) are only displayed once.
    """
    # only display data in first level row
    dupe_mask = table.groupby(level=[0]).cumcount().ne(0)  # type: ignore
    dupe_columns = ['Missing']

    if smd and smd_table is not None:
        optional_columns = optional_columns + list(smd_table.columns)
    for col in optional_columns:
        if col in table.columns.values:
            dupe_columns.append(col)

    table[dupe_columns] = table[dupe_columns].mask(dupe_mask).fillna('')

    return table


def create_row_labels(columns, alt_labels, label_suffix, nonnormal, 
                      min_max, categorical) -> dict:
    """
    Take the original labels for rows. Rename if alternative labels are
    provided. Append label suffix if label_suffix is True.

    Returns
    ----------
    labels : dictionary
        Dictionary, keys are original column name, values are final label.

    """
    # start with the original column names
    labels = {}
    for c in columns:
        labels[c] = c

    # replace column names with alternative names if provided
    if alt_labels:
        for k in alt_labels.keys():
            labels[k] = alt_labels[k]

    # append the label suffix
    if label_suffix:
        for k in labels.keys():
            if k in nonnormal:
                if min_max and k in min_max:
                    labels[k] = "{}, {}".format(labels[k],
                                                "median [min,max]")
                else:
                    labels[k] = "{}, {}".format(labels[k],
                                                "median [Q1,Q3]")
            elif k in categorical:
                labels[k] = "{}, {}".format(labels[k], "n (%)")
            else:
                if min_max and k in min_max:
                    labels[k] = "{}, {}".format(labels[k],
                                                "mean [min,max]")
                else:
                    labels[k] = "{}, {}".format(labels[k],
                                                "mean (SD)")

    return labels


def reorder_columns(table, optional_columns, groupby, order, overall):
    """
    Reorder columns for consistent, predictable formatting.
    """
    if groupby and order and (groupby in order):
        header = ['{}'.format(v) for v in table.columns.levels[1].values]  # type: ignore
        cols = order[groupby] + ['{}'.format(v) for v in header if v not in order[groupby]]
    elif groupby:
        cols = ['{}'.format(v) for v in table.columns.levels[1].values]  # type: ignore
    else:
        cols = ['{}'.format(v) for v in table.columns.values]

    if groupby and overall:
        cols = ['Overall'] + [x for x in cols if x != 'Overall']

    if 'Missing' in cols:
        cols = ['Missing'] + [x for x in cols if x != 'Missing']

    # move optional_columns to the end of the dataframe
    for col in optional_columns:
        if col in cols:
            cols = [x for x in cols if x != col] + [col]

    if groupby:
        table = table.reindex(cols, axis=1, level=1)
    else:
        table = table.reindex(cols, axis=1)

    return table
