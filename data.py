import pandas as pd

def domain_cutoff(df, config):
    """Return the last domain whose cumulative fraction is still <= target_ratio."""
    total = len(df)
    cumulative = 0
    last_valid_domain = None
    for domain in sorted(df[config.domain_col].unique()):
        cumulative += (df[config.domain_col] == domain).sum()
        if cumulative / total <= config.train_ratio:
            last_valid_domain = domain
        else:
            break
    return last_valid_domain

def prepare_splits_ood(df, config, feature_cols, cutoff, seed):
    """Domain-based OOD split.
   
    ID domains (<= cutoff) are split into train and validation by sampling config.val_ratio of each domain group as the val set. OOD domains (> cutoff) form the test set. Returns: X_train, y_train, X_val, y_val, X_test, y_test """
    id_full = df[df[config.domain_col] <= cutoff].copy()
    ood_test = df[df[config.domain_col] > cutoff].copy()
   
    val_parts, train_parts = [], []
    for _, group in id_full.groupby(config.domain_col):
        n_val = max(1, int(round(len(group) * config.val_ratio)))
        val_sample = group.sample(n=n_val, random_state=seed)
        train_parts.append(group.drop(val_sample.index))
        val_parts.append(val_sample)
       
    train_data = pd.concat(train_parts)
    val_data = pd.concat(val_parts)

    return (train_data[feature_cols], train_data['Label'],
            val_data[feature_cols], val_data['Label'],
            ood_test[feature_cols], ood_test['Label'])


def prepare_splits_id(df, config, feature_cols, seed):
    """Random ID split — no domain-based separation.

    Splits the full dataset into 70 % train / 10 % val / 20 % test.

    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_test = int(round(n * 0.20))
    n_val = int(round(n * 0.10))

    test_data  = shuffled.iloc[:n_test]
    val_data   = shuffled.iloc[n_test:n_test + n_val]
    train_data = shuffled.iloc[n_test + n_val:]

    return (
        train_data[feature_cols], train_data[config.target],
        val_data[feature_cols],   val_data[config.target],
        test_data[feature_cols],  test_data[config.target],
    )