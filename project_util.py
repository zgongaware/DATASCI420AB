from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



""" **** DATA COLLECTION **** """


def load_draft_data():

    # Load from CSV
    cols = ['year', 'round', 'pick', 'team', 'player', 'college', 'pos', 'age', 'url']
    drafts = pd.read_csv('data\drafts.csv', usecols=cols)

    # Add key column
    drafts['key'] = drafts['url'].fillna(drafts['player'].astype(str) + '-' + drafts['year'].astype(str))

    return drafts


def load_combine_data():

    # Load from CSV
    cols = ['year', 'player', 'pos', 'college', 'height', 'weight', 'forty', 'vertical', 'broad', 'bench',
            'threecone', 'shuttle', 'url']
    combine = pd.read_csv('data\combines.csv', usecols=cols)

    # Rename columns
    col_names = {'year': 'year_combine', 'player': 'player_combine',
                 'pos': 'pos_combine', 'college': 'college_combine',
                 'url': 'url_combine'}
    combine = combine.rename(columns=col_names)

    # Add key column
    combine['key'] = combine['url_combine'].fillna(
        combine['player_combine'].astype(str) + '-' + combine['year_combine'].astype(str))

    # Remove duplicates based on key column
    combine = combine.drop_duplicates(subset=['key'], keep='first')

    # Add combine_participant column
    combine['combine_participant'] = 1

    return combine


def load_college_data():

    stats = pd.read_csv('data\college_stats.csv', usecols=['url', 'stat', 'value'])

    # Rename columns
    col_names = {'url': 'key', 'stat': 'metric'}
    stats = stats.rename(columns=col_names)

    # Roll up duplicate stats
    stats = stats.groupby(['key', 'metric'], as_index=False).agg({'value': 'max'})

    # Format stat column
    stats['metric'] = stats['metric'].str.replace('.', '_')

    # Pivot
    stats = stats.pivot(index='key', columns='metric', values='value').reset_index()

    # Fill NaN's
    stats.fillna(value=0, inplace=True)

    return stats


def combine_data_sets(draft, comb, stats):
    # Combine draft and combine data sets
    combined = draft.merge(comb, on='key', how='outer')

    # Combine duplicate columns
    combined['year'] = draft['year'].combine_first(comb['year_combine'])
    combined['player'] = draft['player'].combine_first(comb['player_combine'])
    combined['pos'] = draft['pos'].combine_first(comb['pos_combine'])
    combined['college'] = draft['college'].combine_first(comb['college_combine'])

    # Drop extra columns
    combined.drop(['year_combine', 'player_combine', 'pos_combine', 'url',
                   'url_combine', 'college_combine'], axis=1, inplace=True)

    # Drop columns we don't wish to include
    drop_cols = ['attempts', 'scrim_avg', 'scrim_tds', 'comp_pct', 'int_rate',
                 'int_yards_avg', 'kick_fgm', 'kick_return_avg', 'kick_xpm', 'punt_return_avg',
                 'rec_avg', 'rush_avg', 'safety', 'scrim_plays', 'td_fr', 'td_int', 'td_kr', 'td_pr', 'td_rec',
                 'td_rush', 'twopm', 'yards_per_attempt',
                 'kick_return_yards', 'kick_returns', 'fum_tds',
                 'kick_return_td', 'punt_return_td', 'punt_returns', 'pass_yards', 'rec_td', 'solo_tackes',
                 'ast_tackles', 'seasons', 'rush_yds', 'rec_yards', 'sacks', 'int_yards', 'pass_tds', 'td_tot',
                 'pass_ints', 'receptions', 'adj_yards_per_attempt',
                 'punt_return_yards', 'fum_rec', 'fum_yds', 'int_td', 'rush_td']

    stats = stats.drop(drop_cols, axis=1)

    # Bring in stats data set
    combined = combined.merge(stats, on='key', how='outer')

    # Drop missing rows with no positions
    combined.dropna(subset=['pos'], inplace=True)

    # Fill NaNs in stats columns to 0
    combined.loc[:, stats.columns] = combined.loc[:, stats.columns].fillna(0)

    # Flag players who didn't participate in the combine with 0
    combined['combine_participant'].fillna(0, inplace=True)
    combined['pick'].fillna(0, inplace=True)
    combined['round'].fillna(8, inplace=True)
    combined['team'].fillna('Unknown', inplace=True)

    # Convert numeric columns back to int
    int_cols = ['round', 'combine_participant', 'age']

    combined[int_cols] = combined[int_cols].astype(int, errors='ignore')

    return combined


""" **** FEATURE ENGINEERING **** """


def engineer_features(df):
    """
    Execute feature engineering functions.
    :param df:
    :return:
    """
    # Format Height
    df['height'] = df['height'].apply(lambda x: height_to_inches(x))

    # Position Group
    df['position_group'] = df['pos'].map(position_group())

    # Bin Ages
    df['age'] = df['age'].apply(lambda x: bin_age(x))

    # Roll-up Colleges
    df = create_college_tier(df)

    # Impute Combine Stats
    df = impute_combine_stats(df)

    return df


def height_to_inches(x):
    """
    Convert height string (e.g. "6-2") to total inches.
    :param x:
    :return:
    """
    if len(str(x).split('-')) == 1:
        return np.NaN

    else:
        f, i = str(x).split('-')
        inches = (int(f) * 12) + int(i)
        return inches


def bin_age(x):
    """
    Bin ages with 22 as baseline
    :param x:
    :return:
    """

    if x < 22:
        return 'under_22yo'
    if x >= 24:
        return 'over_23yo'
    else:
        return '22-23yo'


def position_group():
    """
    Map positions to generalized group.
    :return:
    """
    position = pd.Series({
        'OL': ['OT', 'OG', 'C', 'OL', 'T', 'G'],
        'WR': ['WR'],
        'QB': ['QB'],
        'RB': ['RB', 'FB'],
        'TE': ['TE'],
        'DL': ['DE', 'DT', 'DL', 'NT'],
        'LB': ['LB', 'ILB', 'OLB'],
        'DB': ['CB', 'DB', 'FS', 'S', 'SS'],
        'ST': ['K', 'P', 'LS']
    })

    position_map = {}
    for k, v in position.iteritems():
        for i in v:
            position_map.update({i: k})

    return position_map


def impute_combine_stats(df):
    """
    Impute missing combine stats by position group
    :param df:
    :return:
    """
    cols = ['pos', 'age', 'height', 'weight', 'forty', 'vertical', 'broad', 'bench',
            'threecone', 'shuttle']

    stdev = df.loc[:, cols].groupby('pos').std()
    mean = df.loc[:, cols].groupby('pos').mean()

    imputer = (mean - stdev).dropna().to_dict()

    for metric in imputer:
        df[metric].fillna(df['position_group'].map(imputer[metric]), inplace=True)

    # Drop rows that are still NaNs
    df.dropna(axis=0, how='any', inplace=True)

    return df


def create_college_tier(df):
    df['draftees'] = df['round'].apply(lambda x: 0 if x == 8 else 1)

    # Roll up colleges by number of draftees
    coll = df.groupby('college').agg({'draftees': 'sum'})

    # Assign tier based on quartile
    coll['college_tier'] = pd.qcut(coll.draftees, q=4, duplicates='drop', labels=[4, 3, 2, 1])

    # Map to dataframe
    df['college_tier'] = df.college.map(coll['college_tier'].to_dict()).astype(int)

    df = df.drop(['draftees'], axis=1)

    return df


""" **** MODELING PREP **** """


def perform_modeling_prep(df, target_col, test_size=0.35):

    df = scale_features(df, target_col)

    df = binarize_columns(df)

    df = drop_extra_cols(df)

    X_train, y_train, X_test, y_test = create_test_and_train_sets(df, target_col, test_size=test_size) #, seed=10)

    return X_train, y_train, X_test, y_test


def scale_features(df, target_col):
    """
    Use min-max scaling on all numeric features.
    :param df:
    :return:
    """

    # Skip Columns
    skip_cols = ['year', 'pick', 'team', 'player', 'key', 'age', 'position_group', 'college', 'pos',
                 'college_tier', 'combine_participant'] + [target_col]
    # Scale features
    scaled_features = {}

    for each in df.drop(skip_cols, axis=1).columns:
        mean, std = df[each].mean(), df[each].std()
        scaled_features[each] = [mean, std]
        df.loc[:, each] = (df[each] - mean) / std

    return df


def binarize_columns(df):
    """
    Convert categorical columns to binary flags
    :param df:
    :return:
    """
    # Position
    df = pd.concat([df, pd.get_dummies(df['pos'], drop_first=True)], axis=1)

    # College
    df = pd.concat([df, pd.get_dummies(df['college_tier'], prefix='tier', drop_first=True)], axis=1)

    # Age
    df = pd.concat([df, pd.get_dummies(df['age'], drop_first=True)], axis=1)

    # Convert spaces to underscore and lower case
    df.columns = df.columns.str.replace('\s+', '_').str.lower()

    return df


def drop_extra_cols(df):
    """
    Drop columns not needed for modeling
    :param df:
    :return:
    """
    df = df.drop(['year', 'pick', 'team', 'player', 'key', 'age', 'position_group', 'college', 'pos',
                  'college_tier'], axis=1)

    return df


def create_test_and_train_sets(df, target_col, test_size=0.25, seed=30):
    """
    Split into training and test sets and break out target column
    :param df:
    :param target_col:
    :param test_size:
    :return:
    """

    # Split into training and testing data
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target_col])

    # Break out feature and target columns
    X_train = train.drop(target_col, axis=1)
    y_train = train[target_col]
    X_test = test.drop(target_col, axis=1)
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test


""" **** MODELING **** """


def perform_lasso_regression(X_train, y_train, X_test, y_test, alpha):
    """
    Perform lasso regression to identify top features
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param alpha:
    :return:
    """
    lasso = Lasso(alpha=alpha)

    y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)

    r2_score_lasso = metrics.r2_score(y_test, y_pred_lasso)

    print("r^2 on test data : %f" % r2_score_lasso)

    return lasso, r2_score_lasso


def perform_random_forest_regressor(X_train, y_train, X_test, y_test):
    """
    Fit Random Forest model and return regressor, predictions, and accuracy score
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    # Define and fit the random forest model
    clf = RandomForestRegressor(n_estimators=100, n_jobs=3)
    clf.fit(X_train, y_train)

    # Create the predictions on the validation data without the label
    y_predict = clf.predict(X_test)

    # Determine accuracy of these predictions
    r2 = clf.score(X_test, y_test)

    return clf, y_predict, r2


def perform_random_forest(X_train, y_train, X_test, y_test):
    """
    Fit Random Forest model and return classifier, predictions, and accuracy score
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    # Define and fit the random forest model
    clf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    clf.fit(X_train, y_train)

    # Create the predictions on the validation data without the label
    y_predict = clf.predict(X_test)

    # Determine accuracy of these predictions
    r2 = clf.score(X_test, y_test)

    return clf, y_predict, r2


def perform_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Fit logistic regression model and return regressor, predictions, and accuracy score
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    # Define and fit the random forest model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Create the predictions on the validation data without the label
    y_predict = clf.predict(X_test)

    # Determine accuracy of these predictions
    accuracy = clf.score(X_test, y_test)

    return clf, y_predict, accuracy


def perform_naive_bayes(X_train, y_train, X_test, y_test):
    """
    Fit naive bayes model and return classifier, predictions, and accuracy score
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    # Define and fit the random forest model
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Create the predictions on the validation data without the label
    y_predict = clf.predict(X_test)

    # Determine accuracy of these predictions
    accuracy = clf.score(X_test, y_test)

    return clf, y_predict, accuracy


""" **** MODEL EVALUATION **** """


def output_performance(r2, actual, predicted, perf_type='R^2'):
    """
    Output model performance metrics.
    :param r2:
    :param actual:
    :param predicted:
    :param type:
    :return:
    """
    if perf_type == 'R^2':
        print("R^2: {}".format(r2))
    elif perf_type == 'accuracy':
        print('Accuracy: {}'.format(r2))

    print("MSE: {}".format(metrics.mean_squared_error(actual, predicted)))
    print("MAE: {}".format(metrics.mean_absolute_error(actual, predicted)))


def plot_performance(actual, predicted):
    """Scatter plot of actual vs predicted values highlighting correct predictions."""
    x = list(range(1, 9))

    y = list(range(1, 9))
    y_max = [i + 0.49 for i in y]
    y_min = [i - 0.49 for i in y]

    c = pd.DataFrame(list(zip(actual, predicted)), columns=['actual', 'predicted'])
    c['match'] = c.apply(lambda x: x.actual == x.predicted.round(), axis=1)
    matches = c[c.match]

    _ = plt.figure(figsize=(8, 8))
    _ = plt.scatter(actual, predicted, alpha=0.25, color='grey')
    _ = plt.scatter(matches.actual, matches.predicted, alpha=0.5, color='navy')
    _ = plt.plot(x, y, color='red')
    _ = plt.plot(x, y_max, linestyle='--', color='red', alpha=0.5)
    _ = plt.plot(x, y_min, linestyle='--', color='red', alpha=0.5)
    _ = plt.xlabel('Actual')
    _ = plt.ylabel('Predicted')
    _ = plt.title('Model Performance - Round Predictions', fontsize=16)

    plt.show()


def plot_feature_importance(coefficients, features):
    """
    Plot features by order of importance to model
    :param model:
    :param features:
    :return:
    """
    feature_imp = pd.Series(coefficients, index=features.columns).sort_values(ascending=False)
    _ = plt.figure(figsize=(8, 11))
    _ = sns.barplot(x=feature_imp, y=feature_imp.index)
    _ = plt.xlabel('Feature Importance Score')
    _ = plt.ylabel('Features')
    _ = plt.title("Visualizing Important Features", fontsize=16)
    plt.legend()

    plt.show()


def plot_prediction_distribution(actual, predicted):
    """
    Plot distributions of actual and predicted data
    :param actual:
    :param predicted:
    :return:
    """
    _ = plt.figure(figsize=(10, 6))
    _ = sns.distplot(predicted, hist=False, label='Predicted')
    _ = sns.distplot(actual, hist=False, label='Actual')
    _.set_xlabel('Round Drafted')
    _.set_ylabel('Distribution')
    _.set_title('Round Distribution - Actual vs Predicted', fontsize=16)
    _ = plt.legend()

    plt.show()


""" **** UNIT TESTS **** """


def perform_unit_tests():

    unit_test_height_to_inches()
    unit_test_bin_age()


def unit_test_height_to_inches():
    tests = ["6-2", "5-0", "10-11", "0-8", 2, float('NaN')]
    expected = [74, 60, 131, 8, np.NaN, np.NaN]
    received = []

    for test in tests:
        received.append(height_to_inches(test))

    assert set(expected) == set(received), "height_to_inches() - DID NOT RECEIVE EXPECTED OUTCOME"


def unit_test_bin_age():
    df = pd.DataFrame([0, 23, 22.2521, 45, 21, 0.23, 24, 22], columns=['age'])

    df['age'] = df['age'].apply(lambda x: bin_age(x))

    assert np.array_equal(df['age'].value_counts().values, np.array([3, 3, 2])),\
        "bin_age() - DID NOT RECEIVE EXPECTED OUTCOME"







