import numpy as np
import pandas as pd


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

    # Bring in stats data set
    combined = combined.merge(stats, on='key', how='outer')

    # Drop missing rows with no positions
    combined.dropna(subset=['pos'], inplace=True)

    # Fill NaNs in stats columns to 0
    combined.loc[:, stats.columns] = combined.loc[:, stats.columns].fillna(0)

    # Flag players who didn't participate in the combine with 0
    combined['combine_participant'].fillna(0, inplace=True)
    combined['pick'].fillna(0, inplace=True)
    combined['round'].fillna(0, inplace=True)
    combined['team'].fillna('Unknown', inplace=True)

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

    # Side of Ball
    df['side_of_ball'] = df['pos'].map(side_of_ball())

    # Position Group
    df['position_group'] = df['pos'].map(position_group())

    # Impute Combine Stats
    df = impute_combine_stats(df)

    # Roll-up Colleges
    df = roll_up_colleges(df)

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


def side_of_ball():
    """
    Map positions to general side of ball.
    :return:
    """
    side = pd.Series({
        'offense': ['T', 'WR', 'RB', 'TE', 'QB', 'G', 'C', 'FB', 'OL', 'OG', 'OT'],
        'defense': ['DE', 'LB', 'DT', 'DB', 'NT', 'DL', 'OLB', 'CB', 'FS', 'ILB', 'SS', 'S'],
        'special_teams': ['K', 'P', 'LS']
        })

    position_map = {}
    for k, v in side.iteritems():
        for i in v:
            position_map.update({i: k})

    return position_map


def impute_combine_stats(df):
    """
    Impute missing combine stats by position group
    :param df:
    :return:
    """
    cols = ['position_group', 'age', 'height', 'weight', 'forty', 'vertical', 'broad', 'bench',
            'threecone', 'shuttle']

    imputer = df.loc[:, cols].groupby('position_group').mean().to_dict()

    for metric in imputer:
        df[metric].fillna(df['position_group'].map(imputer[metric]), inplace=True)

    return df


def classify_college(x):
    """
    Classify colleges based on their z-score.
    :param x:
    :return:
    """
    if type(x.zscore) not in [float, int]:
        return np.NaN
    elif x.zscore >= 2:
        return x['index']
    elif 0 <= x.zscore < 2:
        return "Mid-sized School"
    else:
        return "Small School"


def roll_up_colleges(df):
    """
    Using classify_college(), roll middling and small schools into general categories
    :param df:
    :return:
    """
    # Get counts of players from each school
    colleges = df.college.value_counts().to_frame().reset_index()

    # Get z-score based on number of players
    colleges['zscore'] = (colleges.college - colleges.college.mean()) / colleges.college.std(ddof=0)

    # Roll middling and small schools up
    colleges['rename'] = colleges.apply(classify_college, axis=1)

    # Send new values to dictionary
    new_names = dict(colleges[['index', 'rename']].to_dict('split')['data'])

    # Remap colleges based on dictionary
    df['college'] = df['college'].map(new_names)

    return df


""" **** MODELING PREP **** """


def perform_modeling_prep(df):

    df = drop_extra_cols(df)

    df = factorize_object_columns(df)

    return df


def drop_extra_cols(df):
    """
    Drop columns not needed for modeling
    :param df:
    :return:
    """
    df = df.drop(['year', 'pick', 'team', 'player', 'key'], axis=1)

    return df


def factorize_object_columns(df):
    """
    Convert object (string) columns to factors (numbers)
    :param df:
    :return:
    """
    cols_to_factorize = df.select_dtypes(include=['object']).columns.tolist()

    for col in cols_to_factorize:
        df[col] = pd.factorize(df[col])[0]

    return df


""" **** UNIT TESTS **** """


def perform_unit_tests():

    unit_test_height_to_inches()
    unit_test_classify_colleges()
    unit_test_roll_up_colleges()
    unit_test_factorize_object_columns()


def unit_test_height_to_inches():
    tests = ["6-2", "5-0", "10-11", "0-8", 2, float('NaN')]
    expected = [74, 60, 131, 8, np.NaN, np.NaN]
    received = []

    for test in tests:
        received.append(height_to_inches(test))

    assert set(expected) == set(received), "height_to_inches() - DID NOT RECEIVE EXPECTED OUTCOME"


def unit_test_classify_colleges():
    df = pd.DataFrame.from_dict({'index': ['a', 'b', 'c', 'd'], 'zscore': [3.24, 0, -0.132, 'x']})

    expected = ['a', 'Mid-sized School', 'Small School', np.NaN]

    df['results'] = df.apply(classify_college, axis=1)

    assert set(expected) == set(df['results']), "classify_college() - DID NOT RECEIVE EXPECTED OUTCOME"


def unit_test_roll_up_colleges():

    expected = [15, 25, 40]

    values = []
    for i in range(0, 40):
        values.append('a')

    for i in range(0, 15):
        values.append('b')

    for i in range(0, 5):
        values.append('c')
        values.append('d')
        values.append('e')
        values.append('f')
        values.append('g')

    df = pd.DataFrame.from_dict({"college": values})

    df = roll_up_colleges(df)

    assert set(expected) == set(df['college'].value_counts()), "roll_up_colleges() - DID NOT RECEIVE EXPECTED OUTCOME"


def unit_test_factorize_object_columns():

    data = [[1, '1', True, 321042.241], [2, 'b', False, 131.01], [3, 43, True, np.NaN]]

    df = pd.DataFrame(data=data, columns=['a', 'b', 'c', 'd'])

    df = factorize_object_columns(df)

    assert len(df.select_dtypes(include=['object']).columns) == 0,\
        "factorize_object_columns() - DID NOT RECEIVE EXPECTED OUTCOME"
