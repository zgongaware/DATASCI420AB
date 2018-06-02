
def classify_college(x):
    """
    Classify colleges based on their z-score.
    :param x:
    :return:
    """
    if type(x.zscore) not in [float, int]:
        return np.NaN
    elif x.zscore >= 2:
        #return 'Top School'
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


def unit_test_classify_colleges():
    df = pd.DataFrame.from_dict({'index': ['a', 'b', 'c', 'd'], 'zscore': [3.24, 0, -0.132, 'x']})

    expected = ['Top School', 'Mid-sized School', 'Small School', np.NaN]

    df['results'] = df.apply(classify_college, axis=1)

    assert set(expected) == set(df['results']), "classify_college() - DID NOT RECEIVE EXPECTED OUTCOME"


def impute_combine_values_knn(df):
    """
    Use K-Nearest Neighbors function to impute missing combine and physical values
    :param df:
    :return:
    """
    # Columns to drop
    drop_cols = ['year', 'pick', 'team', 'player', 'key', 'position_group', 'college', 'pos']

    # Columns to impute
    impute_cols = ['age', 'height', 'weight', 'forty', 'vertical', 'broad', 'bench', 'threecone', 'shuttle']

    for col in impute_cols:
        col_df = knn_impute(target=df[col],
                            attributes=df.drop(drop_cols, 1),
                            aggregation_method="median",
                            k_neighbors=5,
                            numeric_distance='euclidean',
                            categorical_distance='hamming',
                            missing_neighbors_threshold=0.5)

        df[col] = col_df

    return df


def perform_svm(X_train, y_train, X_test, y_test):
    """
    Fit SVM classifier model and return classifier, predictions, and accuracy score
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    # Define and fit SVM classifier
    clf = svm.SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)

    # Create the predictions on the validation data without the label
    y_predict = clf.predict(X_test)

    # Determine accuracy of these predictions
    accuracy = clf.score(X_test, y_test)

    return clf, y_predict, accuracy


def position_correction(x):
    """
    Correct erroneous positions
    :return:
    """

    if x in ['P', 'K', 'LS']:
        return 'ST'
    elif x == 'T':
        return 'OT'
    elif x == 'G':
        return 'OG'
    elif x == 'S':
        return 'FS'
    elif x == 'NT':
        return 'DT'
    else:
        return x