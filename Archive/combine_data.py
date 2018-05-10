from sqlalchemy import create_engine
import config       # Credentials stored in config.py in working directory
import pandas as pd


def main():

    # Define postgres engine
    engine = create_engine('postgresql://{}:{}@localhost:5432/DataSets'.format(config.username, config.password))

    print("Attempt 1...")

    # Process
    first = merge_data(engine, attempt=1)
    first = process_attempt(first)

    # Find complete records
    complete = find_complete_records(first)
    print(len(complete))


    print('Attempt 2...')

    # Process
    second = merge_data(engine, attempt=2, predecessor=complete)
    second = process_attempt(second)

    # Merge attempts
    first = merge_attempts(first, second)

    # Find complete records
    complete = find_complete_records(first)
    print(len(complete))

    print('Attempt 3...')

    # Process
    third = merge_data(engine, attempt=3, predecessor=complete)
    third = process_attempt(third)

    # Merge attempts
    first = merge_attempts(first, third)

    # Find complete records
    complete = find_complete_records(first)
    print(len(complete))

    print('Attempt 4...')

    # Process
    fourth = merge_data(engine, attempt=4, predecessor=complete)
    fourth = process_attempt(fourth)

    # Merge attempts
    first = merge_attempts(first, fourth)

    # Find complete records
    complete = find_complete_records(first)
    print(len(complete))

    print('Attempt 5...')

    # Process
    fifth = merge_data(engine, attempt=5, predecessor=complete)
    fifth = process_attempt(fifth)

    # Merge attempts
    first = merge_attempts(first, fifth)

    # Find complete records
    complete = find_complete_records(first)
    print(len(complete))

    print('Attempt 6...')

    # Process
    sixth = merge_data(engine, attempt=6, predecessor=complete)
    sixth = process_attempt(sixth)

    # Merge attempts
    first = merge_attempts(first, sixth)

    # Find complete records
    complete = find_complete_records(first)
    print(len(complete))

    print('Attempt 7...')

    # Process
    seventh = merge_data(engine, attempt=6, predecessor=complete)
    seventh = process_attempt(seventh)

    # Merge attempts
    first = merge_attempts(first, seventh)

    # Find complete records
    complete = find_complete_records(first)
    print(len(complete))

    # Send to database
    send_to_database(first, engine)


def process_attempt(data):
    
    # Create additional features
    data = engineer_features(data)

    # Drop extra columns
    data = drop_columns(data)

    # Drop duplicates
    data = drop_duplicates(data)

    return data


def merge_attempts(data1, data2):

    data2 = data2[data2['player_code'].notnull() & data2['round'].notnull()]

    data1.update(data2)

    return data1


def find_complete_records(df):
    df = df[df['player_code'].notnull() & df['round'].notnull()]

    return df['index']


def drop_duplicates(df):
    # Drop duplicates
    df = df.drop_duplicates(subset=['first_name', 'last_name', 'position', 'year'], keep='last')

    return df


def send_to_database(df, engine):

    print("Populate players_final table...")

    df.to_sql('players_final', engine, index=False, if_exists='replace')

    print("players_final loaded")


def merge_data(engine, attempt=1, predecessor=[]):

    print("Load and combine data sources...")

    # Years of available data
    years = range(2005, 2014)

    # Load game stats table and format
    stats = pd.read_sql_table('player_stats_combined', engine)
    stats = stats.dropna(axis=0, how='any', subset=['first_name', 'last_name', 'position'])

    stats['first_name'] = stats['first_name'].apply(lambda x: x.lower().replace(".",""))
    stats['last_name'] = stats['last_name'].apply(lambda x: x.lower().replace(".",""))
    stats['year'] = stats['year'].apply(lambda x: x+1)
    stats['position'] = stats['position'].map(roll_up_positions())
    stats['position_group'] = stats['position'].map(position_group())

    # Load combine stats table and format
    comb = pd.read_sql_table('combine_data', engine)
    comb = comb[comb['year'].isin(years)]
    comb['first_name'] = comb['first_name'].apply(lambda x: x.lower().replace(".",""))
    comb['last_name'] = comb['last_name'].apply(lambda x: x.lower().replace(".",""))
    comb['position'] = comb['position'].map(roll_up_positions())
    comb['position_group'] = comb['position'].map(position_group())

    if len(predecessor) > 0:
        comb = comb[~comb['index'].isin(predecessor)]

    # Merge as best we can
    if attempt == 1:

        both = comb.merge(stats, on=('first_name', 'last_name', 'position', 'year'), how='outer')
        print("Attempt 1 - Complete.")

    elif attempt == 2:

        both = comb.merge(stats, on=('first_name', 'last_name', 'position_group', 'year'), how='inner')

        both['position'] = both['position_x']
        both = both.drop(['position_x', 'position_y'], axis=1)

        print("Attempt 2 - Complete.")

    elif attempt == 3:

        stats['year'] = stats['year'].apply(lambda x: x-1)
        both = comb.merge(stats, on=('first_name', 'last_name', 'position', 'year'), how='inner')
        print("Attempt 3 - Complete.")
        
    elif attempt == 4:
        
        both = comb.merge(stats, on=('first_name', 'last_name', 'position_group'), how='inner')

        both['position'] = both['position_x']
        both['year'] = both['year_x']
        both = both.drop(['year_x', 'year_y', 'position_x', 'position_y'], axis=1)

        print("Attempt 4 - Complete.")

    elif attempt == 5:
        
        both = comb.merge(stats, on=('first_name', 'last_name', 'year'), how='inner')

        both['position'] = both['position_x']
        both = both.drop(['position_x', 'position_y'], axis=1)

        print("Attempt 5 - Complete.")
        
    elif attempt == 6:
        
        both = comb.merge(stats, on=('first_name', 'last_name'), how='inner')

        both['position'] = both['position_x']
        both['year'] = both['year_x']
        both = both.drop(['year_x', 'year_y', 'position_x', 'position_y'], axis=1)

        print("Attempt 6 - Complete.")

    elif attempt == 7:

        both = comb.merge(stats, left_on=('first_name', 'last_name', 'college'),
                          right_on=('first_name', 'last_name', 'team_name'), how='inner')

        both['position'] = both['position_x']
        both['year'] = both['year_x']
        both = both.drop(['year_x', 'year_y', 'position_x', 'position_y'], axis=1)

        print("Attempt 7 - Complete.")

    return both


def drop_columns(df):

    print("Remove extraneous columns...")

    # Drop extra columns
    drop_cols = ['index_x', 'weight_x', 'weight_y', 'Unnamed: 26', 'Unnamed: 27', 'heightinchestotal',
                 'heightfeet', 'heightinches', 'index_y', 'level_0', 'name', 'team_name','home_country',
                 'home_state', 'home_town', 'last_school', 'misc_ret', 'misc_ret_td', 'misc_ret_yard', 'def_2xp_att',
                 'def_2xp_made', 'fumble_lost', 'nflgrade', 'off_2xp_att', 'off_2xp_made', 'off_xp_kick_att',
                 'pass_conv', 'pick', 'pickround', 'picktotal', 'punt_yard', 'sack_yard', 'tackle_for_loss_yard',
                 'wonderlic', 'pass_att', 'field_goal_att']
    
    df = df.drop(drop_cols, axis=1)

    print("Complete.")
    
    return df


def engineer_features(df):

    print("Engineer additional features...")
    
    # Combine height / weight fields
    df['height'] = df['height'].fillna(df['heightinchestotal'])
    df['weight'] = df['weight_x'].fillna(df['weight_y'])
    df['college'] = df['team_name'].fillna(df['college'])
    
    # Add side of ball
    df['side_of_ball'] = df['position'].map(side_of_ball())

    # Completion percentage
    df['pass_comp_percentage'] = df['pass_comp']*1.0 / df['pass_att']

    # Field goal percentage
    df['field_goal_percentage'] = df['field_goal_made']*1.0 / df['field_goal_att']

    # Extra point percentage
    df['xp_percentage'] = df['off_xp_kick_made']*1.0 / df['off_xp_kick_att']

    # Rushing yards per carry
    df['rushing_ypc'] = df['rush_yard']*1.0 / df['rush_att']

    # Receiving yards per catch
    df['receiving_ypc'] = df['rec_yards']*1.0 / df['rec']

    # Pass yards per completion
    df['passing_ypc'] = df['pass_comp']*1.0 / df['pass_yard']

    print("Complete.")
    
    return df


def side_of_ball():
    side = pd.Series({
        'offense': ['FB', 'TE', 'WR', 'OL', 'OG', 'OT', 'C', 'QB', 'RB', 'HB', 'SE', 'TB', 'FL', 'OC', 'SB'],
        'defense': ['DT', 'S', 'CB', 'LB', 'DL', 'DB', 'OLB', 'DE', 'DS', 'FS', 'SS', 'WLB', 'NT', 'ILB', 'MLB',
                    'SLB', 'RV', 'NG', 'ROV'],
        'special_teams': ['K', 'PK', 'P', 'LS', 'SN'],
        'other': ['HOLD', 'ATH']
        })

    position_map = {}
    for k, v in side.iteritems():
        for i in v:
            position_map.update({i: k})

    return position_map


def roll_up_positions():
    position = pd.Series({
        'OL': ['OT', 'OG', 'C', 'OL'],
        'WR': ['WR', 'SE', 'FL'],
        'QB': ['QB'],
        'RB': ['RB', 'HB', 'TB', 'SB'],
        'TE': ['TE'],
        'FB': ['FB'],
        'DE': ['DE'],
        'DT': ['DT', 'DL', 'NG', 'NT'],
        'ILB': ['ILB', 'LB', 'MLB'],
        'OLB': ['OLB', 'SLB', 'WLB'],
        'CB': ['CB', 'DB'],
        'FS': ['FS', 'ROV'],
        'SS': ['SS', 'DS', 'S'],
        'K': ['K', 'PK'],
        'P': ['P'],
        'LS': ['LS', 'SN'],
        'OTH': ['HOLD', 'ATH']
        })

    position_map = {}
    for k, v in position.iteritems():
        for i in v:
            position_map.update({i: k})

    return position_map


def position_group():
    position = pd.Series({
        'OL': ['OT', 'OG', 'C', 'OL'],
        'WR': ['WR', 'SE', 'FL'],
        'QB': ['QB'],
        'RB': ['RB', 'HB', 'TB', 'SB', 'FB'],
        'TE': ['TE'],
        'DL': ['DE', 'DT', 'DL', 'NG', 'NT'],
        'LB': ['ILB', 'LB', 'MLB', 'OLB', 'SLB', 'WLB'],
        'DB': ['CB', 'DB', 'FS', 'ROV', 'SS', 'DS', 'S'],
        'ST': ['K', 'PK', 'P', 'LS', 'SN'],
        'OTH': ['HOLD', 'ATH']
        })

    position_map = {}
    for k, v in position.iteritems():
        for i in v:
            position_map.update({i: k})

    return position_map




if __name__ == "__main__":
    main()
