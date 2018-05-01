from sqlalchemy import create_engine
import config       # Credentials stored in config.py in working directory
import pandas as pd


def main():

    # Define postgres engine
    engine = create_engine('postgresql://{}:{}@localhost:5432/DataSets'.format(config.username, config.password))

    # Merge data
    both = merge_data(engine)

    # Create additional features
    both = engineer_features(both)

    # Drop extra columns
    both = drop_columns(both)

    # Drop duplicates
    both = drop_duplicates(both)

    # Send to database
    send_to_database(both, engine)


def drop_duplicates(df):
    # Drop duplicates
    df = df.drop_duplicates(subset=['first_name', 'last_name', 'position', 'year'], keep='last')

    return df


def send_to_database(df, engine):

    print("Populate players_final table...")

    df.to_sql('players_final', engine, index=False, if_exists='replace')

    print("players_final loaded")


def merge_data(engine):

    print("Load and combine data sources...")

    # Years of available data
    years = range(2005, 2014)

    # Load game stats table and format
    stats = pd.read_sql_table('player_stats_combined', engine)
    stats = stats.dropna(axis=0, how='any', subset=['first_name', 'last_name', 'position'])

    stats['first_name'] = stats['first_name'].apply(lambda x: x.lower())
    stats['last_name'] = stats['last_name'].apply(lambda x: x.lower())
    stats['year'] = stats['year'].apply(lambda x: x+1)
    stats['position'] = stats['position'].map(roll_up_positions())

    # Load combine stats table and format
    comb = pd.read_sql_table('combine_data', engine)
    comb = comb[comb['year'].isin(years)]
    comb['first_name'] = comb['first_name'].apply(lambda x: x.lower())
    comb['last_name'] = comb['last_name'].apply(lambda x: x.lower())
    comb['position'] = comb['position'].map(roll_up_positions())

    # Merge as best we can
    both = stats.merge(comb, on=('first_name', 'last_name', 'position', 'year'), how='outer')

    # Try again
    both = both.rename(columns={'weight_x': 'weight'})

    retry = both[both['player_code'].isnull()]
    retry = retry.loc[:, comb.columns]

    retry = retry.merge(stats, on=('first_name', 'last_name', 'position'), how='inner')

    retry = retry.drop(['year_x', 'weight_y'], axis=1)
    retry = retry.rename(columns={'year_y': 'year'})

    again = pd.concat([both, retry])
    again = again.sort_values(by=['first_name', 'last_name', 'position', 'year', 'round'], na_position='first')

    print("Complete.")

    return again


def drop_columns(df):

    print("Remove extraneous columns...")

    # Drop extra columns
    drop_cols = ['index_x', 'weight_x', 'weight_y', 'Unnamed: 26', 'Unnamed: 27', 'heightinchestotal',
                 'heightfeet', 'heightinches', 'index_y', 'index', 'level_0', 'name', 'team_name','home_country',
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
        'OT': ['OT', 'OG', 'C', 'OL'],
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


if __name__ == "__main__":
    main()
