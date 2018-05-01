from sqlalchemy import create_engine
# import config       # Credentials stored in config.py in working directory
import pandas as pd
import pandas_profiling


def main():

    # Define postgres engine
    # engine = create_engine('postgresql://{}:{}@localhost:5432/DataSets'.format(config.username, config.password))
    engine = create_engine('postgresql://postgres:7red5221@localhost:5432/DataSets')

    # Merge data
    both = merge_data(engine)

    # Create additional features
    both = engineer_features(both)

    # Drop extra columns
    both = drop_columns(both)

    # Send to database
    #both.to_sql('players_final', engine, index=False, if_exists='replace')


def merge_data(engine):

    # Years of available data
    years = range(2005, 2014)

    # Load game stats table and format
    stats = pd.read_sql_table('player_stats_combined', engine)
    stats = stats.dropna(axis=0, how='any', subset=['first_name', 'last_name', 'position'])

    stats['first_name'] = stats['first_name'].apply(lambda x: x.lower())
    stats['last_name'] = stats['last_name'].apply(lambda x: x.lower())
    stats['year'] = stats['year'].apply(lambda x: x+1)

    # Load combine stats table and format
    comb = pd.read_sql_table('combine_data', engine)
    comb = comb[comb['year'].isin(years)]
    comb['first_name'] = comb['first_name'].apply(lambda x: x.lower())
    comb['last_name'] = comb['last_name'].apply(lambda x: x.lower())

    # Merge as best we can
    both = stats.merge(comb, on=('first_name', 'last_name', 'position', 'year'), how='outer')

    # Drop duplicates
    both = both.drop_duplicates(subset=['first_name', 'last_name', 'position', 'year'], keep='last')

    return both


def drop_columns(df):
    # Drop extra columns
    drop_cols = ['index_x', 'weight_x', 'weight_y', 'Unnamed: 26', 'Unnamed: 27', 'heightinchestotal',
                 'heightfeet', 'heightinches', 'index_y', 'index', 'level_0', 'name', 'team_name','home_country',
                 'home_state', 'home_town', 'last_school', 'misc_ret', 'misc_ret_td', 'misc_ret_yard', 'def_2xp_att',
                 'def_2xp_made', 'fumble_lost', 'nflgrade', 'off_2xp_att', 'off_2xp_made', 'off_xp_kick_att',
                 'pass_conv', 'pick', 'pickround', 'picktotal', 'punt_yard', 'sack_yard', 'tack_for_loss_yard', 
                 'wonderlic', 'pass_att', 'field_goal_att']
    
    df = df.drop(drop_cols, axis=1)
    
    return df


def engineer_features(df):
    
    # Combine height / weight fields
    df['height'] = df['height'].fillna(df['heightinchestotal'])
    df['weight'] = df['weight_x'].fillna(df['weight_y'])
    df['college'] = df['team_name'].fillna(df['college'])
    
    # Add side of ball
    df['side_of_ball'] = df['position'].map(side_of_ball())
    
    return df


def side_of_ball():
    side = pd.Series({
        'offense': ['FB', 'TE', 'WR', 'OL', 'OG', 'OT', 'C', 'QB', 'RB', 'HB', 'SE', 'TB', 'FL', 'OC', 'SB'],
        'defense': ['DT', 'S', 'CB', 'LB', 'DL', 'DB', 'OLB', 'DE', 'DS', 'FS', 'SS', 'WLB', 'NT', 'ILB', 'MLB',
                    'SLB', 'RV', 'NG', 'ROV', 'SN'],
        'special_teams': ['K', 'PK', 'P', 'LS'],
        'other': ['HOLD', 'ATH']
        })

    position_map = {}
    for k, v in side.iteritems():
        for i in v:
            position_map.update({i:k})

    return position_map


if __name__ == "__main__":
    main()
