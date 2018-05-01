from sqlalchemy import create_engine
import config       # Credentials stored in config.py in working directory
import pandas as pd


def main():

    # Define postgres engine
    engine = create_engine('postgresql://{}:{}@localhost:5432/DataSets'.format(config.username, config.password))

    # Years of available data
    years = range(2005, 2014)

    # Aggregate career stats
    aggregate_careers(years, engine)

    # Consolidate player demographics
    roll_up_demographics(years, engine)

    # Combine stats and demographics
    add_player_demographics(engine)


def aggregate_year(year, engine):

    # Load game statistics
    stats = pd.read_sql_table("player_game_stats_{}".format(year), engine)

    # Fix column names
    stats.columns = [c.replace('/', '_') for c in stats.columns]
    stats.columns = [c.replace('-', '_') for c in stats.columns]

    # Drop game_code column
    stats = stats.drop(['game_code'], axis=1)

    # Roll stats up to yearly aggregates
    yearly_stats = stats.groupby(by='player_code', as_index=False).sum()

    return yearly_stats


def aggregate_careers(years, engine):

    for year in years:
        if year == years[0]:
            # Aggregate yearly stats
            df = aggregate_year(year, engine)

            df['year'] = year

            # Send to database
            df.to_sql("player_career_stats", engine, if_exists='replace')

            print('{} added to player_career_stats'.format(year))

        else:
            # Pull stats to date
            current = pd.read_sql_table("player_career_stats", engine)

            # Pull this year's stats
            addl = aggregate_year(year, engine)
            addl['year'] = year

            # Find max year
            y = pd.concat([current[['player_code', 'year']], addl[['player_code', 'year']]], ignore_index=True)
            y = y.groupby(by='player_code', as_index=False).max()

            # Concat data frames
            new = pd.concat([current, addl], ignore_index=True)

            # Re-aggregate
            new = new.groupby(by='player_code', as_index=False).sum()

            # Assign year
            new.update(y)

            # Send to database
            new.to_sql("player_career_stats", engine, index=False, if_exists='replace')

            print('{} added to player_career_stats'.format(year))

    print('player_career_stats table generated')


def roll_up_demographics(years, engine):

    demo = pd.DataFrame()

    for year in years:
        # Pull players for year
        year_df = pd.read_sql_table("player_{}".format(year), engine)

        # Concatenate year into demo
        demo = pd.concat([demo, year_df])

        print('{} added'.format(year))

    # Rank class
    rank = {'FR': 1, 'SO': 2, 'JR': 3, 'SR': 4}

    demo['class_rank'] = demo['class'].map(rank)

    # Keep only last year played
    demo = demo.sort_values(['player_code', 'class_rank'], na_position='first')

    demo = demo.drop_duplicates(subset='player_code', keep='last')

    # Bring in team info
    team = pd.read_sql_table('team', engine)

    demo = demo.merge(team, on='team_code', how='left')

    # Bring in conference info
    conf = pd.read_sql_table('conference', engine)

    demo = demo.merge(conf, on='conference_code', how='left')

    # Clean up columns
    demo = demo.rename(columns={'name_x': 'team_name', 'name_y': 'conference'})

    drop_cols = ['class_rank', 'index_x', 'index_y', 'index', 'team_code', 'uniform_number',
                 'conference_code', 'index']

    demo = demo.drop(drop_cols, axis=1)

    # Send to database
    demo.to_sql("player_info", engine, if_exists='replace')

    print('player_info table generated')


def add_player_demographics(engine):

    # Pull career stats
    stats = pd.read_sql_table("player_career_stats", engine)

    # Pull demographics
    demo = pd.read_sql_table("player_info", engine)

    # Combine tables
    comb = demo.merge(stats, on='player_code')

    # Send to database
    comb.to_sql('player_stats_combined', engine, index=False, if_exists='replace')

    print('players_stats_combined table generated')


if __name__ == "__main__":
    main()
