from sqlalchemy import create_engine
import config       # Credentials stored in config.py in working directory
import pandas as pd


def main():

    # Define postgres engine
    engine = create_engine('postgresql://{}:{}@localhost:5432/DataSets'.format(config.username, config.password))

    # Years of available data
    years = range(2005, 2014)

    for year in years:

        # Load csv into data frame
        df = pd.read_csv('college-football-statistics\cfbstats-com-{}-1-5-0\player.csv'.format(year))

        # Remove capitals and spaces from column names
        df.columns = [c.lower() for c in df.columns]
        df.columns = [c.replace(' ', '_') for c in df.columns]

        # Load to SQL table
        df.to_sql("player_{}".format(year), engine)

        print('{} complete'.format(year))

    # Load team data
    team = pd.read_csv("Course Project\college-football-statistics\cfbstats-com-2013-1-5-0\\team.csv")

    # Remove capitals and spaces from column names
    team.columns = [c.lower() for c in team.columns]
    team.columns = [c.replace(' ', '_') for c in team.columns]

    # Load to SQL table
    team.to_sql("team", engine)

    # Load conference data
    conf = pd.read_csv('Course Project\college-football-statistics\cfbstats-com-2013-1-5-0\conference.csv')

    # Remove capitals and spaces from column names
    conf.columns = [c.lower() for c in conf.columns]
    conf.columns = [c.replace(' ', '_') for c in conf.columns]

    # Load to SQL table
    conf.to_sql("conference", engine)


if __name__ == "__main__":
    main()
