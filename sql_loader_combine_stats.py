from sqlalchemy import create_engine
import config       # Credentials stored in config.py in working directory
import pandas as pd


def main():

    # Define postgres engine
    engine = create_engine('postgresql://{}:{}@localhost:5432/DataSets'.format(config.username, config.password))

    # Load draft rounds
    load_draft_rounds(engine)

    # Load combine data
    load_combine_data(engine)


def load_draft_rounds(engine):
    # Read CSV
    draft = pd.read_csv('draft_rounds.csv')

    # Load to database
    draft.to_sql('draft_rounds', engine, if_exists='replace')

    print('draft_rounds table populated')


def load_combine_data(engine):
    # Read CSV
    comb = pd.read_csv('combine.csv')

    comb = comb.rename(columns={'firstname': 'first_name', 'lastname': 'last_name'})

    # Load to database
    comb.to_sql('combine_data', engine, if_exists='replace')

    print('Combine data loaded.')


if __name__ == "__main__":
    main()
