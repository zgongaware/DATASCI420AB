# DATASCI420AB
Course Project for DATASCI 420 AB

**Description**

What are the most predictive features of a given player to determine what round of
the NFL draft he'll be selected in?

**Requirements**

Data stored in local PostgreSQL database.  Credentials for database stored in
local (uncommitted) *config.py* file.  

Format of *config.py* -

username=''

password=''

**Data Sources:**
- [Combine / Draft Data](http://nflsavant.com/about.php)
- [College Player Statistics by Game](https://www.kaggle.com/mhixon/college-football-statistics)

Script Order:
- sql_loader_game_stats.py
- sql_loader_combine_stats.py
- player_stats_aggregator.py
- combine_data.py
