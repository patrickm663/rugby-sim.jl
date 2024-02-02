# Data
The data is obtained from [Global Rugby Results](https://www.globalrugbyresults.com/2022UnitedRugbyChampionship.html) and saved in `urc_matches_2022_2023.csv`.

Below is the list of fields:

- **Match**: ID for the game
- **Round**: Round the match took place
- **Home_Team**: Home team name
- **Away_Team**: Away team name
- **Home_Score**: Points scored by the home team
- **Away_Score**: Points scored by the away team
- **Diff**: Points difference (home - away)
- **Home_Win**: Indicator for a Home win (worth 4 points)
- **Away_Win**: Indicator for an Away win (worth 4 points)
- **Draw**: Indicator for a Draw (worth 2 points)
- **Home_Country**: Country the Home team is from (Italy, Ireland, Scotland, South Africa, or Wales)
- **Away_Country**: Country the Awat team is from (Italy, Ireland, Scotland, South Africa, or Wales)

In total, it comprises 144 regular season games, 4 Quarter Finals, 2 Semi-Finals, and 1 Final. 16 teams take part, playing 18 regular season games.

For the simulator, the target is the point difference in a given match, which is used to determine win status.

**Note: there is no Bonus Point data currently (awarded if a team scores at least 4 tries in a match), but may be included in future iterations.**
