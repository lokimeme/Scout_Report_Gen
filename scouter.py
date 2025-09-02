import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import os
from scipy.stats import percentileofscore

class PlayerStats:
    def __init__(self, name, pos, ppg, apg, tov, three_p, three_pa, rpg, spg, bpg):
        self.name = name
        self.position = pos
        self.points_per_game = ppg
        self.assists_per_game = apg
        self.turnovers_per_game = tov
        self.three_point_percentage = three_p
        self.three_point_attempts_per_game = three_pa
        self.rebounds_per_game = rpg
        self.steals_per_game = spg
        self.blocks_per_game = bpg

def get_player_stats_df(year=2024):
    cache_filename = f"nba_stats_{year}.csv"

    if os.path.exists(cache_filename):
        print(f"Loading data from local cache: {cache_filename}")
        try:
            df = pd.read_csv(cache_filename)
            return df
        except Exception as e:
            print(f"Error reading cache file: {e}. Will re-scrape the data.")

    print(f"No local cache found for {year}. Scraping data from Basketball-Reference...")
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_html(io.StringIO(response.text), attrs={'id': 'per_game_stats'})[0]
        
        df = df.drop(df[df['Player'] == 'Player'].index)
        df = df.fillna(0)
        # Add 'Pos' to the columns being processed
        stat_cols = ['PTS', 'AST', 'TOV', '3P%', '3PA', 'TRB', 'STL', 'BLK', 'MP']
        df[stat_cols] = df[stat_cols].apply(pd.to_numeric, errors='coerce')

        print(f"Saving data to cache: {cache_filename}")
        df.to_csv(cache_filename, index=False)

        return df
    except Exception as e:
        print(f"Error scraping data: {e}")
        return None

def calculate_percentile_rank(full_df, player_series, stat_col):
    position = player_series['Pos'].split('-')[0] # Handles positions like 'C-PF'
    
    # Filter for players at the same position who have played meaningful minutes
    positional_df = full_df[(full_df['Pos'].str.contains(position)) & (full_df['MP'] > 10)]
    
    if positional_df.empty:
        return 50 # Return a default value if no comparable players are found

    player_stat_value = player_series[stat_col]
    
    # Get all the stat values for that position
    positional_stats = positional_df[stat_col].values
    
    # Calculate the percentile
    percentile = percentileofscore(positional_stats, player_stat_value)
    
    return int(percentile)

def generate_scouting_report(player: PlayerStats, full_df: pd.DataFrame, player_series: pd.Series) -> str:
    report_lines = []
    
    report_lines.append("========================================")
    report_lines.append(f"SCOUTING REPORT: {player.name} ({player.position})")
    report_lines.append("========================================")
    
    # --- Offensive Analysis ---
    report_lines.append("\nOFFENSE:")
    scoring_percentile = calculate_percentile_rank(full_df, player_series, 'PTS')
    if scoring_percentile >= 90:
        report_lines.append(f"- Elite scorer, ranking in the {scoring_percentile}th percentile for his position.")
    elif scoring_percentile >= 70:
        report_lines.append(f"- Strong scoring option ({scoring_percentile}th percentile for his position).")
    else:
        report_lines.append(f"- Not a primary scorer ({scoring_percentile}th percentile).")

    if player.three_point_percentage >= 0.38 and player.three_point_attempts_per_game >= 5:
        report_lines.append("- A lethal outside shooter on high volume.")
    elif player.three_point_percentage >= 0.36:
        report_lines.append("- A capable and efficient shooter from distance.")
    else:
        report_lines.append("- Not a significant threat from three-point range.")
        
    # --- Playmaking Analysis ---
    report_lines.append("\nPLAYMAKING:")
    assist_percentile = calculate_percentile_rank(full_df, player_series, 'AST')
    if assist_percentile >= 85:
        report_lines.append(f"- Exceptional playmaker ({assist_percentile}th percentile in assists for his position).")
    elif assist_percentile >= 60:
         report_lines.append(f"- Good facilitator ({assist_percentile}th percentile in assists).")
    else:
        report_lines.append(f"- Primarily looks for his own shot ({assist_percentile}th percentile in assists).")

    # --- Rebounding & Defense ---
    report_lines.append("\nDEFENSE & REBOUNDING:")
    rebound_percentile = calculate_percentile_rank(full_df, player_series, 'TRB')
    if rebound_percentile >= 90:
        report_lines.append(f"- Dominant force on the glass ({rebound_percentile}th percentile for his position).")
    elif rebound_percentile >= 70:
        report_lines.append(f"- Strong positional rebounder ({rebound_percentile}th percentile).")
    else:
        report_lines.append(f"- Average rebounder for his position ({rebound_percentile}th percentile).")

    report_lines.append("\n--- END OF REPORT ---\n")
    return "\n".join(report_lines)

def main():
    """
    Main function to drive the user interaction and report generation.
    """
    selected_year = 0
    while True:
        try:
            year_input = input("Enter the season year to analyze (e.g., 2024 for the 2023-24 season): ")
            selected_year = int(year_input)
            if 1950 <= selected_year <= 2025:
                break
            else:
                print("Please enter a year between 1950 and 2025.")
        except ValueError:
            print("Invalid input. Please enter a valid year.")

    print(f"\n--- Loading data for the {selected_year-1}-{selected_year % 100} season ---")
    stats_df = get_player_stats_df(year=selected_year)
    
    if stats_df is None:
        print("Could not retrieve player data. Exiting.")
        return

    print(f"--- Data loaded successfully ---\n")

    while True:
        choice = input("Would you like to find a 'player' or a 'team'? (or type 'quit'): ").lower()

        if choice == 'quit':
            break
        
        elif choice == 'player':
            player_name = input("Enter the player's name: ")
            player_data = stats_df[stats_df['Player'].str.lower() == player_name.lower()]
            
            if not player_data.empty:
                p_series = None
                if 'TOT' in player_data['Tm'].values:
                    p_series = player_data[player_data['Tm'] == 'TOT'].iloc[0]
                else:
                    p_series = player_data.iloc[0]

                player_obj = PlayerStats(
                    name=p_series['Player'],
                    pos=p_series['Pos'],
                    ppg=p_series['PTS'],
                    apg=p_series['AST'],
                    tov=p_series['TOV'],
                    three_p=p_series['3P%'],
                    three_pa=p_series['3PA'],
                    rpg=p_series['TRB'],
                    spg=p_series['STL'],
                    bpg=p_series['BLK']
                )
                # Pass the full DataFrame and the specific player's data series to the report generator
                print(generate_scouting_report(player_obj, stats_df, p_series))
            else:
                print(f"Player '{player_name}' not found.")

        elif choice == 'team':
            team_abbr = input("Enter the 3-letter team abbreviation (e.g., LAL, GSW): ").upper()
            team_df = stats_df[stats_df['Tm'] == team_abbr].sort_values(by='MP', ascending=False)

            if not team_df.empty:
                unit_choice = input("Do you want the 'starters' or the 'bench'? ").lower()
                
                player_list_df = pd.DataFrame()
                if unit_choice == 'starters':
                    player_list_df = team_df.head(5)
                elif unit_choice == 'bench':
                    player_list_df = team_df.iloc[5:]
                else:
                    print("Invalid choice. Please enter 'starters' or 'bench'.")
                    continue
                
                for index, p_series in player_list_df.iterrows():
                    player_obj = PlayerStats(
                        name=p_series['Player'],
                        pos=p_series['Pos'],
                        ppg=p_series['PTS'],
                        apg=p_series['AST'],
                        tov=p_series['TOV'],
                        three_p=p_series['3P%'],
                        three_pa=p_series['3PA'],
                        rpg=p_series['TRB'],
                        spg=p_series['STL'],
                        bpg=p_series['BLK']
                    )
                    print(generate_scouting_report(player_obj, stats_df, p_series))
            else:
                print(f"Team '{team_abbr}' not found.")
        
        else:
            print("Invalid input. Please enter 'player', 'team', or 'quit'.")

if __name__ == "__main__":
    main()


