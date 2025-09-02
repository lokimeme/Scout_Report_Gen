import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import os 

class PlayerStats:
    def __init__(self, name, ppg, apg, tov, three_p, three_pa, rpg, spg, bpg):
        self.name = name
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

    # Check if the cached file already exists
    if os.path.exists(cache_filename):
        print(f"Loading data from local cache: {cache_filename}")
        try:
            # If it exists, load it directly from the CSV and return
            df = pd.read_csv(cache_filename)
            return df
        except Exception as e:
            print(f"Error reading cache file: {e}. Will re-scrape the data.")

    # If cache doesn't exist, proceed with scraping
    print(f"No local cache found for {year}. Scraping data from Basketball-Reference...")
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_html(io.StringIO(response.text), attrs={'id': 'per_game_stats'})[0]
        
        # Data Cleaning
        df = df.drop(df[df['Player'] == 'Player'].index)
        df = df.fillna(0)
        stat_cols = ['PTS', 'AST', 'TOV', '3P%', '3PA', 'TRB', 'STL', 'BLK', 'MP']
        for col in stat_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.drop_duplicates(subset='Player', keep='first')

        print(f"Saving data to cache: {cache_filename}")
        df.to_csv(cache_filename, index=False)

        return df
    except Exception as e:
        print(f"Error scraping data: {e}")
        return None

def generate_scouting_report(player: PlayerStats) -> str:
    report_lines = []
    
    report_lines.append("========================================")
    report_lines.append(f"SCOUTING REPORT: {player.name}")
    report_lines.append("========================================")
    
    # Offensive Analysis
    report_lines.append("\nOFFENSE:")
    if player.points_per_game >= 25.0:
        report_lines.append("- An elite, go-to scoring option.")
    elif player.points_per_game >= 18.0:
        report_lines.append("- A strong and reliable secondary scorer.")
    else:
        report_lines.append("- Primarily a role player on offense.")

    if player.three_point_percentage >= 0.38 and player.three_point_attempts_per_game >= 5:
        report_lines.append("- A lethal outside shooter on high volume.")
    elif player.three_point_percentage >= 0.36:
        report_lines.append("- A capable and efficient shooter from distance.")
    else:
        report_lines.append("- Not a significant threat from three-point range.")
        
    # Playmaking Analysis
    report_lines.append("\nPLAYMAKING:")
    if player.turnovers_per_game > 0:
        atr = player.assists_per_game / player.turnovers_per_game
        if atr >= 2.5:
            report_lines.append("- Excellent decision-maker who protects the ball.")
        elif atr >= 1.5:
            report_lines.append("- Solid playmaker, but can be prone to mistakes.")
        else:
            report_lines.append("- Struggles with turnovers; not a primary creator.")
    elif player.assists_per_game > 3.0:
        report_lines.append("- Incredibly safe and effective playmaker.")
    else:
        report_lines.append("- Not a primary ball-handler.")

    # Rebounding & Defense
    report_lines.append("\nDEFENSE & REBOUNDING:")
    if player.rebounds_per_game >= 10.0:
        report_lines.append("- Dominant force on the glass.")
    elif player.rebounds_per_game >= 7.0:
        report_lines.append("- Strong positional rebounder.")
    else:
        report_lines.append("- Average rebounder for his position.")
    
    combined_stocks = player.steals_per_game + player.blocks_per_game
    if combined_stocks >= 3.0:
        report_lines.append("- Elite, game-changing defensive playmaker.")
    elif combined_stocks >= 1.5:
        report_lines.append("- Positive contributor on the defensive end.")
    else:
        report_lines.append("- Not a major factor in creating turnovers or protecting the rim.")

    report_lines.append("\n--- END OF REPORT ---\n")
    return "\n".join(report_lines)

def main():
    selected_year = 0
    while True:
        try:
            year_input = input("Enter the season year to analyze (e.g., 2024 for the 2023-24 season): ")
            selected_year = int(year_input)
            if 1950 <= selected_year <= 2025: # Basic validation for a reasonable year range
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
                p_series = player_data.iloc[0]
                player_obj = PlayerStats(
                    name=p_series['Player'],
                    ppg=p_series['PTS'],
                    apg=p_series['AST'],
                    tov=p_series['TOV'],
                    three_p=p_series['3P%'],
                    three_pa=p_series['3PA'],
                    rpg=p_series['TRB'],
                    spg=p_series['STL'],
                    bpg=p_series['BLK']
                )
                print(generate_scouting_report(player_obj))
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
                        ppg=p_series['PTS'],
                        apg=p_series['AST'],
                        tov=p_series['TOV'],
                        three_p=p_series['3P%'],
                        three_pa=p_series['3PA'],
                        rpg=p_series['TRB'],
                        spg=p_series['STL'],
                        bpg=p_series['BLK']
                    )
                    print(generate_scouting_report(player_obj))
            else:
                print(f"Team '{team_abbr}' not found.")
        
        else:
            print("Invalid input. Please enter 'player', 'team', or 'quit'.")

if __name__ == "__main__":
    main()


