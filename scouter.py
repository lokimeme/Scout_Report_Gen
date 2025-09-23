import pandas
import requests
from bs4 import BeautifulSoup, Comment
import io
import os
from scipy.stats import percentileofscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy

class PlayerStats:
    def __init__(self, name, pos, archetype, ppg, apg, tov, three_p, three_pa, rpg, spg, bpg, per, ts_percentage, win_shares, pct_fga_0_3, fg_pct_0_3, pct_fga_3_10, fg_pct_3_10, pct_fga_10_16, fg_pct_10_16, pct_fga_16_3p, fg_pct_16_3p, pct_fga_3p, pct_3pa_corner):
        self.name = name
        self.position = pos
        self.archetype = archetype
        self.points_per_game = ppg
        self.assists_per_game = apg
        self.turnovers_per_game = tov
        self.three_point_percentage = three_p
        self.three_point_attempts_per_game = three_pa
        self.rebounds_per_game = rpg
        self.steals_per_game = spg
        self.blocks_per_game = bpg
        self.per = per
        self.ts_percentage = ts_percentage
        self.win_shares = win_shares
        self.pct_fga_0_3 = pct_fga_0_3
        self.fg_pct_0_3 = fg_pct_0_3
        self.pct_fga_3_10 = pct_fga_3_10
        self.fg_pct_3_10 = fg_pct_3_10
        self.pct_fga_10_16 = pct_fga_10_16
        self.fg_pct_10_16 = fg_pct_10_16
        self.pct_fga_16_3p = pct_fga_16_3p
        self.fg_pct_16_3p = fg_pct_16_3p
        self.pct_fga_3p = pct_fga_3p
        self.pct_3pa_corner = pct_3pa_corner


def get_player_stats_df(year=2024):
    cache_filename = f"nba_stats_{year}.csv"

    if os.path.exists(cache_filename):
        print(f"Loading per-game data from local cache: {cache_filename}")
        try:
            df = pandas.read_csv(cache_filename)
            if 'Tm' not in df.columns or 'Age' not in df.columns:
                 raise ValueError("Cache file is missing required columns.")
            return df
        except Exception as e:
            print(f"Error reading cache file or cache is invalid: {e}. Deleting and re-scraping.")
            os.remove(cache_filename)

    print(f"No local cache found for {year}. Scraping per-game data from Basketball-Reference...")
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
        response = requests.get(url)
        response.raise_for_status()
        
        df = pandas.read_html(io.StringIO(response.text), attrs={'id': 'per_game_stats'})[0]
        
        if isinstance(df.columns, pandas.MultiIndex):
            df.columns = df.columns.droplevel(0)

        df = df.drop(df[df['Player'] == 'Player'].index)
        
        if 'Team' in df.columns:
            df = df.rename(columns={'Team': 'Tm'})
            
        required_cols = ['Player', 'Tm', 'Age']
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"Scraped data is missing one of the required columns: {required_cols}. Columns found: {df.columns.tolist()}")

        df = df.fillna(0)
        df['Age'] = pandas.to_numeric(df['Age'], errors='coerce')
        stat_cols = ['PTS', 'AST', 'TOV', '3P%', '3PA', 'TRB', 'STL', 'BLK', 'MP', 'FGA']
        df[stat_cols] = df[stat_cols].apply(pandas.to_numeric, errors='coerce')

        print(f"Saving per-game data to cache: {cache_filename}")
        df.to_csv(cache_filename, index=False)

        return df
    except Exception as e:
        print(f"Error during scraping or data processing: {e}")
        return None

def get_advanced_stats_df(year=2024):
    cache_filename = f"nba_advanced_stats_{year}.csv"
    
    if os.path.exists(cache_filename):
        print(f"Loading advanced data from local cache: {cache_filename}")
        try:
            df = pandas.read_csv(cache_filename)
            if 'Tm' not in df.columns or 'Age' not in df.columns:
                raise ValueError("Advanced stats cache is missing required columns.")
            return df
        except Exception as e:
            print(f"Error reading cache file or cache is invalid: {e}. Deleting and re-scraping.")
            os.remove(cache_filename)
            
    print(f"No local cache found for {year}. Scraping advanced data from Basketball-Reference...")
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        response = requests.get(url)
        response.raise_for_status()

        html_text = response.text.replace('<!--', '').replace('-->', '')
        
        df = pandas.read_html(io.StringIO(html_text), attrs={'id': 'advanced_stats'})[0]

        if isinstance(df.columns, pandas.MultiIndex):
            df.columns = df.columns.droplevel(0)
            
        df = df.drop(df[df['Player'] == 'Player'].index).reset_index(drop=True)
        
        if 'Team' in df.columns:
            df = df.rename(columns={'Team': 'Tm'})

        df = df.fillna(0)
        df['Age'] = pandas.to_numeric(df['Age'], errors='coerce')
        stat_cols = ['PER', 'TS%', 'WS']
        df[stat_cols] = df[stat_cols].apply(pandas.to_numeric, errors='coerce')

        print(f"Saving advanced data to cache: {cache_filename}")
        df.to_csv(cache_filename, index=False)

        return df
    except Exception as e:
        print(f"An unexpected error occurred in get_advanced_stats_df: {e}")
        return None

def get_shooting_stats_df(year=2024):
    cache_filename = f"nba_shooting_stats_{year}.csv"
    
    if os.path.exists(cache_filename):
        print(f"Loading shooting data from local cache: {cache_filename}")
        try:
            df = pandas.read_csv(cache_filename)
            if 'Tm' not in df.columns or 'Age' not in df.columns:
                raise ValueError("Shooting stats cache is missing required columns.")
            return df
        except Exception as e:
            print(f"Error reading cache file or cache is invalid: {e}. Deleting and re-scraping.")
            os.remove(cache_filename)
            
    print(f"No local cache found for {year}. Scraping shooting data from Basketball-Reference...")
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_shooting.html"
        response = requests.get(url)
        response.raise_for_status()

        html_text = response.text.replace('<!--', '').replace('-->', '')
        
        df = pandas.read_html(io.StringIO(html_text), attrs={'id': 'shooting_stats'})[0]

        if isinstance(df.columns, pandas.MultiIndex):
            new_cols = []
            for col in df.columns:
                if 'Unnamed' in col[0]:
                    new_cols.append(col[1])
                else:
                    new_cols.append(f"{col[0]}_{col[1]}")
            df.columns = new_cols
        
        df = df.drop(df[df['Player'] == 'Player'].index).reset_index(drop=True)
        
        if 'Team' in df.columns:
            df = df.rename(columns={'Team': 'Tm'})

        df = df.fillna(0)
        df['Age'] = pandas.to_numeric(df['Age'], errors='coerce')
        
        stat_cols = [col for col in df.columns if '%' in col or 'Dist.' in col]
        df[stat_cols] = df[stat_cols].apply(pandas.to_numeric, errors='coerce')

        print(f"Saving shooting data to cache: {cache_filename}")
        df.to_csv(cache_filename, index=False)

        return df
    except Exception as e:
        print(f"An unexpected error occurred in get_shooting_stats_df: {e}")
        return None

def add_player_archetypes(df):
    stats_for_clustering = ['PTS', 'AST', 'TRB', 'STL', 'BLK', '3PA', 'FGA']
    
    df_filtered = df[df['MP'] >= 15].copy()
    
    if df_filtered.empty:
        df['Archetype'] = 'Uncategorized'
        return df

    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(df_filtered[stats_for_clustering])
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df_filtered['Cluster'] = kmeans.fit_predict(scaled_stats)
    
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_profiles = pandas.DataFrame(cluster_centers, columns=stats_for_clustering)
    
    archetype_map = {}
    for i, row in cluster_profiles.iterrows():
        if row['PTS'] >= 20 and row['AST'] >= 5:
            archetype_map[i] = "Primary Offensive Engine"
        elif row['BLK'] >= 1.5 or row['TRB'] >= 10:
            archetype_map[i] = "Interior Defensive Anchor"
        elif row['AST'] >= 6:
            archetype_map[i] = "High-Volume Playmaker"
        elif row['PTS'] >= 18:
            archetype_map[i] = "Volume Scorer"
        elif row['3PA'] >= 5 and row['STL'] >= 1:
            archetype_map[i] = "3&D Wing"
        elif row['TRB'] >= 7 and row['PTS'] >= 10:
             archetype_map[i] = "Rebounding Forward"
        elif row['PTS'] >= 10 and row['AST'] >= 3:
            archetype_map[i] = "All-Around Contributor"
        else:
            archetype_map[i] = "Role Player"
            
    df_filtered['Archetype'] = df_filtered['Cluster'].map(archetype_map)
    
    archetype_series = df_filtered.set_index(['Player', 'Tm', 'Age'])['Archetype']

    df_indexed = df.set_index(['Player', 'Tm', 'Age'])
    
    df['Archetype'] = df_indexed.index.map(archetype_series)
    
    df['Archetype'] = df['Archetype'].fillna('Low-Minutes Player')
    
    return df

def calculate_league_percentile_rank(full_df, player_series, stat_col):
    comparison_df = full_df[full_df['MP'] > 10]

    if comparison_df.empty or len(comparison_df) < 2:
        return 50

    player_stat_value = player_series[stat_col]
    comparison_stats = comparison_df[stat_col].values
    percentile = percentileofscore(comparison_stats, player_stat_value)
    
    return int(percentile)

def calculate_positional_percentile_rank(full_df, player_series, stat_col):
    position = player_series['Pos'].split('-')[0]
    comparison_df = full_df[(full_df['Pos'].str.contains(position)) & (full_df['MP'] > 10)]

    if comparison_df.empty or len(comparison_df) < 2:
        return 50

    player_stat_value = player_series[stat_col]
    comparison_stats = comparison_df[stat_col].values
    percentile = percentileofscore(comparison_stats, player_stat_value)
    
    return int(percentile)


def generate_comparison_chart(player_series, full_df):
    archetype = player_series['Archetype']
    archetype_df = full_df[full_df['Archetype'] == archetype].copy()
    
    if archetype_df.empty:
        print("Cannot generate chart: No other players of this archetype found.")
        return
    
    rebound_percentile = calculate_positional_percentile_rank(full_df, player_series, 'TRB')
    assist_percentile = calculate_positional_percentile_rank(full_df, player_series, 'AST')

    if rebound_percentile > assist_percentile:
        x_stat, y_stat = 'TRB', 'PTS'
        x_label, y_label = 'Rebounds Per Game (RPG)', 'Points Per Game (PPG)'
        title_profile = 'Scoring & Rebounding Profile'
    else:
        x_stat, y_stat = 'AST', 'PTS'
        x_label, y_label = 'Assists Per Game (APG)', 'Points Per Game (PPG)'
        title_profile = 'Offensive Profile'

    top_players = archetype_df.sort_values(by=y_stat, ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(top_players[x_stat], top_players[y_stat], color='blue', label=f'Top 10 "{archetype}" Players')

    for i, player in top_players.iterrows():
        ax.annotate(player['Player'], (player[x_stat], player[y_stat]), xytext=(5,-5), textcoords='offset points')

    player_x = player_series[x_stat]
    player_y = player_series[y_stat]
    ax.scatter(player_x, player_y, color='red', s=150, zorder=5, label=player_series['Player'])
    ax.annotate(player_series['Player'], (player_x, player_y), xytext=(5,-5), textcoords='offset points', weight='bold')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_profile} for '{archetype}' Archetype")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    plt.savefig('scouting_report_chart.png')
    plt.close()
    print("\nComparison chart 'scouting_report_chart.png' has been saved.")

def generate_scouting_report(player: PlayerStats, full_df: pandas.DataFrame, player_series: pandas.Series) -> str:
    report_lines = []
    
    report_lines.append("========================================")
    report_lines.append(f"SCOUTING REPORT: {player.name} ({player.position})")
    report_lines.append(f"PROJECTED ARCHETYPE: {player.archetype}")
    report_lines.append("========================================")
    
    report_lines.append("\nOFFENSE:")
    scoring_percentile_league = calculate_league_percentile_rank(full_df, player_series, 'PTS')
    scoring_percentile_pos = calculate_positional_percentile_rank(full_df, player_series, 'PTS')
    if scoring_percentile_league >= 90:
        report_lines.append(f"- Elite scorer, ranking in the {scoring_percentile_league}th percentile among all players ({scoring_percentile_pos}th percentile for his position).")
    elif scoring_percentile_league >= 70:
        report_lines.append(f"- Strong scoring option ({scoring_percentile_league}th percentile among all players, {scoring_percentile_pos}th percentile for his position).")
    else:
        report_lines.append(f"- Not a primary scorer ({scoring_percentile_league}th percentile among all players, {scoring_percentile_pos}th percentile for his position).")

    if 'pct_fga_0_3' in player_series and player_series['pct_fga_0_3'] > 0:
        report_lines.append("\nSHOOTING PROFILE:")
        
        is_efficient_rim = player.fg_pct_0_3 > 0.65
        is_efficient_mid = player.fg_pct_10_16 > 0.43 or player.fg_pct_16_3p > 0.43
        is_efficient_3p = player.three_point_percentage > 0.36
        has_volume_mid = player.pct_fga_10_16 + player.pct_fga_16_3p > 0.20
        has_volume_3p = player.pct_fga_3p > 0.30

        if is_efficient_rim and is_efficient_mid and is_efficient_3p and has_volume_mid and has_volume_3p:
            report_lines.append("- A true three-level threat who can score efficiently from all over the floor.")

        if player.pct_fga_0_3 > 0.40:
            report_lines.append(f"- Heavily relies on attacking the rim, taking {player.pct_fga_0_3:.1%} of shots inside 3 feet.")
        elif player.pct_fga_3p > 0.50:
            report_lines.append(f"- A three-point specialist, taking {player.pct_fga_3p:.1%} of his shots from beyond the arc.")

        if player.fg_pct_0_3 > 0.70:
            report_lines.append("- An elite finisher at the basket.")
        elif player.fg_pct_0_3 < 0.55:
            report_lines.append("- Struggles to finish effectively at the rim.")

        mid_range_fga_pct = player.pct_fga_3_10 + player.pct_fga_10_16 + player.pct_fga_16_3p
        if mid_range_fga_pct > 0.35:
            mid_range_fg_pct_weighted = ((player.pct_fga_3_10 * player.fg_pct_3_10) + (player.pct_fga_10_16 * player.fg_pct_10_16) + (player.pct_fga_16_3p * player.fg_pct_16_3p)) / mid_range_fga_pct if mid_range_fga_pct > 0 else 0
            if mid_range_fg_pct_weighted > 0.44:
                report_lines.append("- A very effective mid-range shooter on high volume.")
            elif mid_range_fg_pct_weighted < 0.38:
                report_lines.append("- An inefficient scorer who settles for a lot of mid-range shots.")

        if player.pct_3pa_corner > 0.30:
            report_lines.append("- Often spots up in the corner, a key indicator of an off-ball role.")

        
    report_lines.append("\nPLAYMAKING:")
    assist_percentile_league = calculate_league_percentile_rank(full_df, player_series, 'AST')
    assist_percentile_pos = calculate_positional_percentile_rank(full_df, player_series, 'AST')
    if assist_percentile_league >= 90:
        report_lines.append(f"- Elite playmaker, ranking in the {assist_percentile_league}th percentile across the league ({assist_percentile_pos}th percentile for his position).")
    elif assist_percentile_league >= 70:
         report_lines.append(f"- A strong facilitator ({assist_percentile_league}th percentile across the league, {assist_percentile_pos}th percentile for his position).")
    else:
        report_lines.append(f"- Primarily looks for his own shot ({assist_percentile_league}th percentile in assists, {assist_percentile_pos}th percentile for his position).")

    report_lines.append("\nDEFENSE & REBOUNDING:")
    rebound_percentile_league = calculate_league_percentile_rank(full_df, player_series, 'TRB')
    rebound_percentile_pos = calculate_positional_percentile_rank(full_df, player_series, 'TRB')
    if rebound_percentile_league >= 90:
        report_lines.append(f"- Dominant force on the glass ({rebound_percentile_league}th percentile among all players, {rebound_percentile_pos}th percentile for his position).")
    elif rebound_percentile_league >= 70:
        report_lines.append(f"- A strong rebounder ({rebound_percentile_league}th percentile among all players, {rebound_percentile_pos}th percentile for his position).")
    else:
        report_lines.append(f"- An average rebounder ({rebound_percentile_league}th percentile among all players, {rebound_percentile_pos}th percentile for his position).")
    
    if player.per > 0 or player.win_shares > 0:
        report_lines.append("\nADVANCED METRICS:")
        if player.per > 25.0:
            report_lines.append(f"- MVP-level production with a PER of {player.per:.1f}.")
        elif player.per > 20.0:
            report_lines.append(f"- All-Star level impact with a PER of {player.per:.1f}.")
        elif player.per > 15.0:
            report_lines.append(f"- Solid rotation player with a PER of {player.per:.1f}.")
        elif player.per > 10.0:
            report_lines.append(f"- Fringe rotation player with a PER of {player.per:.1f}.")
        else:
            report_lines.append(f"- Below replacement-level player (PER of {player.per:.1f}).")

        if player.ts_percentage > 0.620:
            report_lines.append(f"- Elite, hyper-efficient scorer (TS% of {player.ts_percentage:.3f}).")
        elif player.ts_percentage > 0.600:
            report_lines.append(f"- Excellent efficiency for a scorer (TS% of {player.ts_percentage:.3f}).")
        elif player.ts_percentage > 0.570:
            report_lines.append(f"- Good, above-average efficiency (TS% of {player.ts_percentage:.3f}).")
        elif player.ts_percentage > 0.540:
            report_lines.append(f"- Solid, league-average efficiency (TS% of {player.ts_percentage:.3f}).")
        else:
            report_lines.append(f"- Below-average scorer in terms of efficiency (TS% of {player.ts_percentage:.3f}).")
            
        if player.win_shares > 12.0:
            report_lines.append(f"- MVP-caliber impact on winning ({player.win_shares:.1f} Win Shares).")
        elif player.win_shares > 10.0:
            report_lines.append(f"- All-NBA level contributor to team success ({player.win_shares:.1f} Win Shares).")
        elif player.win_shares > 5.0:
            report_lines.append(f"- A key, positive contributor to winning ({player.win_shares:.1f} Win Shares).")
        elif player.win_shares > 2.0:
            report_lines.append(f"- A solid, contributing rotation player ({player.win_shares:.1f} Win Shares).")
        else:
            report_lines.append(f"- A fringe player with limited impact on winning ({player.win_shares:.1f} Win Shares).")

    report_lines.append("\n--- END OF REPORT ---")
    return "\n".join(report_lines)

def process_player_selection(p_series, stats_df):
    player_obj = PlayerStats(
        name=p_series['Player'],
        pos=p_series['Pos'],
        archetype=p_series['Archetype'],
        ppg=p_series['PTS'],
        apg=p_series['AST'],
        tov=p_series['TOV'],
        three_p=p_series['3P%'],
        three_pa=p_series['3PA'],
        rpg=p_series['TRB'],
        spg=p_series['STL'],
        bpg=p_series['BLK'],
        per=p_series['PER'],
        ts_percentage=p_series['TS%'],
        win_shares=p_series['WS'],
        pct_fga_0_3=p_series.get('%FGA by Distance_0-3', 0),
        fg_pct_0_3=p_series.get('FG% by Distance_0-3', 0),
        pct_fga_3_10=p_series.get('%FGA by Distance_3-10', 0),
        fg_pct_3_10=p_series.get('FG% by Distance_3-10', 0),
        pct_fga_10_16=p_series.get('%FGA by Distance_10-16', 0),
        fg_pct_10_16=p_series.get('FG% by Distance_10-16', 0),
        pct_fga_16_3p=p_series.get('%FGA by Distance_16-3p', 0),
        fg_pct_16_3p=p_series.get('FG% by Distance_16-3p', 0),
        pct_fga_3p=p_series.get('%FGA by Distance_3P', 0),
        pct_3pa_corner=p_series.get('Corner 3s_% 3PA', 0)
    )
    print(generate_scouting_report(player_obj, stats_df, p_series))
    generate_comparison_chart(p_series, stats_df)

def select_player(stats_df):
    player_name_query = input("Enter a player's name or part of a name: ")
    player_data = stats_df[stats_df['Player'].str.contains(player_name_query, case=False, na=False)].copy()
    player_data = player_data.sort_values(by='Tm', ascending=False)
    player_data = player_data.drop_duplicates(subset='Player', keep='first')

    if player_data.empty:
        print(f"No players found matching '{player_name_query}'.")
        return None
    elif len(player_data) == 1:
        print(f"Found one player: {player_data.iloc[0]['Player']}")
        return player_data.iloc[0]
    else:
        print("Multiple players found. Please select one:")
        player_data = player_data.reset_index(drop=True)
        for index, row in player_data.iterrows():
            print(f"  {index + 1}: {row['Player']} ({row['Tm']})")
        
        while True:
            try:
                selection = int(input(f"Enter a number (1-{len(player_data)}): "))
                if 1 <= selection <= len(player_data):
                    return player_data.iloc[selection - 1]
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    return None

def display_comparison(p1_series, p2_series):
    print("\n========================================")
    print("PLAYER COMPARISON")
    print("========================================")
    
    p1_name = p1_series['Player']
    p2_name = p2_series['Player']

    print(f"{'Stat':<15} | {p1_name:<25} | {p2_name:<25}")
    print("-" * 67)

    stats_to_show = {
        'Archetype': 'Archetype',
        'Points': 'PTS',
        'Assists': 'AST',
        'Rebounds': 'TRB',
        'TS%': 'TS%',
        'PER': 'PER',
        'Win Shares': 'WS'
    }

    for display_name, stat_key in stats_to_show.items():
        p1_stat = p1_series.get(stat_key, 'N/A')
        p2_stat = p2_series.get(stat_key, 'N/A')
        
        if isinstance(p1_stat, float):
            p1_stat = f"{p1_stat:.3f}" if stat_key == 'TS%' else f"{p1_stat:.1f}"
        if isinstance(p2_stat, float):
            p2_stat = f"{p2_stat:.3f}" if stat_key == 'TS%' else f"{p2_stat:.1f}"
            
        print(f"{display_name:<15} | {p1_stat:<25} | {p2_stat:<25}")
    
    print("-" * 67)

def generate_head_to_head_chart(p1_series, p2_series):
    stats_to_plot = {
        'Points': 'PTS',
        'Assists': 'AST',
        'Rebounds': 'TRB',
        'PER': 'PER',
        'WS': 'WS'
    }
    
    p1_stats = [p1_series.get(key, 0) for key in stats_to_plot.values()]
    p2_stats = [p2_series.get(key, 0) for key in stats_to_plot.values()]
    
    labels = list(stats_to_plot.keys())
    
    x = numpy.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, p1_stats, width, label=p1_series['Player'])
    rects2 = ax.bar(x + width/2, p2_stats, width, label=p2_series['Player'])

    ax.set_ylabel('Value')
    ax.set_title(f"Head-to-Head Comparison: {p1_series['Player']} vs. {p2_series['Player']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig('player_comparison_chart.png')
    plt.close()
    print("\nComparison bar chart 'player_comparison_chart.png' has been saved.")


def main():
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
    per_game_df = get_player_stats_df(year=selected_year)
    
    if per_game_df is None:
        print("Could not retrieve per-game player data. Exiting.")
        return

    advanced_df = get_advanced_stats_df(year=selected_year)
    shooting_df = get_shooting_stats_df(year=selected_year)
    
    stats_df = per_game_df
    if advanced_df is not None:
        advanced_cols_to_merge = advanced_df[['Player', 'Tm', 'Age', 'PER', 'TS%', 'WS']]
        stats_df = pandas.merge(stats_df, advanced_cols_to_merge, on=['Player', 'Tm', 'Age'], how='left')
        stats_df[['PER', 'TS%', 'WS']] = stats_df[['PER', 'TS%', 'WS']].fillna(0)
    else:
        print("\nWarning: Could not retrieve advanced stats. Proceeding with basic stats only.\n")
        stats_df['PER'] = 0
        stats_df['TS%'] = 0
        stats_df['WS'] = 0

    if shooting_df is not None:
        shooting_cols_to_merge = shooting_df.drop(columns=['Pos', 'G', 'MP', 'Rk'])
        stats_df = pandas.merge(stats_df, shooting_cols_to_merge, on=['Player', 'Tm', 'Age'], how='left')
        stats_df.fillna(0, inplace=True)
    else:
        print("\nWarning: Could not retrieve shooting stats. Reports will not include shooting profile.\n")


    stats_df = add_player_archetypes(stats_df)
    print(f"--- Data loaded and archetypes identified successfully ---\n")

    while True:
        choice = input("Would you like to find a 'player', 'team', 'compare' players, or 'quit'? ").lower()

        if choice == 'quit':
            break
        
        elif choice == 'player':
            p_series = select_player(stats_df)
            if p_series is not None:
                process_player_selection(p_series, stats_df)

        elif choice == 'compare':
            print("\nSelect the first player to compare:")
            p1_series = select_player(stats_df)
            if p1_series is None:
                continue

            print("\nSelect the second player to compare:")
            p2_series = select_player(stats_df)
            if p2_series is None:
                continue
            
            display_comparison(p1_series, p2_series)
            generate_head_to_head_chart(p1_series, p2_series)

        elif choice == 'team':
            team_abbr = input("Enter the 3-letter team abbreviation (e.g., LAL, GSW): ").upper()
            team_df = stats_df[stats_df['Tm'] == team_abbr].sort_values(by='MP', ascending=False)

            if not team_df.empty:
                unit_choice = input("Do you want the 'starters' or the 'bench'? ").lower()
                
                player_list_df = pandas.DataFrame()
                if unit_choice == 'starters':
                    player_list_df = team_df.head(5)
                elif unit_choice == 'bench':
                    player_list_df = team_df.iloc[5:]
                else:
                    print("Invalid choice. Please enter 'starters' or 'bench'.")
                    continue
                
                for index, p_series in player_list_df.iterrows():
                    process_player_selection(p_series, stats_df)
            else:
                print(f"Team '{team_abbr}' not found.")
        
        else:
            print("Invalid input. Please enter 'player', 'team', 'compare', or 'quit'.")

if __name__ == "__main__":
    main()


