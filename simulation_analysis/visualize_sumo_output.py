"""
SUMO Lane Data Visualization Script
Converts SUMO XML output files to comprehensive visualizations
"""

import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Color palette for scenarios
COLORS = {
    'V0_morning': '#2E86AB',
    'V0_evening': '#A23B72',
    'V1_morning': '#F18F01',
    'V1_evening': '#C73E1D',
    'V2_morning': '#06A77D',
    'V2_evening': '#D4A5A5'
}

# Define custom ordering: morning before evening within each version
SCENARIO_ORDER = [
    'V0_morning', 'V0_evening',
    'V1_morning', 'V1_evening',
    'V2_morning', 'V2_evening'
]


def sort_scenarios(scenarios):
    """
    Sort scenarios so that within each version, morning comes before evening.
    Returns scenarios in order: V0_morning, V0_evening, V1_morning, V1_evening, etc.
    """
    # Create a mapping for custom sort order
    order_map = {scenario: i for i, scenario in enumerate(SCENARIO_ORDER)}
    
    # Sort by custom order, with any unknown scenarios at the end
    return sorted(scenarios, key=lambda x: order_map.get(x, 999))


def parse_sumo_xml(file_path):
    """
    Parse SUMO XML output file and extract lane data into a DataFrame.
    
    Returns:
        DataFrame with columns: edge_id, lane_id, and all metrics
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return pd.DataFrame()
    
    data = []
    
    # Find the interval element
    interval = root.find('.//interval')
    if interval is None:
        print(f"No interval data found in {file_path}")
        return pd.DataFrame()
    
    # Extract interval metadata
    interval_begin = float(interval.get('begin', 0))
    interval_end = float(interval.get('end', 0))
    
    # Iterate through edges
    for edge in interval.findall('edge'):
        edge_id = edge.get('id')
        
        # Iterate through lanes
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            
            # Extract all attributes
            lane_data = {
                'edge_id': edge_id,
                'lane_id': lane_id,
                'interval_begin': interval_begin,
                'interval_end': interval_end,
                'interval_duration': interval_end - interval_begin
            }
            
            # Extract all numeric metrics
            numeric_attrs = [
                'sampledSeconds', 'traveltime', 'overlapTraveltime',
                'density', 'laneDensity', 'occupancy', 'waitingTime',
                'timeLoss', 'speed', 'speedRelative', 'departed',
                'arrived', 'entered', 'left', 'laneChangedFrom',
                'laneChangedTo', 'teleported'
            ]
            
            for attr in numeric_attrs:
                value = lane.get(attr)
                if value is not None:
                    try:
                        lane_data[attr] = float(value)
                    except ValueError:
                        lane_data[attr] = 0.0
                else:
                    lane_data[attr] = 0.0
            
            data.append(lane_data)
    
    df = pd.DataFrame(data)
    return df


def load_all_data(data_dir='Data Output'):
    """
    Load all SUMO output files and combine into a single DataFrame.
    """
    data_dir = Path(data_dir)
    all_data = []
    
    files = {
        'V0_morning': data_dir / 'Output_LaneData_V0_morning-2.xml',
        'V0_evening': data_dir / 'Output_LaneData_V0_evening-2.xml',
        'V1_morning': data_dir / 'Output_LaneData_V1_morning-2.xml',
        'V1_evening': data_dir / 'Output_LaneData_V1_evening-2.xml',
        'V2_morning': data_dir / 'Output_LaneData_V2_morning-2.xml',
        'V2_evening': data_dir / 'Output_LaneData_V2_evening-2.xml'
    }
    
    for scenario, file_path in files.items():
        if not file_path.exists():
            print(f"File not found: {file_path}, skipping...")
            continue
            
        print(f"Loading {scenario}...")
        df = parse_sumo_xml(file_path)
        
        if not df.empty:
            df['scenario'] = scenario
            df['version'] = scenario.split('_')[0]
            df['time_period'] = scenario.split('_')[1]
            all_data.append(df)
            print(f"  Loaded {len(df)} lanes")
        else:
            print(f"  No data found in {file_path}")
    
    if not all_data:
        print("No data loaded!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal lanes loaded: {len(combined_df)}")
    return combined_df


def plot_speed_comparison(df, output_dir='plots'):
    """Compare average speeds across scenarios."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average speed by scenario
    speed_by_scenario = df.groupby('scenario')['speed'].mean()
    speed_by_scenario = speed_by_scenario.reindex(sort_scenarios(speed_by_scenario.index))
    
    axes[0].bar(speed_by_scenario.index, speed_by_scenario.values, 
                color=[COLORS.get(s, '#666666') for s in speed_by_scenario.index])
    axes[0].set_title('Average Speed by Scenario', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Average Speed (m/s)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Speed distribution
    for scenario in sort_scenarios(df['scenario'].unique()):
        speeds = df[df['scenario'] == scenario]['speed']
        axes[1].hist(speeds, alpha=0.6, label=scenario, 
                    color=COLORS.get(scenario, '#666666'), bins=50)
    
    axes[1].set_title('Speed Distribution by Scenario', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Speed (m/s)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'speed_comparison.png'}")
    plt.close()


def plot_density_comparison(df, output_dir='plots'):
    """Compare density metrics across scenarios."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Average density by scenario
    density_by_scenario = df.groupby('scenario')['density'].mean()
    density_by_scenario = density_by_scenario.reindex(sort_scenarios(density_by_scenario.index))
    axes[0, 0].bar(density_by_scenario.index, density_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in density_by_scenario.index])
    axes[0, 0].set_title('Average Density by Scenario', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('Average Density (veh/km)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Density distribution
    for scenario in sort_scenarios(df['scenario'].unique()):
        densities = df[df['scenario'] == scenario]['density']
        axes[0, 1].hist(densities, alpha=0.6, label=scenario,
                       color=COLORS.get(scenario, '#666666'), bins=50)
    axes[0, 1].set_title('Density Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Density (veh/km)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Occupancy comparison
    occupancy_by_scenario = df.groupby('scenario')['occupancy'].mean()
    occupancy_by_scenario = occupancy_by_scenario.reindex(sort_scenarios(occupancy_by_scenario.index))
    axes[1, 0].bar(occupancy_by_scenario.index, occupancy_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in occupancy_by_scenario.index])
    axes[1, 0].set_title('Average Occupancy by Scenario', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Scenario')
    axes[1, 0].set_ylabel('Average Occupancy (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Density vs Speed scatter
    for scenario in sort_scenarios(df['scenario'].unique()):
        subset = df[df['scenario'] == scenario]
        axes[1, 1].scatter(subset['density'], subset['speed'], 
                          alpha=0.5, label=scenario,
                          color=COLORS.get(scenario, '#666666'), s=20)
    axes[1, 1].set_title('Density vs Speed', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Density (veh/km)')
    axes[1, 1].set_ylabel('Speed (m/s)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'density_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'density_comparison.png'}")
    plt.close()


def plot_travel_time_comparison(df, output_dir='plots'):
    """Compare travel time and time loss metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Average travel time
    tt_by_scenario = df.groupby('scenario')['traveltime'].mean()
    tt_by_scenario = tt_by_scenario.reindex(sort_scenarios(tt_by_scenario.index))
    axes[0, 0].bar(tt_by_scenario.index, tt_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in tt_by_scenario.index])
    axes[0, 0].set_title('Average Travel Time by Scenario', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('Average Travel Time (s)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Time loss comparison
    timeloss_by_scenario = df.groupby('scenario')['timeLoss'].mean()
    timeloss_by_scenario = timeloss_by_scenario.reindex(sort_scenarios(timeloss_by_scenario.index))
    axes[0, 1].bar(timeloss_by_scenario.index, timeloss_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in timeloss_by_scenario.index])
    axes[0, 1].set_title('Average Time Loss by Scenario', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Scenario')
    axes[0, 1].set_ylabel('Average Time Loss (s)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Waiting time comparison
    waiting_by_scenario = df.groupby('scenario')['waitingTime'].mean()
    waiting_by_scenario = waiting_by_scenario.reindex(sort_scenarios(waiting_by_scenario.index))
    axes[1, 0].bar(waiting_by_scenario.index, waiting_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in waiting_by_scenario.index])
    axes[1, 0].set_title('Average Waiting Time by Scenario', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Scenario')
    axes[1, 0].set_ylabel('Average Waiting Time (s)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Travel time distribution
    for scenario in sort_scenarios(df['scenario'].unique()):
        tt = df[df['scenario'] == scenario]['traveltime']
        axes[1, 1].hist(tt, alpha=0.6, label=scenario,
                       color=COLORS.get(scenario, '#666666'), bins=50)
    axes[1, 1].set_title('Travel Time Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Travel Time (s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'travel_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'travel_time_comparison.png'}")
    plt.close()


def plot_version_comparison(df, output_dir='plots'):
    """Compare V0 vs V1 for morning and evening separately."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    metrics = ['speed', 'density', 'occupancy', 'traveltime', 'timeLoss', 'waitingTime']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data for comparison
        comparison_data = []
        for time_period in ['morning', 'evening']:
            for version in ['V0', 'V1', 'V2']:
                version_data = df[(df['version'] == version) & (df['time_period'] == time_period)][metric]
                
                if not version_data.empty:
                    comparison_data.append({
                        'Time Period': time_period.capitalize(),
                        'Version': version,
                        'Value': version_data.mean()
                    })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            x = np.arange(len(comp_df['Time Period'].unique()))
            width = 0.25
            
            for i, version in enumerate(['V0', 'V1', 'V2']):
                version_data = comp_df[comp_df['Version'] == version]
                if not version_data.empty:
                    values = [version_data[version_data['Time Period'] == tp]['Value'].values[0] 
                            if len(version_data[version_data['Time Period'] == tp]) > 0 else 0
                            for tp in comp_df['Time Period'].unique()]
                    ax.bar(x + i*width, values, width, label=version,
                          color=COLORS.get(f'{version}_morning', '#666666'), alpha=0.8)
            
            ax.set_xlabel('Time Period')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()}: Version Comparison', fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(comp_df['Time Period'].unique())
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'version_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'version_comparison.png'}")
    plt.close()


def plot_traffic_flow_metrics(df, output_dir='plots'):
    """Visualize traffic flow metrics (entered, left, teleported)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Vehicles entered
    entered_by_scenario = df.groupby('scenario')['entered'].sum()
    entered_by_scenario = entered_by_scenario.reindex(sort_scenarios(entered_by_scenario.index))
    axes[0, 0].bar(entered_by_scenario.index, entered_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in entered_by_scenario.index])
    axes[0, 0].set_title('Total Vehicles Entered by Scenario', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('Total Vehicles')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Vehicles left
    left_by_scenario = df.groupby('scenario')['left'].sum()
    left_by_scenario = left_by_scenario.reindex(sort_scenarios(left_by_scenario.index))
    axes[0, 1].bar(left_by_scenario.index, left_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in left_by_scenario.index])
    axes[0, 1].set_title('Total Vehicles Left by Scenario', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Scenario')
    axes[0, 1].set_ylabel('Total Vehicles')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Teleported vehicles (vehicles that couldn't complete route)
    teleported_by_scenario = df.groupby('scenario')['teleported'].sum()
    teleported_by_scenario = teleported_by_scenario.reindex(sort_scenarios(teleported_by_scenario.index))
    axes[1, 0].bar(teleported_by_scenario.index, teleported_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in teleported_by_scenario.index])
    axes[1, 0].set_title('Total Teleported Vehicles by Scenario', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Scenario')
    axes[1, 0].set_ylabel('Total Teleported Vehicles')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Lane changes
    df['total_lane_changes'] = df['laneChangedFrom'] + df['laneChangedTo']
    lane_changes_by_scenario = df.groupby('scenario')['total_lane_changes'].sum()
    lane_changes_by_scenario = lane_changes_by_scenario.reindex(sort_scenarios(lane_changes_by_scenario.index))
    axes[1, 1].bar(lane_changes_by_scenario.index, lane_changes_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in lane_changes_by_scenario.index])
    axes[1, 1].set_title('Total Lane Changes by Scenario', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Scenario')
    axes[1, 1].set_ylabel('Total Lane Changes')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'traffic_flow_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'traffic_flow_metrics.png'}")
    plt.close()


def create_summary_statistics(df, output_dir='plots'):
    """Create a summary statistics table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    metrics = ['speed', 'density', 'occupancy', 'traveltime', 'timeLoss', 
               'waitingTime', 'entered', 'left', 'teleported']
    
    summary = df.groupby('scenario')[metrics].agg(['mean', 'std', 'min', 'max'])
    
    # Save to CSV
    summary.to_csv(output_dir / 'summary_statistics.csv')
    print(f"Saved: {output_dir / 'summary_statistics.csv'}")
    
    # Create a visual table
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table (show mean values)
    table_data = []
    for scenario in sort_scenarios(df['scenario'].unique()):
        row = [scenario]
        for metric in metrics:
            mean_val = df[df['scenario'] == scenario][metric].mean()
            row.append(f"{mean_val:.2f}")
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Scenario'] + [m.capitalize() for m in metrics],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(metrics) + 1):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Summary Statistics by Scenario', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_statistics_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'summary_statistics_table.png'}")
    plt.close()


def plot_correlation_heatmap(df, output_dir='plots'):
    """Create correlation heatmaps for each scenario."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    metrics = ['speed', 'density', 'occupancy', 'traveltime', 'timeLoss', 
               'waitingTime', 'entered', 'left']
    
    sorted_scenarios = sort_scenarios(df['scenario'].unique())
    n_scenarios = len(sorted_scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6*n_scenarios, 5))
    
    if n_scenarios == 1:
        axes = [axes]
    
    for idx, scenario in enumerate(sorted_scenarios):
        scenario_data = df[df['scenario'] == scenario][metrics]
        corr_matrix = scenario_data.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, ax=axes[idx],
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        axes[idx].set_title(f'Correlation Matrix: {scenario}', 
                           fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'correlation_heatmap.png'}")
    plt.close()


def plot_performance_metrics(df, output_dir='plots'):
    """Plot comprehensive performance metrics across scenarios."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Performance index: speed relative to free flow
    speed_rel_by_scenario = df.groupby('scenario')['speedRelative'].mean()
    speed_rel_by_scenario = speed_rel_by_scenario.reindex(sort_scenarios(speed_rel_by_scenario.index))
    axes[0, 0].bar(speed_rel_by_scenario.index, speed_rel_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in speed_rel_by_scenario.index])
    axes[0, 0].set_title('Average Speed Relative to Free Flow', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('Speed Relative')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Free Flow')
    axes[0, 0].legend()
    
    # Efficiency: vehicles per time loss
    df['efficiency'] = df['entered'] / (df['timeLoss'] + 1)  # +1 to avoid division by zero
    efficiency_by_scenario = df.groupby('scenario')['efficiency'].mean()
    efficiency_by_scenario = efficiency_by_scenario.reindex(sort_scenarios(efficiency_by_scenario.index))
    axes[0, 1].bar(efficiency_by_scenario.index, efficiency_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in efficiency_by_scenario.index])
    axes[0, 1].set_title('Traffic Efficiency (Vehicles/Time Loss)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Scenario')
    axes[0, 1].set_ylabel('Efficiency Index')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Congestion severity: occupancy vs waiting time
    for scenario in sort_scenarios(df['scenario'].unique()):
        subset = df[df['scenario'] == scenario]
        axes[1, 0].scatter(subset['occupancy'], subset['waitingTime'],
                          alpha=0.5, label=scenario, s=30,
                          color=COLORS.get(scenario, '#666666'))
    axes[1, 0].set_title('Congestion Analysis: Occupancy vs Waiting Time', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Occupancy (%)')
    axes[1, 0].set_ylabel('Waiting Time (s)')
    axes[1, 0].legend(loc='best', ncol=2, fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    
    # Teleportation rate (vehicles that couldn't complete route)
    df['teleport_rate'] = df['teleported'] / (df['entered'] + df['teleported'] + 1) * 100
    teleport_rate_by_scenario = df.groupby('scenario')['teleport_rate'].mean()
    teleport_rate_by_scenario = teleport_rate_by_scenario.reindex(sort_scenarios(teleport_rate_by_scenario.index))
    axes[1, 1].bar(teleport_rate_by_scenario.index, teleport_rate_by_scenario.values,
                   color=[COLORS.get(s, '#666666') for s in teleport_rate_by_scenario.index])
    axes[1, 1].set_title('Teleportation Rate (Route Completion Failure)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Scenario')
    axes[1, 1].set_ylabel('Teleportation Rate (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_metrics.png'}")
    plt.close()


def plot_edge_level_analysis(df, output_dir='plots'):
    """Analyze performance at edge level."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find top edges by traffic volume
    edge_totals = df.groupby('edge_id')['entered'].sum().sort_values(ascending=False)
    top_edges = edge_totals.head(15).index.tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Top edges by total vehicles
    axes[0, 0].barh(range(len(top_edges)), edge_totals[top_edges].values, color='steelblue')
    axes[0, 0].set_yticks(range(len(top_edges)))
    axes[0, 0].set_yticklabels([e[:20] + '...' if len(e) > 20 else e for e in top_edges], fontsize=8)
    axes[0, 0].set_xlabel('Total Vehicles Entered')
    axes[0, 0].set_title('Top 15 Edges by Traffic Volume', fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Average speed by edge (top 15)
    avg_speed_by_edge = df.groupby('edge_id')['speed'].mean().sort_values(ascending=False)
    top_speed_edges = avg_speed_by_edge.head(15)
    axes[0, 1].barh(range(len(top_speed_edges)), top_speed_edges.values, color='coral')
    axes[0, 1].set_yticks(range(len(top_speed_edges)))
    axes[0, 1].set_yticklabels([e[:20] + '...' if len(e) > 20 else e for e in top_speed_edges.index], 
                               fontsize=8)
    axes[0, 1].set_xlabel('Average Speed (m/s)')
    axes[0, 1].set_title('Top 15 Edges by Average Speed', fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Most congested edges (highest time loss)
    time_loss_by_edge = df.groupby('edge_id')['timeLoss'].mean().sort_values(ascending=False)
    top_congested = time_loss_by_edge.head(15)
    axes[1, 0].barh(range(len(top_congested)), top_congested.values, color='darkred')
    axes[1, 0].set_yticks(range(len(top_congested)))
    axes[1, 0].set_yticklabels([e[:20] + '...' if len(e) > 20 else e for e in top_congested.index], 
                               fontsize=8)
    axes[1, 0].set_xlabel('Average Time Loss (s)')
    axes[1, 0].set_title('Top 15 Most Congested Edges', fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Edge performance by scenario (box plot)
    edge_performance_data = []
    sorted_scenarios = sort_scenarios(df['scenario'].unique())
    for scenario in sorted_scenarios:
        scenario_edges = df[df['scenario'] == scenario]
        # Calculate performance as speed / (density + 1) to normalize
        scenario_edges['performance'] = scenario_edges['speed'] / (scenario_edges['density'] + 1)
        edge_performance_data.append(scenario_edges['performance'].values)
    
    axes[1, 1].boxplot(edge_performance_data, labels=sorted_scenarios)
    axes[1, 1].set_title('Edge Performance Distribution by Scenario', fontweight='bold')
    axes[1, 1].set_xlabel('Scenario')
    axes[1, 1].set_ylabel('Performance Index (Speed/Density)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_level_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'edge_level_analysis.png'}")
    plt.close()


def plot_congestion_analysis(df, output_dir='plots'):
    """Detailed congestion analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Congestion levels by scenario
    # Define congestion levels based on occupancy
    def categorize_congestion(occ):
        if occ < 10:
            return 'Free Flow'
        elif occ < 30:
            return 'Moderate'
        elif occ < 50:
            return 'Heavy'
        else:
            return 'Severe'
    
    df['congestion_level'] = df['occupancy'].apply(categorize_congestion)
    congestion_counts = df.groupby(['scenario', 'congestion_level']).size().unstack(fill_value=0)
    # Reindex to ensure correct scenario ordering (morning before evening within each version)
    sorted_scenarios = sort_scenarios(congestion_counts.index)
    congestion_counts = congestion_counts.reindex(sorted_scenarios)
    congestion_counts.plot(kind='bar', ax=axes[0, 0], color=['green', 'yellow', 'orange', 'red'])
    axes[0, 0].set_title('Congestion Level Distribution by Scenario', fontweight='bold')
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('Number of Lanes')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend(title='Congestion Level')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Time loss vs density relationship
    for scenario in sort_scenarios(df['scenario'].unique()):
        subset = df[df['scenario'] == scenario]
        axes[0, 1].scatter(subset['density'], subset['timeLoss'],
                          alpha=0.4, label=scenario, s=25,
                          color=COLORS.get(scenario, '#666666'))
    axes[0, 1].set_title('Time Loss vs Density', fontweight='bold')
    axes[0, 1].set_xlabel('Density (veh/km)')
    axes[0, 1].set_ylabel('Time Loss (s)')
    axes[0, 1].legend(loc='best', ncol=2, fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    
    # Waiting time distribution
    for scenario in sort_scenarios(df['scenario'].unique()):
        waiting_times = df[df['scenario'] == scenario]['waitingTime']
        axes[1, 0].hist(waiting_times, alpha=0.6, label=scenario, bins=50,
                       color=COLORS.get(scenario, '#666666'))
    axes[1, 0].set_title('Waiting Time Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Waiting Time (s)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend(loc='best', ncol=2, fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlim(0, df['waitingTime'].quantile(0.95))  # Remove extreme outliers for clarity
    
    # Average metrics by version
    version_metrics = df.groupby('version').agg({
        'speed': 'mean',
        'density': 'mean',
        'timeLoss': 'mean',
        'occupancy': 'mean'
    })
    # Ensure version order: V0, V1, V2
    version_metrics = version_metrics.reindex(['V0', 'V1', 'V2'])
    
    x = np.arange(len(version_metrics.index))
    width = 0.2
    metrics_to_plot = ['speed', 'density', 'timeLoss', 'occupancy']
    for i, metric in enumerate(metrics_to_plot):
        # Normalize values for comparison (0-1 scale)
        values = version_metrics[metric].values
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
        axes[1, 1].bar(x + i*width, normalized, width, label=metric.capitalize(), alpha=0.8)
    
    axes[1, 1].set_title('Normalized Metrics by Version', fontweight='bold')
    axes[1, 1].set_xlabel('Version')
    axes[1, 1].set_ylabel('Normalized Value (0-1)')
    axes[1, 1].set_xticks(x + width * 1.5)
    axes[1, 1].set_xticklabels(version_metrics.index)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'congestion_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'congestion_analysis.png'}")
    plt.close()


def main():
    """Main function to run all visualizations."""
    print("=" * 60)
    print("SUMO Output Visualization Script")
    print("=" * 60)
    
    # Load data
    df = load_all_data('Data Output')
    
    if df.empty:
        print("No data to visualize!")
        return
    
    # Create output directory in subfolder with ordered graphics
    output_dir = Path('plots/lane_data/ordered')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    # Generate all plots
    plot_speed_comparison(df, output_dir)
    plot_density_comparison(df, output_dir)
    plot_travel_time_comparison(df, output_dir)
    plot_version_comparison(df, output_dir)
    plot_traffic_flow_metrics(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    create_summary_statistics(df, output_dir)
    
    # Generate new additional plots
    plot_performance_metrics(df, output_dir)
    plot_edge_level_analysis(df, output_dir)
    plot_congestion_analysis(df, output_dir)
    
    print("\n" + "=" * 60)
    print("All visualizations completed!")
    print(f"Plots saved in: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

