"""
SUMO Loop Detector Data Visualization Script
Converts SUMO loop detector XML output files to comprehensive visualizations
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
plt.rcParams['figure.figsize'] = (14, 8)
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

# Line styles for variants (to make them more visible)
VARIANT_LINESTYLES = {
    'V0': '-',
    'V1': '--',
    'V2': '-.'
}

# Markers for time periods
TIME_MARKERS = {
    'morning': 'o',
    'evening': 's'
}


def sort_scenarios(scenarios):
    """
    Sort scenarios so that within each version, morning comes before evening.
    Returns scenarios in order: V0_morning, V0_evening, V1_morning, V1_evening, etc.
    """
    # Create a mapping for custom sort order
    order_map = {scenario: i for i, scenario in enumerate(SCENARIO_ORDER)}
    
    # Sort by custom order, with any unknown scenarios at the end
    return sorted(scenarios, key=lambda x: order_map.get(x, 999))


def parse_loop_xml(file_path):
    """
    Parse SUMO loop detector XML output file and extract interval data into a DataFrame.
    
    Returns:
        DataFrame with columns: detector_id, interval info, and all metrics
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return pd.DataFrame()
    
    data = []
    
    # Iterate through interval elements
    for interval in root.findall('interval'):
        interval_data = {
            'detector_id': interval.get('id'),
            'begin': float(interval.get('begin', 0)),
            'end': float(interval.get('end', 0)),
        }
        
        # Calculate interval duration and midpoint for time series
        interval_data['duration'] = interval_data['end'] - interval_data['begin']
        interval_data['time_midpoint'] = (interval_data['begin'] + interval_data['end']) / 2
        
        # Extract all numeric metrics
        numeric_attrs = [
            'nVehContrib', 'flow', 'occupancy', 'speed',
            'harmonicMeanSpeed', 'length', 'nVehEntered'
        ]
        
        for attr in numeric_attrs:
            value = interval.get(attr)
            if value is not None:
                try:
                    interval_data[attr] = float(value)
                except ValueError:
                    interval_data[attr] = np.nan
            else:
                interval_data[attr] = np.nan
        
        data.append(interval_data)
    
    df = pd.DataFrame(data)
    return df


def load_all_loop_data(data_dir='Loop Data'):
    """
    Load all SUMO loop detector output files and combine into a single DataFrame.
    """
    data_dir = Path(data_dir)
    all_data = []
    
    files = {
        'V0_morning': data_dir / 'Output_LoopData_V0_morning.xml',
        'V0_evening': data_dir / 'Output_LoopData_V0_evening.xml',
        'V1_morning': data_dir / 'Output_LoopData_V1_morning.xml',
        'V1_evening': data_dir / 'Output_LoopData_V1_evening.xml',
        'V2_morning': data_dir / 'Output_LoopData_V2_morning.xml',
        'V2_evening': data_dir / 'Output_LoopData_V2_evening.xml'
    }
    
    for scenario, file_path in files.items():
        if not file_path.exists():
            print(f"File not found: {file_path}, skipping...")
            continue
            
        print(f"Loading {scenario}...")
        df = parse_loop_xml(file_path)
        
        if not df.empty:
            # Filter to only include e1_03N (bridge) and e1_07N (tunnel) loops
            df = df[df['detector_id'].str.startswith('e1_03') | df['detector_id'].str.startswith('e1_07')]
            
            if not df.empty:
                # Add location classification
                df['location'] = df['detector_id'].apply(
                    lambda x: 'Bridge' if x.startswith('e1_03') else 'Tunnel'
                )
                df['scenario'] = scenario
                df['version'] = scenario.split('_')[0]
                df['time_period'] = scenario.split('_')[1]
                all_data.append(df)
                print(f"  Loaded {len(df)} intervals from {df['detector_id'].nunique()} detectors ({df['location'].unique()})")
        else:
            print(f"  No data found in {file_path}")
    
    if not all_data:
        print("No data loaded!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal intervals loaded: {len(combined_df)}")
    print(f"Unique detectors: {combined_df['detector_id'].nunique()}")
    print(f"Bridge detectors: {combined_df[combined_df['location'] == 'Bridge']['detector_id'].nunique()}")
    print(f"Tunnel detectors: {combined_df[combined_df['location'] == 'Tunnel']['detector_id'].nunique()}")
    return combined_df


def plot_flow_time_series(df, output_dir='plots'):
    """Plot flow rate time series comparing Bridge vs Tunnel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Flow over time: Bridge vs Tunnel by scenario
    # Use variant line style and location marker to make both distinctions clear
    for scenario in sort_scenarios(df['scenario'].unique()):
        scenario_data = df[df['scenario'] == scenario].copy()
        scenario_data['time_hours'] = scenario_data['time_midpoint'] / 3600
        
        version = scenario.split('_')[0]
        time_period = scenario.split('_')[1]
        
        for location in ['Bridge', 'Tunnel']:
            location_data = scenario_data[scenario_data['location'] == location]
            if not location_data.empty:
                flow_by_time = location_data.groupby('time_hours')['flow'].mean()
                if isinstance(flow_by_time, pd.Series):
                    # Use variant line style, location-based marker, and scenario color
                    linestyle = VARIANT_LINESTYLES.get(version, '-')
                    marker = 'o' if location == 'Bridge' else 's'
                    axes[0].plot(flow_by_time.index, flow_by_time.values,
                               label=f"{scenario} - {location}", linewidth=2.5,
                               linestyle=linestyle, marker=marker, markersize=4,
                               color=COLORS.get(scenario, '#666666'))
    
    axes[0].set_title('Flow Rate Over Time: Bridge vs Tunnel', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('Flow Rate (veh/h)')
    axes[0].legend(loc='best', ncol=2, fontsize=8)
    axes[0].grid(alpha=0.3)
    
    # Average flow: Bridge vs Tunnel by scenario
    flow_comparison = df.groupby(['scenario', 'location'])['flow'].mean().unstack()
    # Reindex to ensure correct ordering
    sorted_scenarios = sort_scenarios(flow_comparison.index)
    flow_comparison = flow_comparison.reindex(sorted_scenarios)
    x = np.arange(len(flow_comparison.index))
    width = 0.35
    if 'Bridge' in flow_comparison.columns and 'Tunnel' in flow_comparison.columns:
        axes[1].bar(x - width/2, flow_comparison['Bridge'], width, 
                   label='Bridge', color='#2E86AB', alpha=0.8)
        axes[1].bar(x + width/2, flow_comparison['Tunnel'], width,
                   label='Tunnel', color='#A23B72', alpha=0.8)
    axes[1].set_title('Average Flow: Bridge vs Tunnel by Scenario', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Average Flow (veh/h)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(flow_comparison.index, rotation=45)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Total flow by location
    total_flow_by_location = df.groupby('location')['flow'].sum()
    axes[2].bar(total_flow_by_location.index, total_flow_by_location.values,
               color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[2].set_title('Total Flow by Location (All Scenarios)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Location')
    axes[2].set_ylabel('Total Flow (veh/h)')
    axes[2].grid(axis='y', alpha=0.3)
    
    # Flow distribution comparison
    bridge_flows = df[df['location'] == 'Bridge']['flow']
    tunnel_flows = df[df['location'] == 'Tunnel']['flow']
    axes[3].hist(bridge_flows, bins=50, alpha=0.6, label='Bridge', color='#2E86AB')
    axes[3].hist(tunnel_flows, bins=50, alpha=0.6, label='Tunnel', color='#A23B72')
    axes[3].set_title('Flow Distribution: Bridge vs Tunnel', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Flow (veh/h)')
    axes[3].set_ylabel('Frequency')
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loop_flow_time_series.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loop_flow_time_series.png'}")
    plt.close()


def plot_speed_time_series(df, output_dir='plots'):
    """Plot speed time series comparing Bridge vs Tunnel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Speed over time: Bridge vs Tunnel
    valid_df = df[df['speed'] > 0].copy()
    
    for scenario in sort_scenarios(valid_df['scenario'].unique()):
        scenario_data = valid_df[valid_df['scenario'] == scenario].copy()
        scenario_data['time_hours'] = scenario_data['time_midpoint'] / 3600
        
        version = scenario.split('_')[0]
        
        for location in ['Bridge', 'Tunnel']:
            location_data = scenario_data[scenario_data['location'] == location]
            if not location_data.empty:
                speed_by_time = location_data.groupby('time_hours')['speed'].mean()
                if isinstance(speed_by_time, pd.Series):
                    linestyle = VARIANT_LINESTYLES.get(version, '-')
                    marker = 'o' if location == 'Bridge' else 's'
                    axes[0].plot(speed_by_time.index, speed_by_time.values,
                               label=f"{scenario} - {location}", linewidth=2.5,
                               linestyle=linestyle, marker=marker, markersize=4,
                               color=COLORS.get(scenario, '#666666'))
    
    axes[0].set_title('Speed Over Time: Bridge vs Tunnel', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].legend(loc='best', ncol=2, fontsize=8)
    axes[0].grid(alpha=0.3)
    
    # Average speed: Bridge vs Tunnel by scenario
    speed_comparison = valid_df.groupby(['scenario', 'location'])['speed'].mean().unstack()
    # Reindex to ensure correct ordering
    sorted_scenarios = sort_scenarios(speed_comparison.index)
    speed_comparison = speed_comparison.reindex(sorted_scenarios)
    x = np.arange(len(speed_comparison.index))
    width = 0.35
    if 'Bridge' in speed_comparison.columns and 'Tunnel' in speed_comparison.columns:
        axes[1].bar(x - width/2, speed_comparison['Bridge'], width,
                   label='Bridge', color='#2E86AB', alpha=0.8)
        axes[1].bar(x + width/2, speed_comparison['Tunnel'], width,
                   label='Tunnel', color='#A23B72', alpha=0.8)
    axes[1].set_title('Average Speed: Bridge vs Tunnel by Scenario', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Average Speed (m/s)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(speed_comparison.index, rotation=45)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Speed distribution
    bridge_speeds = valid_df[valid_df['location'] == 'Bridge']['speed']
    tunnel_speeds = valid_df[valid_df['location'] == 'Tunnel']['speed']
    axes[2].hist(bridge_speeds, bins=50, alpha=0.6, label='Bridge', color='#2E86AB')
    axes[2].hist(tunnel_speeds, bins=50, alpha=0.6, label='Tunnel', color='#A23B72')
    axes[2].set_title('Speed Distribution: Bridge vs Tunnel', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Speed (m/s)')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    # Harmonic mean speed comparison
    valid_harmonic = df[df['harmonicMeanSpeed'] > 0].copy()
    harmonic_comparison = valid_harmonic.groupby(['scenario', 'location'])['harmonicMeanSpeed'].mean().unstack()
    # Reindex to ensure correct ordering
    sorted_scenarios = sort_scenarios(harmonic_comparison.index)
    harmonic_comparison = harmonic_comparison.reindex(sorted_scenarios)
    x2 = np.arange(len(harmonic_comparison.index))
    if 'Bridge' in harmonic_comparison.columns and 'Tunnel' in harmonic_comparison.columns:
        axes[3].bar(x2 - width/2, harmonic_comparison['Bridge'], width,
                   label='Bridge', color='#2E86AB', alpha=0.8)
        axes[3].bar(x2 + width/2, harmonic_comparison['Tunnel'], width,
                   label='Tunnel', color='#A23B72', alpha=0.8)
    axes[3].set_title('Harmonic Mean Speed: Bridge vs Tunnel', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Scenario')
    axes[3].set_ylabel('Harmonic Mean Speed (m/s)')
    axes[3].set_xticks(x2)
    axes[3].set_xticklabels(harmonic_comparison.index, rotation=45)
    axes[3].legend()
    axes[3].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loop_speed_time_series.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loop_speed_time_series.png'}")
    plt.close()


def plot_occupancy_time_series(df, output_dir='plots'):
    """Plot occupancy time series comparing Bridge vs Tunnel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Occupancy over time: Bridge vs Tunnel
    for scenario in sort_scenarios(df['scenario'].unique()):
        scenario_data = df[df['scenario'] == scenario].copy()
        scenario_data['time_hours'] = scenario_data['time_midpoint'] / 3600
        
        version = scenario.split('_')[0]
        
        for location in ['Bridge', 'Tunnel']:
            location_data = scenario_data[scenario_data['location'] == location]
            if not location_data.empty:
                occupancy_by_time = location_data.groupby('time_hours')['occupancy'].mean()
                if isinstance(occupancy_by_time, pd.Series):
                    linestyle = VARIANT_LINESTYLES.get(version, '-')
                    marker = 'o' if location == 'Bridge' else 's'
                    axes[0].plot(occupancy_by_time.index, occupancy_by_time.values,
                               label=f"{scenario} - {location}", linewidth=2.5,
                               linestyle=linestyle, marker=marker, markersize=4,
                               color=COLORS.get(scenario, '#666666'))
    
    axes[0].set_title('Occupancy Over Time: Bridge vs Tunnel', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('Occupancy (%)')
    axes[0].legend(loc='best', ncol=2, fontsize=8)
    axes[0].grid(alpha=0.3)
    
    # Average occupancy: Bridge vs Tunnel by scenario
    occupancy_comparison = df.groupby(['scenario', 'location'])['occupancy'].mean().unstack()
    # Reindex to ensure correct ordering
    sorted_scenarios = sort_scenarios(occupancy_comparison.index)
    occupancy_comparison = occupancy_comparison.reindex(sorted_scenarios)
    x = np.arange(len(occupancy_comparison.index))
    width = 0.35
    if 'Bridge' in occupancy_comparison.columns and 'Tunnel' in occupancy_comparison.columns:
        axes[1].bar(x - width/2, occupancy_comparison['Bridge'], width,
                   label='Bridge', color='#2E86AB', alpha=0.8)
        axes[1].bar(x + width/2, occupancy_comparison['Tunnel'], width,
                   label='Tunnel', color='#A23B72', alpha=0.8)
    axes[1].set_title('Average Occupancy: Bridge vs Tunnel by Scenario', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Average Occupancy (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(occupancy_comparison.index, rotation=45)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Occupancy distribution
    bridge_occ = df[df['location'] == 'Bridge']['occupancy']
    tunnel_occ = df[df['location'] == 'Tunnel']['occupancy']
    axes[2].hist(bridge_occ, bins=50, alpha=0.6, label='Bridge', color='#2E86AB')
    axes[2].hist(tunnel_occ, bins=50, alpha=0.6, label='Tunnel', color='#A23B72')
    axes[2].set_title('Occupancy Distribution: Bridge vs Tunnel', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Occupancy (%)')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    # Peak occupancy comparison
    peak_occupancy = df.groupby(['scenario', 'location'])['occupancy'].max().unstack()
    # Reindex to ensure correct ordering
    sorted_scenarios = sort_scenarios(peak_occupancy.index)
    peak_occupancy = peak_occupancy.reindex(sorted_scenarios)
    x2 = np.arange(len(peak_occupancy.index))
    if 'Bridge' in peak_occupancy.columns and 'Tunnel' in peak_occupancy.columns:
        axes[3].bar(x2 - width/2, peak_occupancy['Bridge'], width,
                   label='Bridge', color='#2E86AB', alpha=0.8)
        axes[3].bar(x2 + width/2, peak_occupancy['Tunnel'], width,
                   label='Tunnel', color='#A23B72', alpha=0.8)
    axes[3].set_title('Peak Occupancy: Bridge vs Tunnel by Scenario', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Scenario')
    axes[3].set_ylabel('Peak Occupancy (%)')
    axes[3].set_xticks(x2)
    axes[3].set_xticklabels(peak_occupancy.index, rotation=45)
    axes[3].legend()
    axes[3].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loop_occupancy_time_series.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loop_occupancy_time_series.png'}")
    plt.close()


def plot_version_comparison(df, output_dir='plots'):
    """Compare metrics across versions (V0, V1, V2) for Bridge vs Tunnel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    metrics = ['flow', 'speed', 'occupancy', 'nVehEntered']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data for comparison: Bridge vs Tunnel by version
        comparison_data = []
        for location in ['Bridge', 'Tunnel']:
            for version in ['V0', 'V1', 'V2']:
                location_data = df[(df['version'] == version) & 
                                  (df['location'] == location)]
                
                if metric == 'speed':
                    location_data = location_data[location_data['speed'] > 0]
                
                if not location_data.empty:
                    mean_val = location_data[metric].mean()
                    comparison_data.append({
                        'Location': location,
                        'Version': version,
                        'Value': mean_val
                    })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Create grouped bar chart
            x = np.arange(len(comp_df['Location'].unique()))
            width = 0.25
            
            for i, version in enumerate(['V0', 'V1', 'V2']):
                version_data = comp_df[comp_df['Version'] == version]
                if not version_data.empty:
                    values = [version_data[version_data['Location'] == loc]['Value'].values[0] 
                            if len(version_data[version_data['Location'] == loc]) > 0 else 0
                            for loc in comp_df['Location'].unique()]
                    ax.bar(x + i*width, values, width, label=version,
                          color=COLORS.get(f'{version}_morning', '#666666'), alpha=0.8)
            
            ax.set_xlabel('Location')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()}: Bridge vs Tunnel by Version', fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(comp_df['Location'].unique())
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loop_version_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loop_version_comparison.png'}")
    plt.close()


def plot_detector_heatmap(df, output_dir='plots'):
    """Create heatmaps showing flow patterns for Bridge vs Tunnel detectors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Select scenarios for heatmap - use sorted order to ensure morning before evening
    available_scenarios = sort_scenarios(df['scenario'].unique())
    # Filter to only include scenarios that have data
    scenarios_to_plot = [s for s in available_scenarios if s in df['scenario'].unique()]
    
    n_scenarios = len(scenarios_to_plot)
    if n_scenarios == 0:
        print("No matching scenarios for heatmap")
        return
    
    fig, axes = plt.subplots(n_scenarios, 2, figsize=(20, 6*n_scenarios))
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    for scenario in scenarios_to_plot:
        if scenario not in df['scenario'].unique():
            continue
            
        scenario_data = df[df['scenario'] == scenario].copy()
        scenario_data['time_hours'] = scenario_data['time_midpoint'] / 3600
        
        # Bridge heatmap
        bridge_data = scenario_data[scenario_data['location'] == 'Bridge']
        if not bridge_data.empty:
            pivot_bridge = bridge_data.pivot_table(
                values='flow',
                index='detector_id',
                columns='time_hours',
                aggfunc='mean'
            )
            pivot_bridge = pivot_bridge.sort_index()
            sns.heatmap(pivot_bridge, annot=False, fmt='.0f', cmap='YlOrRd',
                       ax=axes[plot_idx, 0], cbar_kws={'label': 'Flow (veh/h)'})
            axes[plot_idx, 0].set_title(f'Bridge Detectors: {scenario}', 
                                      fontsize=12, fontweight='bold')
            axes[plot_idx, 0].set_xlabel('Time (hours)')
            axes[plot_idx, 0].set_ylabel('Detector ID')
        
        # Tunnel heatmap
        tunnel_data = scenario_data[scenario_data['location'] == 'Tunnel']
        if not tunnel_data.empty:
            pivot_tunnel = tunnel_data.pivot_table(
                values='flow',
                index='detector_id',
                columns='time_hours',
                aggfunc='mean'
            )
            pivot_tunnel = pivot_tunnel.sort_index()
            sns.heatmap(pivot_tunnel, annot=False, fmt='.0f', cmap='YlOrRd',
                       ax=axes[plot_idx, 1], cbar_kws={'label': 'Flow (veh/h)'})
            axes[plot_idx, 1].set_title(f'Tunnel Detectors: {scenario}', 
                                      fontsize=12, fontweight='bold')
            axes[plot_idx, 1].set_xlabel('Time (hours)')
            axes[plot_idx, 1].set_ylabel('Detector ID')
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loop_detector_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loop_detector_heatmap.png'}")
    plt.close()


def plot_flow_speed_relationship(df, output_dir='plots'):
    """Plot fundamental diagram: flow vs speed relationship for Bridge vs Tunnel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Filter out invalid speeds
    valid_data = df[df['speed'] > 0].copy()
    
    # Bridge fundamental diagram
    bridge_data = valid_data[valid_data['location'] == 'Bridge']
    for scenario in sort_scenarios(bridge_data['scenario'].unique()):
        scenario_data = bridge_data[bridge_data['scenario'] == scenario]
        version = scenario.split('_')[0]
        marker = 'o' if 'morning' in scenario else 's'
        axes[0].scatter(scenario_data['speed'], scenario_data['flow'],
                       alpha=0.6, label=scenario, s=40,
                       color=COLORS.get(scenario, '#666666'), marker=marker,
                       edgecolors='black', linewidths=0.5)
    axes[0].set_title('Fundamental Diagram: Bridge', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Speed (m/s)')
    axes[0].set_ylabel('Flow (veh/h)')
    axes[0].legend(loc='best', ncol=2, fontsize=8)
    axes[0].grid(alpha=0.3)
    
    # Tunnel fundamental diagram
    tunnel_data = valid_data[valid_data['location'] == 'Tunnel']
    for scenario in sort_scenarios(tunnel_data['scenario'].unique()):
        scenario_data = tunnel_data[tunnel_data['scenario'] == scenario]
        version = scenario.split('_')[0]
        marker = 'o' if 'morning' in scenario else 's'
        axes[1].scatter(scenario_data['speed'], scenario_data['flow'],
                       alpha=0.6, label=scenario, s=40,
                       color=COLORS.get(scenario, '#666666'), marker=marker,
                       edgecolors='black', linewidths=0.5)
    axes[1].set_title('Fundamental Diagram: Tunnel', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Speed (m/s)')
    axes[1].set_ylabel('Flow (veh/h)')
    axes[1].legend(loc='best', ncol=2, fontsize=8)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loop_flow_speed_diagram.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loop_flow_speed_diagram.png'}")
    plt.close()


def plot_detector_statistics(df, output_dir='plots'):
    """Plot statistics comparing Bridge vs Tunnel detectors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Top detectors by total flow: Bridge vs Tunnel
    bridge_detectors = df[df['location'] == 'Bridge'].groupby('detector_id')['flow'].sum().sort_values(ascending=False)
    tunnel_detectors = df[df['location'] == 'Tunnel'].groupby('detector_id')['flow'].sum().sort_values(ascending=False)
    
    top_bridge = bridge_detectors.head(5)
    top_tunnel = tunnel_detectors.head(5)
    
    x = np.arange(max(len(top_bridge), len(top_tunnel)))
    width = 0.35
    axes[0, 0].barh(x[:len(top_bridge)] - width/2, top_bridge.values, width,
                    label='Bridge', color='#2E86AB', alpha=0.8)
    axes[0, 0].barh(x[:len(top_tunnel)] + width/2, top_tunnel.values, width,
                    label='Tunnel', color='#A23B72', alpha=0.8)
    axes[0, 0].set_yticks(x[:max(len(top_bridge), len(top_tunnel))])
    axes[0, 0].set_yticklabels(list(top_bridge.index) + [''] * (len(x) - len(top_bridge)))
    axes[0, 0].set_xlabel('Total Flow (veh/h)')
    axes[0, 0].set_title('Top Detectors by Total Flow: Bridge vs Tunnel', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Average flow by location and scenario
    flow_by_location_scenario = df.groupby(['scenario', 'location'])['flow'].mean().unstack()
    # Reindex to ensure correct ordering
    sorted_scenarios = sort_scenarios(flow_by_location_scenario.index)
    flow_by_location_scenario = flow_by_location_scenario.reindex(sorted_scenarios)
    x2 = np.arange(len(flow_by_location_scenario.index))
    width2 = 0.35
    if 'Bridge' in flow_by_location_scenario.columns and 'Tunnel' in flow_by_location_scenario.columns:
        axes[0, 1].bar(x2 - width2/2, flow_by_location_scenario['Bridge'], width2,
                       label='Bridge', color='#2E86AB', alpha=0.8)
        axes[0, 1].bar(x2 + width2/2, flow_by_location_scenario['Tunnel'], width2,
                       label='Tunnel', color='#A23B72', alpha=0.8)
    axes[0, 1].set_title('Average Flow by Location and Scenario', fontweight='bold')
    axes[0, 1].set_xlabel('Scenario')
    axes[0, 1].set_ylabel('Average Flow (veh/h)')
    axes[0, 1].set_xticks(x2)
    axes[0, 1].set_xticklabels(flow_by_location_scenario.index, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Flow distribution: Bridge vs Tunnel
    bridge_flows = df[df['location'] == 'Bridge']['flow']
    tunnel_flows = df[df['location'] == 'Tunnel']['flow']
    axes[1, 0].hist(bridge_flows, bins=50, alpha=0.6, label='Bridge', color='#2E86AB')
    axes[1, 0].hist(tunnel_flows, bins=50, alpha=0.6, label='Tunnel', color='#A23B72')
    axes[1, 0].set_xlabel('Flow (veh/h)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Flow Distribution: Bridge vs Tunnel', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Vehicles entered by location and scenario
    veh_by_location = df.groupby(['scenario', 'location'])['nVehEntered'].sum().unstack()
    # Reindex to ensure correct ordering
    sorted_scenarios = sort_scenarios(veh_by_location.index)
    veh_by_location = veh_by_location.reindex(sorted_scenarios)
    x3 = np.arange(len(veh_by_location.index))
    if 'Bridge' in veh_by_location.columns and 'Tunnel' in veh_by_location.columns:
        axes[1, 1].bar(x3 - width2/2, veh_by_location['Bridge'], width2,
                       label='Bridge', color='#2E86AB', alpha=0.8)
        axes[1, 1].bar(x3 + width2/2, veh_by_location['Tunnel'], width2,
                       label='Tunnel', color='#A23B72', alpha=0.8)
    axes[1, 1].set_title('Total Vehicles Entered: Bridge vs Tunnel by Scenario', fontweight='bold')
    axes[1, 1].set_xlabel('Scenario')
    axes[1, 1].set_ylabel('Total Vehicles Entered')
    axes[1, 1].set_xticks(x3)
    axes[1, 1].set_xticklabels(veh_by_location.index, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loop_detector_statistics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loop_detector_statistics.png'}")
    plt.close()


def create_loop_summary_statistics(df, output_dir='plots'):
    """Create summary statistics table for loop data by Bridge vs Tunnel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    metrics = ['flow', 'speed', 'occupancy', 'nVehEntered', 'nVehContrib']
    
    # Filter out invalid speeds
    df_valid = df.copy()
    df_valid.loc[df_valid['speed'] < 0, 'speed'] = np.nan
    
    # Summary by scenario and location
    summary = df_valid.groupby(['scenario', 'location'])[metrics].agg(['mean', 'std'])
    
    # Save to CSV
    summary.to_csv(output_dir / 'loop_summary_statistics.csv')
    print(f"Saved: {output_dir / 'loop_summary_statistics.csv'}")
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(22, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table (show mean values by location)
    table_data = []
    for scenario in sort_scenarios(df['scenario'].unique()):
        for location in ['Bridge', 'Tunnel']:
            location_data = df_valid[(df_valid['scenario'] == scenario) & 
                                    (df_valid['location'] == location)]
            if not location_data.empty:
                row = [f"{scenario} - {location}"]
                for metric in metrics:
                    mean_val = location_data[metric].mean()
                    row.append(f"{mean_val:.2f}")
                table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Scenario - Location'] + [m.capitalize() for m in metrics],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Color header
    for i in range(len(metrics) + 1):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by location
    for i, row_data in enumerate(table_data, start=1):
        if 'Bridge' in row_data[0]:
            for j in range(len(metrics) + 1):
                table[(i, j)].set_facecolor('#E3F2FD')
        elif 'Tunnel' in row_data[0]:
            for j in range(len(metrics) + 1):
                table[(i, j)].set_facecolor('#FCE4EC')
    
    plt.title('Loop Data Summary Statistics: Bridge vs Tunnel', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'loop_summary_statistics_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loop_summary_statistics_table.png'}")
    plt.close()


def main():
    """Main function to run all visualizations."""
    print("=" * 60)
    print("SUMO Loop Detector Data Visualization Script")
    print("Focus: Bridge (e1_03N) vs Tunnel (e1_07N) Detectors")
    print("=" * 60)
    
    # Load data
    df = load_all_loop_data('Loop Data')
    
    if df.empty:
        print("No data to visualize!")
        return
    
    # Create output directory in subfolder
    output_dir = Path('plots/loop_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Delete old files
    print("\nCleaning up old files...")
    old_files = list(output_dir.glob('loop_*.png')) + list(output_dir.glob('loop_*.csv'))
    for old_file in old_files:
        old_file.unlink()
        print(f"  Deleted: {old_file.name}")
    
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    # Generate all plots
    plot_flow_time_series(df, output_dir)
    plot_speed_time_series(df, output_dir)
    plot_occupancy_time_series(df, output_dir)
    plot_version_comparison(df, output_dir)
    plot_detector_heatmap(df, output_dir)
    plot_flow_speed_relationship(df, output_dir)
    plot_detector_statistics(df, output_dir)
    create_loop_summary_statistics(df, output_dir)
    
    print("\n" + "=" * 60)
    print("All visualizations completed!")
    print(f"Plots saved in: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

