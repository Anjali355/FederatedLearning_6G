"""
Network Visualization for 6G Federated Learning
Visualizes device and base station mobility, network topology, and handoffs
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Dict, List
from network_6g import Network6GSimulator, Device6G, DeviceStatus, MovementPattern

class NetworkVisualizer:
    def __init__(self, network_sim: Network6GSimulator):
        self.network = network_sim
        self.fig, (self.ax_map, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize plot elements"""
        self.ax_map.set_xlim(0, self.network.area_size[0])
        self.ax_map.set_ylim(0, self.network.area_size[1])
        self.ax_map.set_xlabel('X Position (m)', fontsize=12)
        self.ax_map.set_ylabel('Y Position (m)', fontsize=12)
        self.ax_map.set_title('6G Network Topology & Mobility', fontsize=14, fontweight='bold')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_aspect('equal')
        
        # Stats plot
        self.ax_stats.set_title('Network Statistics', fontsize=14, fontweight='bold')
        self.ax_stats.axis('off')
        
    def plot_snapshot(self, save_path: str = None):
        """Plot current network state"""
        self.ax_map.clear()
        self.setup_plot()
        
        # Plot base station coverage areas
        for bs in self.network.base_stations:
            color = 'skyblue' if not bs.is_mobile else 'lightcoral'
            alpha = 0.15 if not bs.is_mobile else 0.2
            circle = plt.Circle(bs.location, bs.coverage_radius, 
                              color=color, alpha=alpha, linestyle='--')
            self.ax_map.add_patch(circle)
            
            # Plot base station
            marker = '^' if bs.is_mobile else 's'
            label = 'Mobile BS' if bs.is_mobile else 'Fixed BS'
            self.ax_map.scatter(*bs.location, s=300, marker=marker, 
                              color='red' if bs.is_mobile else 'blue',
                              edgecolors='black', linewidth=2, 
                              label=label, zorder=10)
            
            # BS ID label
            self.ax_map.text(bs.location[0], bs.location[1] + 100, 
                           bs.bs_id, ha='center', fontsize=9, fontweight='bold')
            
            # Show patrol route for mobile BS
            if bs.is_mobile and bs.patrol_waypoints:
                waypoints = np.array(bs.patrol_waypoints + [bs.patrol_waypoints[0]])
                self.ax_map.plot(waypoints[:, 0], waypoints[:, 1], 
                               'r--', alpha=0.5, linewidth=1)
                self.ax_map.scatter(waypoints[:, 0], waypoints[:, 1], 
                                  s=50, c='red', marker='x', alpha=0.6)
            
            # Show trajectory
            if len(bs.trajectory_history) > 1:
                trajectory = np.array(bs.trajectory_history)
                self.ax_map.plot(trajectory[:, 0], trajectory[:, 1], 
                               'r-', alpha=0.3, linewidth=1)
        
        # Plot devices
        status_colors = {
            DeviceStatus.ACTIVE: 'green',
            DeviceStatus.SUSPICIOUS: 'orange',
            DeviceStatus.BLOCKED: 'red',
            DeviceStatus.OFFLINE: 'gray'
        }
        
        pattern_markers = {
            MovementPattern.STATIC: 'o',
            MovementPattern.RANDOM_WALK: 's',
            MovementPattern.DIRECTIONAL: '^',
            MovementPattern.CIRCULAR: 'D',
            MovementPattern.WAYPOINT: 'p'
        }
        
        for device in self.network.devices.values():
            color = status_colors.get(device.status, 'gray')
            marker = pattern_markers.get(device.movement_pattern, 'o')
            
            # Plot device position
            self.ax_map.scatter(*device.location, s=100, marker=marker, 
                              color=color, alpha=0.8, edgecolors='black', 
                              linewidth=1, zorder=5)
            
            # Show velocity vector
            if device.movement_pattern != MovementPattern.STATIC:
                speed = np.sqrt(device.velocity[0]**2 + device.velocity[1]**2)
                if speed > 0.1:
                    scale = 100  # Arrow scale
                    self.ax_map.arrow(device.location[0], device.location[1],
                                    device.velocity[0] * scale, device.velocity[1] * scale,
                                    head_width=30, head_length=40, fc=color, 
                                    ec='black', alpha=0.6, linewidth=0.5)
            
            # Show trajectory
            if len(device.trajectory_history) > 2:
                trajectory = np.array(device.trajectory_history[-20:])  # Last 20 positions
                self.ax_map.plot(trajectory[:, 0], trajectory[:, 1], 
                               color=color, alpha=0.3, linewidth=1)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color='green', label=f'Active ({len([d for d in self.network.devices.values() if d.status == DeviceStatus.ACTIVE])})'),
            mpatches.Patch(color='orange', label=f'Suspicious ({len([d for d in self.network.devices.values() if d.status == DeviceStatus.SUSPICIOUS])})'),
            mpatches.Patch(color='red', label=f'Blocked ({len([d for d in self.network.devices.values() if d.status == DeviceStatus.BLOCKED])})'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                      markersize=10, label='Fixed BS'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=10, label='Mobile BS'),
        ]
        self.ax_map.legend(handles=legend_elements, loc='upper right', 
                         fontsize=10, framealpha=0.9)
        
        # Update statistics panel
        self._update_stats_panel()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Network visualization saved to {save_path}")
        else:
            plt.show()
    
    def _update_stats_panel(self):
        """Update statistics panel"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        stats = self.network.get_network_status()
        mobility_stats = self.network.get_mobility_statistics()
        
        stats_text = f"""
        📊 Network Statistics (t={self.network.simulation_time:.1f}s)
        
        🔷 Devices:
          • Total: {stats['total_devices']}
          • Active: {stats['active_devices']}
          • Suspicious: {stats['suspicious_devices']}
          • Blocked: {stats['blocked_devices']}
        
        🔶 Base Stations:
          • Total: {len(self.network.base_stations)}
          • Mobile: {mobility_stats['mobile_base_stations']}
        
        🚀 Mobility:
          • Avg Device Speed: {mobility_stats['avg_device_speed']:.2f} m/s
          • Max Device Speed: {mobility_stats['max_device_speed']:.2f} m/s
          • Avg BS Speed: {mobility_stats['avg_bs_speed']:.2f} m/s
        
        📡 Base Station Utilization:
        """
        
        for bs_info in stats['base_stations']:
            stats_text += f"\n  • {bs_info['bs_id']}: {bs_info['connected_count']}/{self.network.base_stations[0].max_devices} "
            stats_text += f"({bs_info['utilization']*100:.1f}%)"
        
        # Movement pattern distribution
        pattern_counts = {}
        for device in self.network.devices.values():
            pattern = device.movement_pattern.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        stats_text += "\n\n🔄 Movement Patterns:"
        for pattern, count in pattern_counts.items():
            stats_text += f"\n  • {pattern}: {count}"
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=11, verticalalignment='top', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    def animate(self, duration: int = 60, interval: int = 100, save_path: str = None):
        """Create animation of network mobility"""
        frames = duration * 1000 // interval  # Number of frames
        
        def update(frame):
            # Step simulation
            self.network.step_simulation(time_delta=interval / 1000.0)
            
            # Redraw
            self.ax_map.clear()
            self.setup_plot()
            self.plot_snapshot()
            
            return self.ax_map,
        
        anim = FuncAnimation(self.fig, update, frames=frames, 
                           interval=interval, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()
        
        return anim


def plot_handoff_analysis(network_sim: Network6GSimulator, save_path: str = None):
    """Analyze and plot handoff statistics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count handoffs from logs
    handoff_counts = {}
    for log in network_sim.network_logs:
        if 'Handoff' in log['message']:
            # Extract device ID
            parts = log['message'].split()
            if len(parts) > 1:
                device_id = parts[1]
                handoff_counts[device_id] = handoff_counts.get(device_id, 0) + 1
    
    # Plot 1: Handoff counts per device
    if handoff_counts:
        devices = list(handoff_counts.keys())
        counts = list(handoff_counts.values())
        ax1.bar(devices, counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Device ID', fontsize=12)
        ax1.set_ylabel('Number of Handoffs', fontsize=12)
        ax1.set_title('Handoffs per Device', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Device status distribution over time
    status_counts = {
        'Active': 0,
        'Suspicious': 0,
        'Blocked': 0,
        'Offline': 0
    }
    
    for device in network_sim.devices.values():
        status_counts[device.status.value.capitalize()] = status_counts.get(
            device.status.value.capitalize(), 0) + 1
    
    colors = ['green', 'orange', 'red', 'gray']
    ax2.pie(status_counts.values(), labels=status_counts.keys(), 
           colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Current Device Status Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Handoff analysis saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    from network_6g import FederatedLearning6G
    
    print("Creating 6G network simulation...")
    network = Network6GSimulator(num_base_stations=5, area_size=(5000, 5000), mobile_bs_count=2)
    fl_6g = FederatedLearning6G(network)
    
    # Create devices
    print("Creating federated learning devices...")
    devices = fl_6g.create_fl_devices(20, "synthetic_network_dataset.csv")
    print(f"Created {len(devices)} devices")
    
    # Run simulation for a bit
    print("Running mobility simulation...")
    for _ in range(30):
        network.step_simulation()
    
    # Visualize
    print("Creating visualization...")
    visualizer = NetworkVisualizer(network)
    visualizer.plot_snapshot(save_path='artifacts_flower/network_snapshot.png')
    
    # Handoff analysis
    plot_handoff_analysis(network, save_path='artifacts_flower/handoff_analysis.png')
    
    print("\nDone! Check artifacts_flower/ for visualizations")
