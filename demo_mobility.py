"""
Demo script for 6G Network Mobility Simulation
Run this to see devices and base stations moving in the network
"""

import sys
import time
from network_6g import (
    Network6GSimulator, FederatedLearning6G, Device6G, 
    MovementPattern, NetworkSlice, DeviceStatus
)
import numpy as np

def print_device_info(device: Device6G):
    """Pretty print device information"""
    speed = np.sqrt(device.velocity[0]**2 + device.velocity[1]**2)
    print(f"  • {device.device_id}: {device.movement_pattern.value:15} "
          f"at ({device.location[0]:6.1f}, {device.location[1]:6.1f}) "
          f"Speed: {speed:5.2f} m/s | "
          f"Status: {device.status.value:10} | "
          f"Trust: {device.trust_score:.2f}")

def print_bs_info(bs, idx: int):
    """Pretty print base station information"""
    speed = np.sqrt(bs.velocity[0]**2 + bs.velocity[1]**2) if bs.is_mobile else 0
    bs_type = "Mobile" if bs.is_mobile else "Fixed "
    print(f"  [{bs_type}] {bs.bs_id}: at ({bs.location[0]:6.1f}, {bs.location[1]:6.1f}) "
          f"Speed: {speed:5.2f} m/s | Connected: {len(bs.connected_devices)}/{bs.max_devices}")

def run_demo():
    """Run mobility simulation demo"""
    print("=" * 80)
    print("🌐 6G NETWORK MOBILITY SIMULATION DEMO")
    print("=" * 80)
    
    # Create network with mobile base stations
    print("\n📡 Initializing 6G Network...")
    network = Network6GSimulator(
        num_base_stations=5,      # Total base stations
        area_size=(5000, 5000),   # 5km x 5km area
        mobile_bs_count=2         # 2 mobile/aerial base stations
    )
    
    fl_6g = FederatedLearning6G(network)
    
    # Create diverse devices
    print(f"📱 Creating devices with different mobility patterns...")
    devices = fl_6g.create_fl_devices(15, "synthetic_network_dataset.csv")
    
    print(f"\n✅ Network initialized with {len(devices)} devices and {len(network.base_stations)} base stations")
    print(f"   Coverage Area: {network.area_size[0]}m x {network.area_size[1]}m")
    
    # Initial status
    print("\n" + "=" * 80)
    print("INITIAL NETWORK STATE")
    print("=" * 80)
    
    print("\n📡 Base Stations:")
    for idx, bs in enumerate(network.base_stations):
        print_bs_info(bs, idx)
    
    print("\n📱 Devices:")
    for device in list(network.devices.values())[:10]:  # Show first 10
        print_device_info(device)
    if len(network.devices) > 10:
        print(f"  ... and {len(network.devices) - 10} more devices")
    
    # Simulate trust scoring (some devices become suspicious/blocked)
    print("\n" + "=" * 80)
    print("🔒 APPLYING TRUST SCORES (Simulating Security Analysis)")
    print("=" * 80)
    
    trust_scores = {}
    for i, device_id in enumerate(network.devices.keys()):
        if i < 2:  # Make 2 devices suspicious
            trust_scores[device_id] = 0.80
        elif i < 3:  # Make 1 device blocked
            trust_scores[device_id] = 0.65
        else:
            trust_scores[device_id] = 0.95 + np.random.rand() * 0.05
    
    network.update_trust_scores(trust_scores)
    
    status = network.get_network_status()
    print(f"✅ Active: {status['active_devices']}")
    print(f"⚠️  Suspicious: {status['suspicious_devices']}")
    print(f"🚫 Blocked: {status['blocked_devices']}")
    
    # Run mobility simulation
    print("\n" + "=" * 80)
    print("🚀 RUNNING MOBILITY SIMULATION (30 seconds)")
    print("=" * 80)
    
    handoff_count = 0
    initial_log_count = len(network.network_logs)
    
    print("\nSimulating network activity...")
    for step in range(30):
        network.step_simulation(time_delta=1.0)
        
        # Show progress every 5 steps
        if (step + 1) % 5 == 0:
            mobility_stats = network.get_mobility_statistics()
            print(f"  t={step+1}s: Avg speed={mobility_stats['avg_device_speed']:.2f} m/s, "
                  f"Max speed={mobility_stats['max_device_speed']:.2f} m/s")
            
            # Count handoffs
            new_handoffs = sum(1 for log in network.network_logs[initial_log_count:] 
                             if 'Handoff' in log['message'])
            if new_handoffs > handoff_count:
                print(f"          🔄 {new_handoffs - handoff_count} handoff(s) detected!")
                handoff_count = new_handoffs
    
    # Final status
    print("\n" + "=" * 80)
    print("FINAL NETWORK STATE (After 30s of movement)")
    print("=" * 80)
    
    print("\n📡 Base Stations:")
    for idx, bs in enumerate(network.base_stations):
        print_bs_info(bs, idx)
    
    print("\n📱 Sample Devices (showing position changes):")
    for device in list(network.devices.values())[:5]:
        print_device_info(device)
        if len(device.trajectory_history) > 1:
            start_pos = device.trajectory_history[0]
            end_pos = device.trajectory_history[-1]
            distance = np.sqrt((end_pos[0] - start_pos[0])**2 + 
                             (end_pos[1] - start_pos[1])**2)
            print(f"      ↳ Moved {distance:.1f}m from initial position")
    
    # Statistics
    print("\n" + "=" * 80)
    print("📊 SIMULATION STATISTICS")
    print("=" * 80)
    
    mobility_stats = network.get_mobility_statistics()
    print(f"\n⏱️  Simulation Time: {mobility_stats['simulation_time']:.1f} seconds")
    print(f"📱 Total Devices: {mobility_stats['total_devices']}")
    print(f"🚀 Avg Device Speed: {mobility_stats['avg_device_speed']:.2f} m/s "
          f"({mobility_stats['avg_device_speed'] * 3.6:.1f} km/h)")
    print(f"⚡ Max Device Speed: {mobility_stats['max_device_speed']:.2f} m/s "
          f"({mobility_stats['max_device_speed'] * 3.6:.1f} km/h)")
    print(f"📡 Mobile Base Stations: {mobility_stats['mobile_base_stations']}")
    if mobility_stats['mobile_base_stations'] > 0:
        print(f"🛸 Avg BS Speed: {mobility_stats['avg_bs_speed']:.2f} m/s "
              f"({mobility_stats['avg_bs_speed'] * 3.6:.1f} km/h)")
    
    # Handoff analysis
    handoff_logs = [log for log in network.network_logs if 'Handoff' in log['message']]
    print(f"\n🔄 Total Handoffs: {len(handoff_logs)}")
    
    if handoff_logs:
        print("\n   Recent Handoffs:")
        for log in handoff_logs[-5:]:  # Show last 5
            print(f"   • {log['message']}")
    
    # Movement pattern distribution
    pattern_counts = {}
    for device in network.devices.values():
        pattern = device.movement_pattern.value
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print("\n🔄 Movement Pattern Distribution:")
    for pattern, count in sorted(pattern_counts.items()):
        percentage = (count / len(network.devices)) * 100
        print(f"   • {pattern:15}: {count:2} devices ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ SIMULATION COMPLETE!")
    print("=" * 80)
    print("\n💡 To visualize this network, run:")
    print("   python3 visualize_network.py  (requires matplotlib)")
    print("\n💡 To run federated learning with this network:")
    print("   python3 main.py --data synthetic_network_dataset.csv --num-clients 12 --rounds 10")
    print("\n")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
