import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class DeviceStatus(Enum):
    ACTIVE = "active"
    SUSPICIOUS = "suspicious" 
    BLOCKED = "blocked"
    OFFLINE = "offline"

class NetworkSlice(Enum):
    EMBB = "enhanced_mobile_broadband"  # High data rate
    URLLC = "ultra_reliable_low_latency"  # Critical apps
    MMTC = "massive_machine_type"  # IoT devices

@dataclass
class Device6G:
    device_id: str
    slice_type: NetworkSlice
    location: tuple  # (x, y) coordinates
    signal_strength: float  # 0-1
    battery_level: float  # 0-1
    compute_capacity: float  # FLOPS
    bandwidth_mbps: float
    status: DeviceStatus = DeviceStatus.ACTIVE
    trust_score: float = 1.0
    connection_quality: float = 1.0

class BaseStation6G:
    def __init__(self, bs_id: str, location: tuple, coverage_radius: float = 1000):
        self.bs_id = bs_id
        self.location = location
        self.coverage_radius = coverage_radius
        self.connected_devices: Dict[str, Device6G] = {}
        self.max_devices = 50   # 6G massive connectivity
        
    def can_connect(self, device: Device6G) -> bool:
        # Check distance
        distance = np.sqrt((device.location[0] - self.location[0])**2 + 
                          (device.location[1] - self.location[1])**2)
        return distance <= self.coverage_radius and len(self.connected_devices) < self.max_devices
    
    def connect_device(self, device: Device6G) -> bool:
        if self.can_connect(device) and device.status != DeviceStatus.BLOCKED:
            self.connected_devices[device.device_id] = device
            return True
        return False

class Network6GSimulator:
    def __init__(self, num_base_stations: int = 4, area_size: tuple = (5000, 5000)):
        self.area_size = area_size
        self.base_stations = self._create_base_stations(num_base_stations)
        self.devices: Dict[str, Device6G] = {}
        self.network_logs = []
        self.security_events = []
        self.blocked_devices = set() #2
        
    def _create_base_stations(self, num_bs: int) -> List[BaseStation6G]:
        stations = []
        for i in range(num_bs):
            x = random.uniform(0, self.area_size[0])
            y = random.uniform(0, self.area_size[1])
            stations.append(BaseStation6G(f"BS_{i}", (x, y)))
        return stations
    
    def add_device(self, device: Device6G) -> bool:
        """Add device to network and connect to nearest base station"""
        if device.device_id in self.devices:
            return False
            
        # Find nearest base station
        best_bs = None
        best_distance = float('inf')
        
        for bs in self.base_stations:
            distance = np.sqrt((device.location[0] - bs.location[0])**2 + 
                             (device.location[1] - bs.location[1])**2)
            if distance < best_distance and bs.can_connect(device):
                best_distance = distance
                best_bs = bs
        
        if best_bs and best_bs.connect_device(device):
            self.devices[device.device_id] = device
            device.signal_strength = max(0.1, 1.0 - (best_distance / best_bs.coverage_radius))
            self._log_event(f"Device {device.device_id} connected to {best_bs.bs_id}")
            return True
        
        self._log_event(f"Device {device.device_id} failed to connect - no available base station")
        return False


    def update_trust_scores(self, trust_results: Dict[str, float]):
        """Update device trust scores and status"""
        updated_devices = {
            "suspicious": [],
            "blocked": [],
            "active": []
        }
        
        for device_id, trust_score in trust_results.items():
            if device_id in self.devices:
                device = self.devices[device_id]
                old_status = device.status
                device.trust_score = trust_score
                
                # Clear status from previous sets
                self.blocked_devices.discard(device_id)
                
                # Apply new status
                if trust_score < 0.7:
                    device.status = DeviceStatus.BLOCKED
                    self.blocked_devices.add(device_id)
                    updated_devices["blocked"].append(device_id)
                elif trust_score < 0.85:
                    device.status = DeviceStatus.SUSPICIOUS
                    updated_devices["suspicious"].append(device_id)
                else:
                    device.status = DeviceStatus.ACTIVE
                    updated_devices["active"].append(device_id)
                
                if old_status != device.status:
                    self._log_event(
                        f"Device {device_id} status changed: {old_status.value} -> {device.status.value} "
                        f"(trust={trust_score:.3f})"
                    )
        
        return updated_devices


    def _apply_security_policy(self, device: Device6G, reason: str):
        """Apply network security policies"""
        if device.status == DeviceStatus.SUSPICIOUS:
            # Reduce bandwidth and priority
            device.bandwidth_mbps *= 0.5
            device.connection_quality *= 0.7
            policy = f"Reduced bandwidth to {device.bandwidth_mbps:.1f} Mbps"
            
        elif device.status == DeviceStatus.BLOCKED:
            # Disconnect from all base stations
            for bs in self.base_stations:
                if device.device_id in bs.connected_devices:
                    del bs.connected_devices[device.device_id]
            device.bandwidth_mbps = 0
            device.connection_quality = 0
            policy = "Disconnected from network"
            
        elif device.status == DeviceStatus.ACTIVE:
            # Restore normal service
            device.bandwidth_mbps = self._get_slice_bandwidth(device.slice_type)
            device.connection_quality = device.signal_strength
            policy = "Service restored"
        
        security_event = {
            "timestamp": time.time(),
            "device_id": device.device_id,
            "reason": reason,
            "action": policy,
            "trust_score": device.trust_score
        }
        self.security_events.append(security_event)
        self._log_event(f"SECURITY: {device.device_id} - {reason} -> {policy}")
    
    def _get_slice_bandwidth(self, slice_type: NetworkSlice) -> float:
        """Get default bandwidth for network slice"""
        slice_bandwidths = {
            NetworkSlice.EMBB: 1000.0,    # 1 Gbps for high data rate
            NetworkSlice.URLLC: 100.0,    # 100 Mbps for low latency
            NetworkSlice.MMTC: 10.0       # 10 Mbps for IoT
        }
        return slice_bandwidths.get(slice_type, 100.0)
    
    def simulate_mobility(self, device_id: str):
        """Simulate device movement and handoffs"""
        if device_id not in self.devices:
            return
            
        device = self.devices[device_id]
        if device.status == DeviceStatus.BLOCKED:
            return
            
        # Random movement
        dx = random.uniform(-100, 100)
        dy = random.uniform(-100, 100)
        device.location = (
            max(0, min(self.area_size[0], device.location[0] + dx)),
            max(0, min(self.area_size[1], device.location[1] + dy))
        )
        
        # Check if handoff needed
        self._check_handoff(device)
    
    def _check_handoff(self, device: Device6G):
        """Check if device needs handoff to different base station"""
        current_bs = None
        for bs in self.base_stations:
            if device.device_id in bs.connected_devices:
                current_bs = bs
                break
        
        if current_bs:
            # Find better base station
            best_bs = current_bs
            best_signal = device.signal_strength
            
            for bs in self.base_stations:
                distance = np.sqrt((device.location[0] - bs.location[0])**2 + 
                                 (device.location[1] - bs.location[1])**2)
                signal = max(0.1, 1.0 - (distance / bs.coverage_radius))
                
                if signal > best_signal + 0.2 and bs.can_connect(device):  # Hysteresis
                    best_signal = signal
                    best_bs = bs
            
            # Perform handoff if beneficial
            if best_bs != current_bs:
                del current_bs.connected_devices[device.device_id]
                best_bs.connect_device(device)
                device.signal_strength = best_signal
                self._log_event(f"Handoff: {device.device_id} from {current_bs.bs_id} to {best_bs.bs_id}")
    
    def get_network_status(self) -> Dict:
        """Get current network status"""
        status = {
            "total_devices": len(self.devices),
            "active_devices": len([d for d in self.devices.values() if d.status == DeviceStatus.ACTIVE]),
            "suspicious_devices": len([d for d in self.devices.values() if d.status == DeviceStatus.SUSPICIOUS]),
            "blocked_devices": len([d for d in self.devices.values() if d.status == DeviceStatus.BLOCKED]),
            "base_stations": []
        }
        
        for bs in self.base_stations:
            bs_info = {
                "bs_id": bs.bs_id,
                "connected_count": len(bs.connected_devices),
                "utilization": len(bs.connected_devices) / bs.max_devices

            }
            status["base_stations"].append(bs_info)
        
        return status
    
    def _log_event(self, message: str):
        """Log network events"""
        log_entry = {
            "timestamp": time.time(),
            "message": message
        }
        self.network_logs.append(log_entry)
        print(f"[6G-NET] {message}")

# Integration with existing federated learning code
class FederatedLearning6G:
    def __init__(self, network_sim: Network6GSimulator):
        self.network = network_sim
        
    def create_fl_devices(self, num_devices: int, dataset_path: str):
        """Create 6G devices that participate in federated learning"""
        # This would integrate with existing client creation logic
        devices = []
        
        for i in range(num_devices):
            # Random 6G device properties
            slice_type = random.choice(list(NetworkSlice))
            location = (random.uniform(0, self.network.area_size[0]), 
                       random.uniform(0, self.network.area_size[1]))
            
            device = Device6G(
                device_id=str(i),
                slice_type=slice_type,
                location=location,
                signal_strength=0.8,
                battery_level=random.uniform(0.3, 1.0),
                compute_capacity=random.uniform(1e9, 1e12),  # FLOPS
                bandwidth_mbps=100.0
            )
            
            # Add to network
            if self.network.add_device(device):
                devices.append(device)
        
        return devices
    
    def run_federated_round(self, round_num: int, trust_weights: Dict[str, float]):
        """Simulate one federated learning round with 6G constraints"""
        print(f"\n[6G-FL] Starting federated round {round_num}")
        
        # Update trust scores and apply policies
        self.network.update_trust_scores(trust_weights)
        
        # Simulate device mobility during training
        for device_id in list(self.network.devices.keys()):
            if random.random() < 0.3:  # 30% chance of movement
                self.network.simulate_mobility(device_id)
        
        # Check which devices can participate based on 6G constraints
        participating_devices = self._select_participating_devices()
        
        # Simulate training time with 6G latency
        training_time = self._simulate_training_latency(participating_devices)
        
        print(f"[6G-FL] Round {round_num} completed in {training_time:.2f}s with {len(participating_devices)} devices")
        
        return participating_devices
    
    def _select_participating_devices(self) -> List[Device6G]:
        """Select devices that can participate based on 6G constraints"""
        candidates = []
        
        for device in self.network.devices.values():
            # 6G participation requirements
            can_participate = (
                device.status == DeviceStatus.ACTIVE and
                device.signal_strength > 0.5 and
                device.battery_level > 0.2 and
                device.connection_quality > 0.6
            )
            
            if can_participate:
                candidates.append(device)
            else:
                reason = "low signal" if device.signal_strength <= 0.5 else \
                        "low battery" if device.battery_level <= 0.2 else \
                        "poor connection" if device.connection_quality <= 0.6 else \
                        f"status: {device.status.value}"
                print(f"[6G-FL] Device {device.device_id} excluded: {reason}")
        
        return candidates
    
    def _simulate_training_latency(self, devices: List[Device6G]) -> float:
        """Simulate 6G network latency for federated learning"""
        base_latency = 0.001  # 1ms base 6G latency
        
        # Calculate latency based on network conditions
        max_latency = 0
        for device in devices:
            # Device-specific latency factors
            signal_penalty = (1.0 - device.signal_strength) * 0.01  # Up to 10ms penalty
            slice_latency = {
                NetworkSlice.URLLC: 0.001,  # 1ms
                NetworkSlice.EMBB: 0.004,   # 4ms  
                NetworkSlice.MMTC: 0.010    # 10ms
            }[device.slice_type]
            
            device_latency = base_latency + signal_penalty + slice_latency
            max_latency = max(max_latency, device_latency)
        
        # Add training computation time
        training_time = random.uniform(0.5, 2.0)  # Model training time
        total_time = max_latency + training_time
        
        return total_time
    
    # Add this method to FederatedLearning6G class in network_6g.py
    def create_single_device(self, device_id: str, dataset_path: str, force_connect: bool = False):
        """Create a single 6G device and ensure connection"""
        slice_type = random.choice(list(NetworkSlice))
        
        # Try multiple locations if force_connect is True
        max_attempts = 10 if force_connect else 1
        
        for attempt in range(max_attempts):
            location = (random.uniform(0, self.network.area_size[0]), 
                    random.uniform(0, self.network.area_size[1]))
            
            device = Device6G(
                device_id=device_id,
                slice_type=slice_type,
                location=location,
                signal_strength=0.8,
                battery_level=random.uniform(0.3, 1.0),
                compute_capacity=random.uniform(1e9, 1e12),
                bandwidth_mbps=self.network._get_slice_bandwidth(slice_type)
            )
            
            if self.network.add_device(device):
                return device
        
        return None


# Extension to your existing main.py
class Enhanced6GFederatedLearning:
    def __init__(self, original_main_function):
        self.original_main = original_main_function
        self.network_sim = Network6GSimulator()
        self.fl_6g = FederatedLearning6G(self.network_sim)
        
    def run_6g_simulation(self, args):
        """Run your existing FL code with 6G network simulation overlay"""
        print("=== Starting 6G Network Simulation ===")
        
        # Create 6G devices
        devices_6g = self.fl_6g.create_fl_devices(args.num_clients, args.data)
        
        # Show initial network status
        status = self.network_sim.get_network_status()
        print(f"Network initialized: {status['active_devices']} active devices across {len(self.network_sim.base_stations)} base stations")
        
        # Hook into your existing federated learning
        #Will modify your main.py to call this at each round
        
        return self.original_main(args)
    
    def on_round_complete(self, round_num: int, trust_weights: Dict[str, float]):
        """Call this after each FL round to update 6G network"""
        participating = self.fl_6g.run_federated_round(round_num, trust_weights)
        
        # Show network status after round
        status = self.network_sim.get_network_status()
        print(f"Network status after round {round_num}: {status}")
        
        return participating

# Simple usage example - add this to your main.py
def integrate_6g_simulation():
    """
    Add this function to your main.py to enable 6G simulation
    """
    # Initialize 6G network
    network_6g = Network6GSimulator(num_base_stations=3)
    fl_6g = FederatedLearning6G(network_6g)
    
    # Create devices (you'd call this before your FL starts)
    devices = fl_6g.create_fl_devices(6, "your_dataset.csv")
    
    # In your FL loop, after each round, call:
    # trust_weights = {"0": 0.33, "1": 0.33, "2": 0.33}  # From your FL results
    # participating = fl_6g.run_federated_round(round_num, trust_weights)
    
    return network_6g, fl_6g
