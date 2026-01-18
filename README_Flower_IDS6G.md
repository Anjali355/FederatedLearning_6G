# Federated Learning with 6G Network Security

## Overview
This project implements a secure federated learning system integrated with a 6G network simulation. It features trust-based client evaluation, malicious behavior detection, and network slice management.

## Features

### Trust-Based Federated Learning
- Dynamic trust score calculation  
- Malicious client detection  
- Client behavior monitoring  

### 6G Network Integration
- Network slice management (eMBB, URLLC, mMTC)  
- Base station simulation (fixed and mobile/aerial)
- **Realistic device mobility simulation** with multiple movement patterns:
  - Static devices (IoT sensors)
  - Random walk (pedestrians)
  - Directional movement (vehicles)
  - Circular patterns
- **Dynamic handoff management** between base stations
- **Mobile base station** support with patrol routes
- Real-time network topology visualization
- Movement trajectory tracking

### Security Features
- Trust score evaluation  
- Client blocking mechanism  
- Anomaly detection  
- Network performance monitoring
- Dynamic security policies based on device status  

## Setup

### Prerequisites
Install the required Python libraries:

- PyTorch  
- Flower (Federated Learning)  
- NumPy  
- Pandas  
- Scikit-learn  

### Installation
```bash
pip install flwr torch pandas numpy scikit-learn

Usage
Running the System
python flower_federated_ids_6g.py --data ids6g_synthetic.csv --num-clients 12 --rounds 12

Command Line Arguments

--data: Path to dataset CSV file

--rounds: Number of training rounds

--num-clients: Number of federated clients

--client-frac: Fraction of clients per round

--local-epochs: Local training epochs

--batch-size: Training batch size

--lr: Learning rate

--dp-sigma: Differential privacy noise

--malicious-frac: Fraction of malicious clients

--seed: Random seed

--cpu: Force CPU usage

--outdir: Output directory

## Network Mobility Simulation

### Running Network Visualization
Visualize the 6G network with moving devices and base stations:

```bash
python visualize_network.py
```

This will:
- Create a snapshot of the network topology
- Show device movements and trajectories
- Display base station coverage areas
- Visualize handoffs between base stations
- Generate statistics on mobility patterns

### Movement Patterns

Devices in the network follow different mobility models:

1. **Static** (20% of devices) - IoT sensors, fixed infrastructure
   - Speed: 0 m/s
   - Examples: Smart sensors, traffic cameras

2. **Random Walk** (30% of devices) - Pedestrians, mobile workers
   - Speed: ~1.5 m/s (5 km/h)
   - Random direction changes

3. **Directional** (30% of devices) - Vehicles, drones
   - Speed: ~15 m/s (50 km/h)
   - Maintains direction with occasional turns

4. **Circular** (20% of devices) - Patrol routes, coverage areas
   - Speed: ~8 m/s
   - Circular movement patterns

### Mobile Base Stations

The system supports mobile/aerial base stations that:
- Patrol predefined waypoints
- Provide extended coverage
- Trigger automatic handoffs for connected devices
- Move at speeds up to 50 m/s

### Key Mobility Features

- **Automatic Handoffs**: Devices automatically switch to better base stations
- **Trajectory Tracking**: Full history of device and BS movements
- **Battery Drain**: Movement affects device battery levels
- **Boundary Handling**: Devices bounce at simulation area edges
- **Real-time Updates**: Network topology updates every simulation step

Project Structure

flower_federated_ids_6g.py — main script for federated learning
network_6g.py — 6G network simulation with mobility
visualize_network.py — network visualization tools

artifacts_flower/ — stores logs, models, and simulation results

ids6g_synthetic.csv — example dataset

Output Files

global_model.pt — trained model parameters

client_flags.json — client status flags (benign/malicious)

6g_network_results.json — network simulation results

enhanced_trust_log.csv — trust score tracking

Monitoring

Training logs: training_log.csv (accuracy, suspicious clients, trust weights)

Trust scores and client status in artifacts_flower/

Network simulation results in JSON format

Training metrics and visualization plots

Reset System

To reset the system and start fresh, remove the contents of artifacts_flower/. and delete global_model.pt

Contributing

Fork the repository

Create a feature branch

Commit changes

Push to branch

Create a pull request

Created by [Anjali Rao]
Last Updated: September 2025