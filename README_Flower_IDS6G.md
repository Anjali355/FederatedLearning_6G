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
- Base station simulation  
- Device mobility tracking  

### Security Features
- Trust score evaluation  
- Client blocking mechanism  
- Anomaly detection  
- Network performance monitoring  

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

Project Structure

flower_federated_ids_6g.py — main script for federated learning

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