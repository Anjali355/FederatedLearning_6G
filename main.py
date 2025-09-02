import argparse, os, json
import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from flwr.common import Context

from model import MLP, set_model_params
from client import ClientConfig, FlowerClient
from utils import set_seed, minmax_scale, make_partitions
from strategy import TrustFedAvg
from plot_results import plot_training_curves, plot_model_eval
from network_6g import Network6GSimulator, FederatedLearning6G, NetworkSlice, DeviceStatus


class Integrated6GTrustFedAvg(TrustFedAvg):
    """Enhanced TrustFedAvg that updates 6G network with trust scores"""
    
    def __init__(self, network_6g=None, fl_6g=None, **kwargs):
        super().__init__(**kwargs)
        self.network_6g = network_6g
        self.fl_6g = fl_6g
        
        
        
        # Reset tracking variables
        # self.client_stats = {}
        # self.trust_scores = {}
        # self.parameter_history.clear()
        # self.global_model_history.clear()
        # self.round_metrics.clear()
        # self.detection_history.clear()
        # self.blocked_clients.clear()
        
    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        # Update network with current trust scores
        if self.network_6g:
            self.network_6g.update_trust_scores(self.trust_scores)
            
            # Force status update based on trust scores
            for client_id, trust_score in self.trust_scores.items():
                if client_id in self.network_6g.devices:
                    device = self.network_6g.devices[client_id]
                    if trust_score < 0.85:
                        device.status = DeviceStatus.BLOCKED
                        self.network_6g.blocked_devices.add(client_id)
                    elif trust_score < 0.97:
                        device.status = DeviceStatus.SUSPICIOUS
                    else:
                        device.status = DeviceStatus.ACTIVE
        
        return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV with 'label' column")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument("--client-frac", type=float, default=0.75)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dp-sigma", type=float, default=0.3)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--malicious-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--outdir", type=str, default="artifacts_flower")
    args = parser.parse_args()

    # ------------------------------
    # Setup
    # ------------------------------
    set_seed(args.seed)
    torch_device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------
    # Load dataset
    # ------------------------------
    df = pd.read_csv(args.data)
    y = df["label"].values.astype(np.int64)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    X = minmax_scale(X)

    # Train/test split
    n = len(X)
    n_test = int(0.15 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Partition training data into clients
    K = args.num_clients
    parts = make_partitions(X_train, y_train, K, seed=args.seed)

    # Mark malicious clients
    mcount = max(1, int(args.malicious_frac * K))
    malicious_ids = set(rng.choice(np.arange(K), size=mcount, replace=False))

    # ------------------------------
    # Initialize 6G Network Simulation
    # ------------------------------
    print("[INFO] Initializing 6G network simulation...")
    # Increase base stations to ensure all devices can connect
    network_6g = Network6GSimulator(num_base_stations=max(5, K))
    fl_6g = FederatedLearning6G(network_6g)
    
    # Create 6G devices with better connection success rate
    devices_6g = []
    connection_attempts = 0
    max_attempts = K * 3  # Allow multiple attempts
    
    for i in range(K):
        connection_attempts += 1
        # Create device with more spread out locations to improve connection success
        device_created = fl_6g.create_single_device(
            device_id=str(i), 
            dataset_path=args.data,
            force_connect=True  # We'll add this method
        )
        if device_created:
            devices_6g.append(device_created)
        
        if connection_attempts > max_attempts:
            break
    
    print(f"[INFO] 6G Network initialized with {len(devices_6g)}/{K} devices successfully connected")
    
    # Show initial network topology
    print("\n[6G-TOPOLOGY] Initial Base Station Status:")
    for bs in network_6g.base_stations:
        connected = len(bs.connected_devices)
        print(f"  {bs.bs_id}: {connected} devices connected at location {bs.location}")

    # ------------------------------
    # Define client function
    # ------------------------------
    def client_fn(cid: str):
        k = int(cid)
        P = parts[k]
        cfg = ClientConfig(
            cid=k,
            X=X_train[P],
            y=y_train[P],
            device= torch_device,  # Fixed variable name
            dp_sigma=args.dp_sigma,
            clip_norm=args.clip_norm,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            malicious=(k in malicious_ids),
        )
        return FlowerClient(cfg, d_in=X.shape[1]).to_client()

    # ------------------------------
    # Strategy with 6G Integration
    # ------------------------------
    testset = (X_test, y_test)
    strategy = Integrated6GTrustFedAvg(
        network_6g=network_6g,
        fl_6g=fl_6g,
        testset=testset,
        device= torch_device,
        outdir=args.outdir,
        fraction_fit=args.client_frac,
        min_fit_clients=max(1, int(args.client_frac * K)),
        min_available_clients=K,
    )
    print(f"[DEBUG] Using integrated strategy: {type(strategy).__name__}")

    # ------------------------------
    # Run simulation with 6G integration
    # ------------------------------
    print("\n[INFO] Starting integrated 6G + Federated Learning simulation...")
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=K,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # ------------------------------
    # Display 6G Network Results
    # ------------------------------
    print("\n=== 6G Network Simulation Results ===")
    final_status = network_6g.get_network_status()
    print(f"Final network status: {final_status}")
    
    # Show malicious client detection results
    print(f"\nMalicious Clients (from client_flags.json): {malicious_ids}")
    print(f"6G Network Detection Results:")
    for device_id, device in network_6g.devices.items():
        status_symbol = "🔴" if device.status.value == "blocked" else \
                       "🟡" if device.status.value == "suspicious" else "🟢"
        print(f"  Device {device_id}: {status_symbol} {device.status.value} (trust: {device.trust_score:.3f})")
    
    print(f"\nSecurity Events ({len(network_6g.security_events)} total):")
    if network_6g.security_events:
        for event in network_6g.security_events[-10:]:  # Show last 10 events
            print(f"  Device {event['device_id']}: {event['reason']} -> {event['action']} (trust: {event['trust_score']:.3f})")
    else:
        print("  No security events detected")
    
    # Show base station utilization
    print(f"\nBase Station Utilization:")
    for bs_info in final_status['base_stations']:
        if bs_info['connected_count'] > 0:
            print(f"  {bs_info['bs_id']}: {bs_info['connected_count']} devices ({bs_info['utilization']:.1%} utilization)")

    # ------------------------------
    # Save final global model
    # ------------------------------
    model = MLP(d_in=X.shape[1]).to(torch_device)  # Fixed variable name
    try:
        last_params = hist.global_model_parameters
        if last_params is not None:
            set_model_params(model, parameters_to_ndarrays(last_params))
    except Exception:
        pass
    torch.save(model.state_dict(), os.path.join(args.outdir, "global_model.pt"))

    # Save malicious info + 6G network state
    with open(os.path.join(args.outdir, "client_flags.json"), "w") as f:
        json.dump({int(i): (i in malicious_ids) for i in range(K)}, f, indent=2)
    
    # Save 6G network results
    network_results = {
        "final_status": final_status,
        "security_events": network_6g.security_events,
        "device_trust_scores": {did: d.trust_score for did, d in network_6g.devices.items()},
        "malicious_detected": [did for did, d in network_6g.devices.items() 
                              if d.status in [DeviceStatus.SUSPICIOUS, DeviceStatus.BLOCKED]]
    }
    with open(os.path.join(args.outdir, "6g_network_results.json"), "w") as f:
        json.dump(network_results, f, indent=2, default=str)

    # ------------------------------
    # Generate plots
    # ------------------------------
    print("[INFO] Generating training plots...")
    plot_training_curves(
        log_csv=os.path.join(args.outdir, "training_log.csv"),
        outdir=args.outdir,
    )

    print("[INFO] Generating evaluation plots (ROC, Confusion Matrix)...")
    plot_model_eval(
        model_path=os.path.join(args.outdir, "global_model.pt"),
        X_test=X_test,
        y_test=y_test,
        outdir=args.outdir,
    )

    print(f"[DONE] Results stored in {args.outdir}")


if __name__ == "__main__":
    main()