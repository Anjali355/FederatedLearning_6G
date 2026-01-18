import argparse, os, json
import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from flwr.common import Context

from model import MLP, set_model_params, get_model_params
from client import ClientConfig, FlowerClient
from utils import set_seed, minmax_scale, make_partitions
from strategy import TrustFedAvg
from plot_results import plot_training_curves, plot_model_eval
from network_6g import Network6GSimulator, FederatedLearning6G, NetworkSlice, DeviceStatus
from interactive_dashboard import InteractiveDashboard, create_network_status_for_dashboard


class Integrated6GTrustFedAvg(TrustFedAvg):
    """Enhanced TrustFedAvg that updates 6G network with trust scores"""
    
    def __init__(self, network_6g=None, fl_6g=None, dashboard=None, **kwargs):
        super().__init__(**kwargs)
        self.network_6g = network_6g
        self.fl_6g = fl_6g
        self.dashboard = dashboard
        
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
    parser.add_argument("--dashboard", action="store_true", help="Enable live visualization dashboard")
    
    # Priority 1: Security & Trust Features
    parser.add_argument("--aggregation", type=str, default="fedavg", 
                       choices=["fedavg", "krum", "multi-krum", "trimmed-mean", "median"],
                       help="Byzantine-robust aggregation method")
    parser.add_argument("--use-dp", action="store_true", help="Enable differential privacy")
    parser.add_argument("--dp-epsilon", type=float, default=1.0, help="Privacy budget (epsilon)")
    parser.add_argument("--dp-delta", type=float, default=1e-5, help="Privacy failure probability (delta)")
    parser.add_argument("--use-trust-prediction", action="store_true", 
                       help="Enable trust score prediction and early warning")
    parser.add_argument("--attack-type", type=str, default="none",
                       choices=["none", "backdoor", "poisoning", "sybil", "byzantine"],
                       help="Simulate specific attack type")
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
        # Return NumPyClient directly (don't wrap with to_client for custom simulation)
        return FlowerClient(cfg, d_in=X.shape[1])

    # ------------------------------
    # Strategy with 6G Integration + Dashboard
    # ------------------------------
    # Initialize dashboard (if enabled)
    # ------------------------------
    dashboard = None
    if args.dashboard:
        print("[INFO] Initializing live dashboard...")
        dashboard = InteractiveDashboard(outdir=args.outdir)
        print(f"[INFO] Dashboard will be available at http://localhost:8050")
        print("[INFO] Note: Dashboard reads from CSV files - start it separately with:")
        print(f"      python interactive_dashboard.py")
    else:
        print("[INFO] Dashboard disabled (use --dashboard to enable)")
    
    testset = (X_test, y_test)
    strategy = Integrated6GTrustFedAvg(
        network_6g=network_6g,
        fl_6g=fl_6g,
        dashboard=dashboard,
        testset=testset,
        device=torch_device,
        outdir=args.outdir,
        fraction_fit=args.client_frac,
        min_fit_clients=max(1, int(args.client_frac * K)),
        min_available_clients=K,
        # Priority 1 features
        aggregation_method=args.aggregation,
        use_differential_privacy=args.use_dp,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        use_trust_prediction=args.use_trust_prediction,
    )
    print(f"[DEBUG] Using integrated strategy: {type(strategy).__name__}")
    print(f"[SECURITY] Aggregation: {args.aggregation}, DP: {args.use_dp}, Trust Prediction: {args.use_trust_prediction}")

    # ------------------------------
    # Run simulation with 6G integration (Python 3.14 Compatible)
    # ------------------------------
    print("\n[INFO] Starting integrated 6G + Federated Learning simulation...")
    print("[INFO] Using sequential simulation mode (Python 3.14 compatible - Ray not available)")
    
    # Custom simulation loop for Python 3.14 compatibility (Ray not supported yet)
    from collections import defaultdict
    hist_dict = defaultdict(list)
    
    # Initialize global model
    global_model = MLP(d_in=X.shape[1]).to(torch_device)
    global_params = get_model_params(global_model)
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{args.rounds}")
        print(f"{'='*60}")
        
        # Sample clients for this round
        num_clients_this_round = max(1, int(args.client_frac * K))
        sampled_client_ids = np.random.choice(K, size=num_clients_this_round, replace=False)
        
        # Collect client updates and metrics for dashboard
        results = []
        failures = []
        client_metrics_for_dashboard = {}
        
        for cid in sampled_client_ids:
            try:
                client = client_fn(str(cid))
                params, num_examples, metrics = client.fit(global_params, {})
                results.append((str(cid), fl.common.FitRes(
                    status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
                    parameters=fl.common.ndarrays_to_parameters(params),
                    num_examples=num_examples,
                    metrics=metrics
                )))
                # Store client metrics for dashboard
                client_metrics_for_dashboard[int(cid)] = metrics
            except Exception as e:
                print(f"[ERROR] Client {cid} failed: {e}")
                failures.append((str(cid), str(e)))
        
        # Aggregate updates
        aggregated_params, aggregated_metrics = strategy.aggregate_fit(round_num, results, failures)
        
        if aggregated_params is not None:
            global_params = fl.common.parameters_to_ndarrays(aggregated_params)
        
        # Update dashboard with round metrics
        test_acc = aggregated_metrics.get('test_acc', 0) if aggregated_metrics else 0
        test_loss = aggregated_metrics.get('test_loss', 0) if aggregated_metrics else 0
        
        # Enrich client metrics with trust and detection scores
        for cid in client_metrics_for_dashboard:
            cid_str = str(cid)
            if cid_str in strategy.client_stats:
                client_metrics_for_dashboard[cid]['trust_score'] = strategy.trust_scores.get(cid_str, 1.0)
                client_metrics_for_dashboard[cid]['detection_score'] = sum(strategy.client_stats[cid_str].get('detection_scores', {}).values())
                client_metrics_for_dashboard[cid]['weight'] = strategy.client_stats[cid_str].get('final_weight', 0)
        
        # Dashboard updates automatically by reading CSV files
        # No need to call update_round() - it monitors artifacts_folder/ directory
        
        # Record metrics
        if aggregated_metrics:
            for key, value in aggregated_metrics.items():
                hist_dict[key].append(value)
    
    # Convert to history object format
    class SimulationHistory:
        def __init__(self, metrics_distributed, losses_distributed):
            self.metrics_distributed = metrics_distributed
            self.losses_distributed = losses_distributed
    
    hist = SimulationHistory(
        metrics_distributed={"test_acc": [(r, hist_dict.get("test_acc", [0]*args.rounds)[r-1]) for r in range(1, args.rounds+1)]},
        losses_distributed=[(r, hist_dict.get("test_loss", [0]*args.rounds)[r-1]) for r in range(1, args.rounds+1)]
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
    training_log_path = os.path.join(args.outdir, "enhanced_trust_log.csv")
    if os.path.exists(training_log_path):
        try:
            plot_training_curves(
                log_csv=training_log_path,
                outdir=args.outdir,
            )
        except Exception as e:
            print(f"[WARNING] Could not generate training curves: {e}")
    else:
        print(f"[WARNING] Training log not found at {training_log_path}, skipping training curves")

    print("[INFO] Generating evaluation plots (ROC, Confusion Matrix)...")
    plot_model_eval(
        model_path=os.path.join(args.outdir, "global_model.pt"),
        X_test=X_test,
        y_test=y_test,
        outdir=args.outdir,
    )
    
    # ------------------------------
    # ------------------------------
    # Dashboard info
    # ------------------------------
    if dashboard:
        print("[INFO] Training complete!")
        print(f"[INFO] To view interactive dashboard, run in another terminal:")
        print(f"      python interactive_dashboard.py")
        print(f"      Then open: http://localhost:8050")

    print(f"[DONE] Results stored in {args.outdir}")


if __name__ == "__main__":
    main()