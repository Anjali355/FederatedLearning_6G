"""
Interactive Web Dashboard for Federated Learning + 6G Network
Run this alongside your training for real-time interactive visualization
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
import time


def create_network_status_for_dashboard(network_6g):
    """
    Helper function to extract network status for dashboard visualization
    
    Args:
        network_6g: Network6GSimulator instance
        
    Returns:
        dict with network status information
    """
    if network_6g is None:
        return {
            'base_stations': [],
            'devices': {},
            'network_slices': {}
        }
    
    # Extract base station info
    base_stations = []
    for bs in network_6g.base_stations:
        base_stations.append({
            'id': bs.bs_id,
            'location': bs.location,
            'connected_devices': len(bs.connected_devices),
            'capacity': bs.capacity
        })
    
    # Extract device info
    devices = {}
    for device_id, device in network_6g.devices.items():
        devices[device_id] = {
            'status': device.status.value if hasattr(device.status, 'value') else str(device.status),
            'location': device.location,
            'trust_score': getattr(device, 'trust_score', 0.5),
            'base_station': device.base_station.bs_id if device.base_station else None
        }
    
    # Extract network slice info
    network_slices = {}
    if hasattr(network_6g, 'network_slices'):
        for slice_type, slice_obj in network_6g.network_slices.items():
            network_slices[slice_type] = {
                'allocated_devices': len(getattr(slice_obj, 'allocated_devices', [])),
                'requirements': getattr(slice_obj, 'requirements', {})
            }
    
    return {
        'base_stations': base_stations,
        'devices': devices,
        'network_slices': network_slices
    }


class InteractiveDashboard:
    def __init__(self, outdir="artifacts_flower", port=8050):
        self.outdir = outdir
        self.port = port
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Create the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚀 Federated Learning + 6G Network - Interactive Dashboard",
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
                html.H4("Real-time Monitoring & Analysis",
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': 0}),
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            ),
            
            # Status indicator
            html.Div(id='status-indicator', style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Main content - 2 columns
            html.Div([
                # Left column - Training metrics
                html.Div([
                    # Training progress
                    dcc.Graph(id='training-progress', style={'height': '400px'}),
                    
                    # Client accuracy comparison
                    dcc.Graph(id='client-accuracy', style={'height': '400px'}),
                    
                    # Trust score heatmap
                    dcc.Graph(id='trust-heatmap', style={'height': '500px'}),
                    
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                
                # Right column - Network & Detection
                html.Div([
                    # 6G Network topology
                    dcc.Graph(id='network-topology', style={'height': '500px'}),
                    
                    # Attack detection timeline
                    dcc.Graph(id='detection-timeline', style={'height': '400px'}),
                    
                    # Client contributions
                    dcc.Graph(id='client-contributions', style={'height': '400px'}),
                    
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            ]),
            
            # Footer with statistics
            html.Div(id='statistics-footer', 
                    style={'backgroundColor': '#34495e', 'color': 'white', 
                          'padding': '20px', 'marginTop': '20px', 'textAlign': 'center'})
        ])
    
    def setup_callbacks(self):
        """Setup callback functions for interactive updates"""
        
        @self.app.callback(
            [Output('status-indicator', 'children'),
             Output('training-progress', 'figure'),
             Output('client-accuracy', 'figure'),
             Output('trust-heatmap', 'figure'),
             Output('network-topology', 'figure'),
             Output('detection-timeline', 'figure'),
             Output('client-contributions', 'figure'),
             Output('statistics-footer', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components"""
            
            # Check if data exists
            trust_log_path = os.path.join(self.outdir, "enhanced_trust_log.csv")
            network_path = os.path.join(self.outdir, "6g_network_results.json")
            
            if not os.path.exists(trust_log_path):
                # Return empty/waiting state
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="Waiting for training data...", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False,
                                        font=dict(size=20, color="gray"))
                empty_fig.update_layout(template="plotly_white")
                
                status = html.Div([
                    html.Span("⏳ ", style={'fontSize': '30px'}),
                    html.Span("Waiting for training to start...", 
                             style={'fontSize': '18px', 'color': '#e67e22'})
                ])
                
                return (status, empty_fig, empty_fig, empty_fig, 
                       empty_fig, empty_fig, empty_fig, 
                       "No data available yet. Start training with --dashboard flag.")
            
            # Load data
            try:
                trust_df = pd.read_csv(trust_log_path)
                
                with open(network_path, 'r') as f:
                    network_data = json.load(f)
                
                with open(os.path.join(self.outdir, "client_flags.json"), 'r') as f:
                    malicious_flags = json.load(f)
                    malicious_clients = {int(k) for k, v in malicious_flags.items() if v}
                
            except Exception as e:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Error loading data: {str(e)}", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
                return (html.Div(f"Error: {e}"), empty_fig, empty_fig, empty_fig, 
                       empty_fig, empty_fig, empty_fig, str(e))
            
            # Current round
            current_round = trust_df['round'].max()
            
            # Status indicator
            status = html.Div([
                html.Span("✅ ", style={'fontSize': '30px'}),
                html.Span(f"Live - Round {current_round}", 
                         style={'fontSize': '18px', 'color': '#27ae60', 'fontWeight': 'bold'}),
                html.Span(f" | Last update: {time.strftime('%H:%M:%S')}", 
                         style={'fontSize': '14px', 'color': '#95a5a6', 'marginLeft': '20px'})
            ])
            
            # Create figures
            fig1 = self.create_training_progress(trust_df)
            fig2 = self.create_client_accuracy(trust_df, malicious_clients)
            fig3 = self.create_trust_heatmap(trust_df, malicious_clients)
            fig4 = self.create_network_topology(network_data, malicious_clients)
            fig5 = self.create_detection_timeline(trust_df, malicious_clients)
            fig6 = self.create_client_contributions(trust_df, malicious_clients)
            
            # Statistics footer
            stats = self.create_statistics_footer(trust_df, network_data, malicious_clients)
            
            return status, fig1, fig2, fig3, fig4, fig5, fig6, stats
    
    def create_training_progress(self, df):
        """Create training progress plot"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Accuracy
        fig.add_trace(
            go.Scatter(x=df['round'], y=df['test_acc'], 
                      name='Test Accuracy', mode='lines+markers',
                      line=dict(color='#3498db', width=3),
                      marker=dict(size=8),
                      hovertemplate='Round: %{x}<br>Accuracy: %{y:.4f}<extra></extra>'),
            secondary_y=False
        )
        
        # Loss
        fig.add_trace(
            go.Scatter(x=df['round'], y=df['test_loss'],
                      name='Test Loss', mode='lines+markers',
                      line=dict(color='#e74c3c', width=3),
                      marker=dict(size=8),
                      hovertemplate='Round: %{x}<br>Loss: %{y:.4f}<extra></extra>'),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Round")
        fig.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False, range=[0, 1.05])
        fig.update_yaxes(title_text="<b>Loss</b>", secondary_y=True)
        
        fig.update_layout(
            title="📈 Global Model Performance",
            hovermode='x unified',
            template='plotly_white',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        
        return fig
    
    def create_client_accuracy(self, df, malicious_clients):
        """Create client accuracy comparison"""
        fig = go.Figure()
        
        # Get unique clients
        clients = sorted(df['client_id'].unique())
        
        # Use test_acc since train_acc is not available per client
        acc_col = 'test_acc' if 'test_acc' in df.columns else 'train_acc'
        
        for client_id in clients:
            client_data = df[df['client_id'] == client_id]
            is_malicious = client_id in malicious_clients
            
            fig.add_trace(go.Scatter(
                x=client_data['round'],
                y=client_data[acc_col],
                name=f'Client {client_id}' + (' 🔴' if is_malicious else ''),
                mode='lines+markers',
                line=dict(
                    color='red' if is_malicious else None,
                    width=3 if is_malicious else 1.5
                ),
                marker=dict(size=8 if is_malicious else 5,
                           symbol='x' if is_malicious else 'circle'),
                hovertemplate=f'Client {client_id}<br>Round: %{{x}}<br>Accuracy: %{{y:.4f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title="👥 Individual Client Performance (Test Accuracy)",
            xaxis_title="Round",
            yaxis_title="Test Accuracy",
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(range=[0, 1.05]),
            legend=dict(x=1.05, y=1, bgcolor='rgba(255,255,255,0.8)')
        )
        
        return fig
    
    def create_trust_heatmap(self, df, malicious_clients):
        """Create interactive trust score heatmap"""
        # Pivot data
        pivot = df.pivot(index='client_id', columns='round', values='trust_score')
        
        # Create annotations for malicious clients
        malicious_labels = [f"Client {idx} {'🔴 MALICIOUS' if idx in malicious_clients else ''}" 
                           for idx in pivot.index]
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=malicious_labels,
            colorscale='RdYlGn',
            zmid=0.7,
            zmin=0,
            zmax=1,
            text=np.round(pivot.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='Client: %{y}<br>Round: %{x}<br>Trust: %{z:.4f}<extra></extra>',
            colorbar=dict(title="Trust Score")
        ))
        
        fig.update_layout(
            title="🔥 Trust Score Heatmap (Interactive - Hover for details)",
            xaxis_title="Round",
            yaxis_title="Client",
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_network_topology(self, network_data, malicious_clients):
        """Create interactive 6G network topology"""
        fig = go.Figure()
        
        # Extract data
        devices = network_data.get('device_trust_scores', {})
        final_status = network_data.get('final_status', {})
        bs_data = final_status.get('base_stations', [])
        
        # Plot base stations
        bs_x, bs_y, bs_text, bs_size, bs_color = [], [], [], [], []
        for bs in bs_data:
            if 'location' in bs:
                x, y = bs['location']
            else:
                # Generate position from ID
                bs_id_num = int(bs['bs_id'].replace('BS_', ''))
                x, y = (bs_id_num % 4) * 1200, (bs_id_num // 4) * 1200
            
            count = bs.get('connected_count', 0)
            bs_x.append(x)
            bs_y.append(y)
            bs_text.append(f"{bs['bs_id']}<br>{count} devices")
            bs_size.append(15 + count * 3)
            bs_color.append('#27ae60' if count > 0 else '#95a5a6')
        
        if bs_x:
            fig.add_trace(go.Scatter(
                x=bs_x, y=bs_y,
                mode='markers+text',
                marker=dict(size=bs_size, color=bs_color, symbol='triangle-up',
                           line=dict(width=2, color='white')),
                text=[bs['bs_id'] for bs in bs_data],
                textposition='bottom center',
                name='Base Stations',
                hovertext=bs_text,
                hoverinfo='text'
            ))
        
        # Plot devices
        dev_x, dev_y, dev_text, dev_color, dev_symbol = [], [], [], [], []
        for dev_id, trust in devices.items():
            # Generate position (in real app, get from network data)
            dev_num = int(dev_id)
            x = (dev_num * 800 + 400) % 5000
            y = (dev_num * 600 + 300) % 5000
            
            is_malicious = dev_num in malicious_clients
            dev_x.append(x)
            dev_y.append(y)
            dev_text.append(f"Device {dev_id}<br>Trust: {trust:.3f}")
            
            if is_malicious:
                dev_color.append('#e74c3c')
                dev_symbol.append('x')
            elif trust < 0.5:
                dev_color.append('#e67e22')
                dev_symbol.append('square')
            else:
                dev_color.append('#2ecc71')
                dev_symbol.append('circle')
        
        if dev_x:
            fig.add_trace(go.Scatter(
                x=dev_x, y=dev_y,
                mode='markers+text',
                marker=dict(size=12, color=dev_color, symbol=dev_symbol,
                           line=dict(width=2, color='white')),
                text=[f"D{d}" for d in devices.keys()],
                textposition='top center',
                name='Devices',
                hovertext=dev_text,
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title="📡 6G Network Topology",
            xaxis=dict(title="X Position (m)", range=[0, 5000]),
            yaxis=dict(title="Y Position (m)", range=[0, 5000], scaleanchor="x"),
            template='plotly_white',
            showlegend=True,
            hovermode='closest',
            height=500
        )
        
        return fig
    
    def create_detection_timeline(self, df, malicious_clients):
        """Create attack detection timeline"""
        fig = go.Figure()
        
        # Calculate detection scores (sum of anomalies)
        for client_id in sorted(df['client_id'].unique()):
            client_data = df[df['client_id'] == client_id].copy()
            
            # Calculate total detection score from multiple factors
            detection_cols = [col for col in df.columns if 'detection' in col or 'anomaly' in col]
            if detection_cols:
                client_data['detection_total'] = client_data[detection_cols].sum(axis=1)
            else:
                # Use trust score inverse as proxy
                client_data['detection_total'] = 1 - client_data['trust_score']
            
            is_malicious = client_id in malicious_clients
            
            fig.add_trace(go.Scatter(
                x=client_data['round'],
                y=client_data['detection_total'],
                name=f'Client {client_id}' + (' 🔴' if is_malicious else ''),
                mode='lines+markers',
                line=dict(color='red' if is_malicious else None,
                         width=3 if is_malicious else 1),
                marker=dict(size=8 if is_malicious else 4),
                hovertemplate=f'Client {client_id}<br>Round: %{{x}}<br>Detection Score: %{{y:.4f}}<extra></extra>'
            ))
        
        # Add threshold lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                     annotation_text="Suspicious Threshold",
                     annotation_position="right")
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                     annotation_text="Malicious Threshold",
                     annotation_position="right")
        
        fig.update_layout(
            title="⚠️ Attack Detection Scores (Hover for details)",
            xaxis_title="Round",
            yaxis_title="Detection Score",
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_client_contributions(self, df, malicious_clients):
        """Create client contribution stacked area"""
        # Get latest round data for pie chart instead
        latest_round = df['round'].max()
        latest_data = df[df['round'] == latest_round]
        
        # Create pie chart of contributions
        labels = [f"Client {cid}{' 🔴' if cid in malicious_clients else ''}" 
                 for cid in latest_data['client_id']]
        values = latest_data['weight'] if 'weight' in latest_data.columns else [1] * len(latest_data)
        colors = ['#e74c3c' if cid in malicious_clients else None 
                 for cid in latest_data['client_id']]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hovertemplate='%{label}<br>Weight: %{value:.4f}<br>Percentage: %{percent}<extra></extra>',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title=f"📊 Client Aggregation Weights (Round {latest_round})",
            template='plotly_white'
        )
        
        return fig
    
    def create_statistics_footer(self, df, network_data, malicious_clients):
        """Create statistics footer"""
        current_round = df['round'].max()
        latest_acc = df[df['round'] == current_round]['test_acc'].iloc[0] if len(df[df['round'] == current_round]) > 0 else 0
        latest_loss = df[df['round'] == current_round]['test_loss'].iloc[0] if len(df[df['round'] == current_round]) > 0 else 0
        
        num_clients = len(df['client_id'].unique())
        num_malicious = len(malicious_clients)
        
        # Network stats
        final_status = network_data.get('final_status', {})
        active_devices = final_status.get('active_devices', 0)
        suspicious_devices = final_status.get('suspicious_devices', 0)
        blocked_devices = final_status.get('blocked_devices', 0)
        
        return html.Div([
            html.Div([
                html.Strong("Current Round: "),
                html.Span(f"{current_round}", style={'color': '#3498db', 'fontSize': '18px'}),
            ], style={'display': 'inline-block', 'margin': '0 30px'}),
            
            html.Div([
                html.Strong("Test Accuracy: "),
                html.Span(f"{latest_acc:.4f}", style={'color': '#27ae60', 'fontSize': '18px'}),
            ], style={'display': 'inline-block', 'margin': '0 30px'}),
            
            html.Div([
                html.Strong("Test Loss: "),
                html.Span(f"{latest_loss:.4f}", style={'color': '#e67e22', 'fontSize': '18px'}),
            ], style={'display': 'inline-block', 'margin': '0 30px'}),
            
            html.Div([
                html.Strong("Clients: "),
                html.Span(f"{num_clients} total, {num_malicious} malicious", 
                         style={'fontSize': '16px'}),
            ], style={'display': 'inline-block', 'margin': '0 30px'}),
            
            html.Div([
                html.Strong("Network: "),
                html.Span(f"✅ {active_devices} active", style={'color': '#27ae60', 'margin': '0 10px'}),
                html.Span(f"⚠️ {suspicious_devices} suspicious", style={'color': '#e67e22', 'margin': '0 10px'}),
                html.Span(f"🚫 {blocked_devices} blocked", style={'color': '#e74c3c', 'margin': '0 10px'}),
            ], style={'display': 'inline-block', 'margin': '0 30px'}),
        ])
    
    def run(self, debug=False):
        """Start the dashboard server"""
        print(f"\n{'='*70}")
        print(f"🌐 Interactive Dashboard Starting...")
        print(f"{'='*70}")
        print(f"\n📊 Open your browser and navigate to:")
        print(f"\n    👉  http://localhost:{self.port}\n")
        print(f"💡 Dashboard will auto-update every 2 seconds")
        print(f"🔄 Refresh browser manually if needed")
        print(f"⚡ Press Ctrl+C to stop the server\n")
        print(f"{'='*70}\n")
        
        self.app.run(debug=debug, port=self.port, host='0.0.0.0')


if __name__ == "__main__":
    import sys
    
    outdir = sys.argv[1] if len(sys.argv) > 1 else "artifacts_flower"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8050
    
    dashboard = InteractiveDashboard(outdir=outdir, port=port)
    dashboard.run(debug=False)
