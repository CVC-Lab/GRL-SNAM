#!/usr/bin/env python3
"""
Qualitative Sensing Comparison: CBF Dense vs GRL-SNAM Sparse
Creates figure showing sensing pattern differences between methods
"""

import os, json, argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from collections import defaultdict

# ==== project imports ====
from train_coef_energy import CoefEnergyNet
import scripts.ring_dataset_maxmin as gen
import scripts.spline_stagewise6 as ssi

def mkdir(p): os.makedirs(p, exist_ok=True); return p

# ========= Enhanced Sensing Tracker =========
class SensingPatternTracker:
    """
    Tracks sensing patterns for visualization
    Records: locations, radii, timing, detail level
    """
    
    def __init__(self, world_bounds, method_name):
        self.method_name = method_name
        self.world_bounds = world_bounds
        
        # Sensing pattern data
        self.sensing_events = []  # List of {center, radius, timestep, detail_level, cost}
        self.sensing_grid = {}    # Dict mapping (i,j) -> sensing_count
        self.path_trajectory = [] # Robot trajectory
        self.total_sensing_cost = 0.0
        
        # Grid for visualization
        self.resolution = 0.1
        self.grid_shape = self._compute_grid_shape()
        
    def _compute_grid_shape(self):
        xmin, xmax, ymin, ymax = self.world_bounds
        W = int(np.ceil((xmax - xmin) / self.resolution)) + 1
        H = int(np.ceil((ymax - ymin) / self.resolution)) + 1
        return (H, W)
    
    def world_to_grid(self, x, y):
        xmin, xmax, ymin, ymax = self.world_bounds
        i = int(np.clip((y - ymin) / self.resolution, 0, self.grid_shape[0] - 1))
        j = int(np.clip((x - xmin) / self.resolution, 0, self.grid_shape[1] - 1))
        return i, j
    
    def record_sensing_event(self, center_x, center_y, radius, timestep, detail_level='standard', cost=1.0):
        """Record a sensing event"""
        self.sensing_events.append({
            'center': (center_x, center_y),
            'radius': radius,
            'timestep': timestep,
            'detail_level': detail_level,
            'cost': cost
        })
        
        # Update grid counts
        center_i, center_j = self.world_to_grid(center_x, center_y)
        grid_radius = int(np.ceil(radius / self.resolution))
        
        for di in range(-grid_radius, grid_radius + 1):
            for dj in range(-grid_radius, grid_radius + 1):
                i, j = center_i + di, center_j + dj
                if (0 <= i < self.grid_shape[0] and 0 <= j < self.grid_shape[1] and
                    np.sqrt(di*di + dj*dj) * self.resolution <= radius):
                    key = (i, j)
                    self.sensing_grid[key] = self.sensing_grid.get(key, 0) + (2 if detail_level == 'detailed' else 1)
        
        self.total_sensing_cost += cost
    
    def record_position(self, x, y):
        """Record robot position"""
        self.path_trajectory.append((x, y))
    
    def get_sensing_density_map(self):
        """Return 2D array of sensing density"""
        density_map = np.zeros(self.grid_shape, dtype=float)
        for (i, j), count in self.sensing_grid.items():
            density_map[i, j] = count
        return density_map
    
    def get_coverage_statistics(self):
        """Return sensing coverage statistics"""
        total_cells = self.grid_shape[0] * self.grid_shape[1]
        sensed_cells = len(self.sensing_grid)
        coverage_fraction = sensed_cells / total_cells
        
        if sensed_cells > 0:
            avg_sensing_intensity = np.mean(list(self.sensing_grid.values()))
            max_sensing_intensity = max(self.sensing_grid.values())
        else:
            avg_sensing_intensity = 0.0
            max_sensing_intensity = 0.0
        
        return {
            'coverage_fraction': coverage_fraction,
            'total_sensing_events': len(self.sensing_events),
            'avg_sensing_intensity': avg_sensing_intensity,
            'max_sensing_intensity': max_sensing_intensity,
            'total_cost': self.total_sensing_cost
        }

# ========= CBF with Dense Sensing =========
def cbf_with_dense_sensing(start_xy, goal_xy, world_C, world_R, tracker, 
                          dt=0.02, steps=1500, sensing_radius=0.8):
    """CBF with dense, frequent sensing (simulates constraint updates)"""
    x = np.array(start_xy, dtype=np.float32)
    path = [x.copy()]
    tracker.record_position(x[0], x[1])
    
    def barrier_function(pos, obstacles_C, obstacles_R):
        if len(obstacles_C) == 0: return np.inf
        distances = np.linalg.norm(obstacles_C - pos, axis=1)
        return np.min(distances - obstacles_R - 0.15)
    
    def safe_control(pos, u_nom, obstacles_C, obstacles_R):
        h = barrier_function(pos, obstacles_C, obstacles_R)
        if h > 0: return u_nom
        
        # Simple projection for safety
        if len(obstacles_C) == 0: return u_nom
        distances = np.linalg.norm(obstacles_C - pos, axis=1)
        min_idx = np.argmin(distances - obstacles_R)
        grad_h = (pos - obstacles_C[min_idx]) / distances[min_idx]
        
        # Project away from obstacle
        projection = np.dot(u_nom, grad_h)
        if projection < 0:
            u_safe = u_nom - projection * grad_h
        else:
            u_safe = u_nom
        
        return np.clip(u_safe, -2.0, 2.0)
    
    for t in range(steps):
        # Dense sensing: sense frequently with overlapping regions
        if t % 3 == 0:  # Sense every 3 steps (high frequency)
            # Multiple overlapping sensing regions
            for offset_angle in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
                offset_x = 0.4 * np.cos(offset_angle)
                offset_y = 0.4 * np.sin(offset_angle)
                sense_center_x = x[0] + offset_x
                sense_center_y = x[1] + offset_y
                
                tracker.record_sensing_event(
                    sense_center_x, sense_center_y, sensing_radius, t, 'standard', 1.0
                )
        
        # Extract obstacles in sensing range for control
        C_local, R_local = [], []
        for (cx, cy), r in zip(world_C, world_R):
            if np.linalg.norm([cx - x[0], cy - x[1]]) <= sensing_radius * 2.0:
                C_local.append([cx, cy])
                R_local.append(r)
        C_local = np.array(C_local) if C_local else np.zeros((0, 2))
        R_local = np.array(R_local) if R_local else np.zeros(0)
        
        # CBF control
        to_goal = goal_xy - x
        u_nom = 1.8 * to_goal / (np.linalg.norm(to_goal) + 1e-6)
        u_safe = safe_control(x, u_nom, C_local, R_local)
        
        # Update state
        x = x + dt * u_safe
        path.append(x.copy())
        tracker.record_position(x[0], x[1])
        
        # Success check
        if np.linalg.norm(x - goal_xy) <= 0.25:
            break
    
    success = np.linalg.norm(path[-1] - goal_xy) <= 0.25 if path else False
    return np.array(path, dtype=np.float32), success

# ========= GRL-SNAM with Sparse Adaptive Sensing =========  
def grl_snam_with_sparse_sensing(enhanced_model, cfg_local, world_C, world_R, tracker,
                                steps=1500, device="cpu", base_sensing_radius=1.2):
    """GRL-SNAM with sparse, adaptive sensing"""
    
    # Create planner
    planner = gen.planner_from_cfg(cfg_local, 
                                  ssi.WorldObstacles(np.zeros((0,2)), np.zeros(0), np.zeros(0)), 
                                  cfg_local.k_bulk, cfg_local.gamma_s, cfg_local.d_hat, cfg_local.radius)
    
    path = []
    last_sensing_pos = None
    sensing_cooldown = 0
    
    for t in range(steps):
        sys = planner.sys
        o_w = sys.o.detach().cpu().numpy()
        tracker.record_position(o_w[0], o_w[1])
        
        # Sparse sensing: only sense when needed
        should_sense = False
        
        # Sense if: moved significantly, first step, or exploration bonus high
        if (last_sensing_pos is None or 
            np.linalg.norm(o_w - last_sensing_pos) > 0.6 or  # Moved far enough
            sensing_cooldown <= 0):  # Cooldown expired
            should_sense = True
        
        if should_sense:
            # Minimal initial sensing for prediction
            C_basic, R_basic = [], []
            for (cx, cy), r in zip(world_C, world_R):
                if np.linalg.norm([cx - o_w[0], cy - o_w[1]]) <= base_sensing_radius * 0.7:
                    C_basic.append([cx, cy])
                    R_basic.append(r)
            C_basic = np.array(C_basic) if C_basic else np.zeros((0, 2))
            R_basic = np.array(R_basic) if R_basic else np.zeros(0)
            W_basic = np.ones(len(C_basic)) if len(C_basic) > 0 else np.zeros(0)
            
            # Get neural network predictions
            obs_feats, goal_feats = build_local_feats(o_w, cfg_local.goal, C_basic, R_basic, W_basic)
            obs_mask = (torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device) 
                       if obs_feats.shape[1] else torch.zeros(1,0, dtype=torch.bool, device=device))
            
            with torch.no_grad():
                alphas, beta, gamma, radius_scale, selectivity_thresh, exploration_bonus = enhanced_model(
                    obs_feats.to(device), obs_mask, goal_feats.to(device)
                )
            
            # Extract sensing parameters
            radius_scale_f = float(radius_scale.item())
            selectivity_f = float(selectivity_thresh.item())
            exploration_f = float(exploration_bonus.item())
            al_np = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(R_basic)
            
            # Adaptive sensing decision
            actual_radius = base_sensing_radius * radius_scale_f
            
            # Check for high-importance obstacles
            high_importance_nearby = False
            if len(al_np) > 0:
                high_importance_nearby = np.any(al_np > selectivity_f)
            
            # Record sensing event based on decision
            if high_importance_nearby:
                # Detailed sensing for important regions
                tracker.record_sensing_event(
                    o_w[0], o_w[1], actual_radius, t, 'detailed', actual_radius**2 * 2.5
                )
                sensing_cooldown = 8  # Longer cooldown after detailed sensing
            elif exploration_f > 0.6:
                # Standard exploration sensing
                tracker.record_sensing_event(
                    o_w[0], o_w[1], actual_radius, t, 'standard', actual_radius**2 * 1.0
                )
                sensing_cooldown = 5
            else:
                # Minimal sensing
                tracker.record_sensing_event(
                    o_w[0], o_w[1], actual_radius * 0.6, t, 'minimal', actual_radius**2 * 0.5
                )
                sensing_cooldown = 3
            
            last_sensing_pos = o_w.copy()
        
        sensing_cooldown = max(0, sensing_cooldown - 1)
        
        # Get local obstacles for navigation
        C_local, R_local = [], []
        for (cx, cy), r in zip(world_C, world_R):
            if np.linalg.norm([cx - o_w[0], cy - o_w[1]]) <= base_sensing_radius:
                C_local.append([cx, cy])
                R_local.append(r)
        
        C_local = np.array(C_local) if C_local else np.zeros((0, 2))
        R_local = np.array(R_local) if R_local else np.zeros(0)
        W_local = np.ones(len(C_local)) if len(C_local) > 0 else np.zeros(0)
        
        # Apply alpha weights if available
        if len(al_np) > 0 and len(W_local) > 0:
            min_len = min(len(al_np), len(W_local))
            W_local[:min_len] *= np.maximum(al_np[:min_len], 0.2)
        
        # Step simulation
        world_step = ssi.WorldObstacles(C_local, R_local, W_local, d_hat=cfg_local.d_hat)
        if should_sense:  # Update goal weight when we sense
            planner.stage_field.w_goal = max(0.5, float(beta.squeeze(0).item()) if 'beta' in locals() else 1.0)
            
        info = planner.step(cfg_local.dt, world_step)
        center_xy = np.asarray(info["center"], float)
        path.append(center_xy)
        
        # Success check
        if success_reached(center_xy, cfg_local.goal, cfg_local.goal_tol):
            break
    
    success = success_reached(path[-1], cfg_local.goal, cfg_local.goal_tol) if path else False
    return np.array(path, dtype=np.float32), success

# ========= Utility Functions =========
def build_local_feats(o_w, goal_w, C_w, R_w, W_w):
    o = torch.as_tensor(o_w, dtype=torch.float32)
    g = torch.as_tensor(goal_w, dtype=torch.float32)
    C = torch.as_tensor(C_w, dtype=torch.float32) if C_w.size else torch.zeros(0,2, dtype=torch.float32)
    R = torch.as_tensor(R_w, dtype=torch.float32) if R_w.size else torch.zeros(0, dtype=torch.float32)
    W = torch.as_tensor(W_w, dtype=torch.float32) if W_w.size else torch.zeros(0, dtype=torch.float32)
    if C.ndim == 1: C = C.reshape(0,2)
    dg = (g - o); gdist = torch.linalg.norm(dg).unsqueeze(0)
    goal_feats = torch.stack([dg[0], dg[1], gdist[0], torch.tensor(1.0)], dim=0).unsqueeze(0)
    if C.shape[0] == 0: obs_feats = torch.zeros(1,0,6, dtype=torch.float32)
    else:
        dxdy = (g.unsqueeze(0) - C)
        obs_feats = torch.cat([C, R.unsqueeze(-1), W.unsqueeze(-1), dxdy], dim=-1).unsqueeze(0)
    return obs_feats, goal_feats

def success_reached(last_center, goal_xy, tol):
    return np.linalg.norm(np.asarray(last_center) - np.asarray(goal_xy)) <= tol

def compute_workspace_bounds(C, R, margin=1.0):
    if len(C) == 0: return -5.0, 5.0, -5.0, 5.0
    cx, cy = C[:,0], C[:,1]
    return (float(np.min(cx - R) - margin), float(np.max(cx + R) + margin),
            float(np.min(cy - R) - margin), float(np.max(cy + R) + margin))

# ========= Enhanced Model Wrapper =========
class SimpleEnhancedModel:
    """Simple wrapper that adds sensing predictions to base model"""
    def __init__(self, base_model):
        self.base_model = base_model
    
    def __call__(self, obs_feats, obs_mask, goal_feats):
        # Get base predictions
        alphas, beta, gamma = self.base_model(obs_feats, obs_mask, goal_feats)
        
        # Simple sensing parameter predictions based on obstacles and goal distance
        with torch.no_grad():
            if obs_feats.shape[1] > 0:
                # More obstacles → larger sensing radius
                radius_scale = torch.tensor(1.0 + 0.3 * obs_feats.shape[1] / 10.0)
                # Higher obstacle density → lower selectivity (sense more)
                selectivity = torch.tensor(0.3 + 0.4 * torch.sigmoid(-obs_feats.shape[1] / 5.0))
                # Goal distance → exploration bonus
                goal_dist = torch.linalg.norm(goal_feats[0, :2])
                exploration = torch.tensor(0.2 + 0.6 * torch.sigmoid(goal_dist / 3.0))
            else:
                radius_scale = torch.tensor(0.8)
                selectivity = torch.tensor(0.7)
                exploration = torch.tensor(0.8)
        
        return alphas, beta, gamma, radius_scale, selectivity, exploration

# ========= Visualization Functions =========
def create_qualitative_comparison_figure(cbf_tracker, grl_tracker, world_C, world_R, 
                                        start_xy, goal_xy, save_path):
    """Create the qualitative comparison figure"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Common setup
    for ax in [ax1, ax2]:
        # Draw obstacles
        circles = []
        for (cx, cy), r in zip(world_C, world_R):
            circle = plt.Circle((cx, cy), r, fill=True, facecolor='darkgray', 
                              edgecolor='black', alpha=0.7, linewidth=1.5)
            ax.add_patch(circle)
        
        # Draw start and goal
        ax.plot(start_xy[0], start_xy[1], 'gs', markersize=12, label='Start', markeredgecolor='black')
        ax.plot(goal_xy[0], goal_xy[1], 'r*', markersize=15, label='Goal', markeredgecolor='black')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 10)
        ax.set_ylim(-3, 3)
    
    # Left plot: CBF Dense Sensing
    ax1.set_title('CBF: Dense Constraint Updates\n(High-frequency sensing patches)', fontsize=14, fontweight='bold')
    
    # Draw sensing patches for CBF (dense, overlapping)
    cbf_density = cbf_tracker.get_sensing_density_map()
    xmin, xmax, ymin, ymax = cbf_tracker.world_bounds
    extent = [xmin, xmax, ymin, ymax]
    
    # Create custom colormap for dense sensing
    im1 = ax1.imshow(cbf_density, extent=extent, origin='lower', alpha=0.4, 
                     cmap='Reds', vmin=0, vmax=np.max(cbf_density))
    
    # Highlight high-density sensing regions with explicit patches
    high_density_threshold = np.max(cbf_density) * 0.7
    for event in cbf_tracker.sensing_events[::2]:  # Show every other event to avoid clutter
        center_x, center_y = event['center']
        radius = event['radius']
        alpha_val = 0.15 if event['detail_level'] == 'standard' else 0.25
        
        # Add sensing circles
        sensing_circle = plt.Circle((center_x, center_y), radius, fill=False, 
                                  edgecolor='red', alpha=alpha_val, linewidth=1)
        ax1.add_patch(sensing_circle)
    
    # Draw CBF trajectory
    if cbf_tracker.path_trajectory:
        traj = np.array(cbf_tracker.path_trajectory)
        ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, alpha=0.8, label='CBF Trajectory')
    
    # Right plot: GRL-SNAM Sparse Sensing
    ax2.set_title('GRL-SNAM: Sparse Overlapping Patches\n(Adaptive sensing + online refinement)', fontsize=14, fontweight='bold')
    
    # Draw sensing patches for GRL-SNAM (sparse, selective)
    grl_density = grl_tracker.get_sensing_density_map()
    im2 = ax2.imshow(grl_density, extent=extent, origin='lower', alpha=0.4, 
                     cmap='Blues', vmin=0, vmax=max(np.max(grl_density), 1))
    
    # Draw selective sensing events with different styles based on detail level
    for event in grl_tracker.sensing_events:
        center_x, center_y = event['center']
        radius = event['radius']
        detail = event['detail_level']
        
        if detail == 'detailed':
            # High-detail sensing: solid circle
            sensing_circle = plt.Circle((center_x, center_y), radius, fill=False, 
                                      edgecolor='darkblue', alpha=0.6, linewidth=2.5)
            ax2.add_patch(sensing_circle)
        elif detail == 'standard':
            # Standard sensing: dashed circle
            sensing_circle = plt.Circle((center_x, center_y), radius, fill=False, 
                                      edgecolor='blue', alpha=0.4, linewidth=1.5, linestyle='--')
            ax2.add_patch(sensing_circle)
        else:
            # Minimal sensing: dotted circle
            sensing_circle = plt.Circle((center_x, center_y), radius, fill=False, 
                                      edgecolor='lightblue', alpha=0.3, linewidth=1, linestyle=':')
            ax2.add_patch(sensing_circle)
    
    # Draw GRL-SNAM trajectory
    if grl_tracker.path_trajectory:
        traj = np.array(grl_tracker.path_trajectory)
        ax2.plot(traj[:, 0], traj[:, 1], 'orange', linewidth=3, alpha=0.9, label='GRL-SNAM Trajectory')
    
    # Add legends and statistics
    cbf_stats = cbf_tracker.get_coverage_statistics()
    grl_stats = grl_tracker.get_coverage_statistics()
    
    # Add text boxes with statistics
    cbf_text = (f"Coverage: {cbf_stats['coverage_fraction']:.1%}\n"
                f"Events: {cbf_stats['total_sensing_events']}\n" 
                f"Avg Intensity: {cbf_stats['avg_sensing_intensity']:.1f}\n"
                f"Cost: {cbf_stats['total_cost']:.1f}")
    
    grl_text = (f"Coverage: {grl_stats['coverage_fraction']:.1%}\n"
                f"Events: {grl_stats['total_sensing_events']}\n"
                f"Avg Intensity: {grl_stats['avg_sensing_intensity']:.1f}\n"
                f"Cost: {grl_stats['total_cost']:.1f}")
    
    ax1.text(0.02, 0.98, cbf_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.text(0.02, 0.98, grl_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legends
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    
    # Add custom legend for sensing patterns
    from matplotlib.lines import Line2D
    
    # CBF sensing legend
    cbf_legend_elements = [
        Line2D([0], [0], color='red', alpha=0.5, linewidth=2, label='Dense sensing patches')
    ]
    ax1.legend(handles=cbf_legend_elements + ax1.get_legend().legend_handles, loc='lower right')
    
    # GRL-SNAM sensing legend  
    grl_legend_elements = [
        Line2D([0], [0], color='darkblue', linewidth=2.5, label='Detailed sensing'),
        Line2D([0], [0], color='blue', linewidth=1.5, linestyle='--', label='Standard sensing'),
        Line2D([0], [0], color='lightblue', linewidth=1, linestyle=':', label='Minimal sensing')
    ]
    ax2.legend(handles=grl_legend_elements + ax2.get_legend().legend_handles, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'cbf_stats': cbf_stats,
        'grl_stats': grl_stats,
        'sensing_efficiency_ratio': grl_stats['total_cost'] / max(cbf_stats['total_cost'], 1.0)
    }

# ========= Main Script =========
def main():
    parser = argparse.ArgumentParser("Qualitative Sensing Comparison Visualization")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--case", type=str, default="case1-tight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="qualitative_sensing_figures")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    base_model = CoefEnergyNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    base_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    base_model.eval()
    
    # Create enhanced model wrapper
    enhanced_model = SimpleEnhancedModel(base_model)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = mkdir(f"{args.output_dir}/{timestamp}")
    
    # Set seed and generate environment
    np.random.seed(args.seed)
    cfg = gen.GenCfg()
    cfg.seed = args.seed
    gen.set_all_seeds(cfg.seed)
    
    if args.case.startswith("case1"):
        C, R, W = gen.sample_obstacles_case1_tight(cfg)
    else:
        C, R, W = gen.sample_obstacles_case2_harder(cfg)
    
    # Define challenging start/goal
    world_bounds = compute_workspace_bounds(C, R, margin=2.0)
    start_xy = np.array([-1.5, -0.3])
    goal_xy = np.array([8.5, 0.5])
    
    print(f"Running qualitative comparison...")
    print(f"Environment: {len(C)} obstacles")
    print(f"Start: {start_xy}, Goal: {goal_xy}")
    
    # Create trackers
    cbf_tracker = SensingPatternTracker(world_bounds, "CBF")
    grl_tracker = SensingPatternTracker(world_bounds, "GRL-SNAM")
    
    # Run CBF with dense sensing
    print("Running CBF with dense sensing...")
    cbf_path, cbf_success = cbf_with_dense_sensing(
        start_xy, goal_xy, C, R, cbf_tracker
    )
    print(f"CBF: Success={cbf_success}, Path length={len(cbf_path)}")
    
    # Run GRL-SNAM with sparse adaptive sensing
    print("Running GRL-SNAM with sparse adaptive sensing...")
    
    # Configure GRL-SNAM
    cfg_local = gen.GenCfg()
    cfg_local.d_hat = 0.5
    cfg_local.radius = 0.4
    cfg_local.k_bulk = 15.0
    cfg_local.gamma_s = 0.08
    cfg_local.dt = 0.025
    cfg_local.start = start_xy.astype(np.float32)
    cfg_local.goal = goal_xy.astype(np.float32)
    cfg_local.goal_tol = 0.3
    
    grl_path, grl_success = grl_snam_with_sparse_sensing(
        enhanced_model, cfg_local, C, R, grl_tracker, device=device
    )
    print(f"GRL-SNAM: Success={grl_success}, Path length={len(grl_path)}")
    
    # Create visualization
    figure_path = os.path.join(output_dir, "qualitative_sensing_comparison.png")
    comparison_stats = create_qualitative_comparison_figure(
        cbf_tracker, grl_tracker, C, R, start_xy, goal_xy, figure_path
    )
    
    # Save detailed results
    results = {
        'environment': {
            'n_obstacles': len(C),
            'world_bounds': world_bounds,
            'start': start_xy.tolist(),
            'goal': goal_xy.tolist()
        },
        'cbf_results': {
            'success': cbf_success,
            'path_length': len(cbf_path),
            'sensing_stats': comparison_stats['cbf_stats']
        },
        'grl_snam_results': {
            'success': grl_success, 
            'path_length': len(grl_path),
            'sensing_stats': comparison_stats['grl_stats']
        },
        'comparison': {
            'sensing_efficiency_ratio': comparison_stats['sensing_efficiency_ratio'],
            'coverage_reduction': 1.0 - (comparison_stats['grl_stats']['coverage_fraction'] / 
                                        max(comparison_stats['cbf_stats']['coverage_fraction'], 0.001)),
            'sensing_events_ratio': comparison_stats['grl_stats']['total_sensing_events'] / 
                                  max(comparison_stats['cbf_stats']['total_sensing_events'], 1)
        }
    }
    
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n=== Qualitative Comparison Results ===")
    print(f"Figure saved: {figure_path}")
    print(f"CBF - Coverage: {comparison_stats['cbf_stats']['coverage_fraction']:.1%}, "
          f"Events: {comparison_stats['cbf_stats']['total_sensing_events']}, "
          f"Cost: {comparison_stats['cbf_stats']['total_cost']:.1f}")
    print(f"GRL-SNAM - Coverage: {comparison_stats['grl_stats']['coverage_fraction']:.1%}, "
          f"Events: {comparison_stats['grl_stats']['total_sensing_events']}, "
          f"Cost: {comparison_stats['grl_stats']['total_cost']:.1f}")
    print(f"GRL-SNAM sensing efficiency: {comparison_stats['sensing_efficiency_ratio']:.2f}x of CBF cost")
    
    coverage_reduction = results['comparison']['coverage_reduction']
    events_ratio = results['comparison']['sensing_events_ratio']
    print(f"Coverage reduction: {coverage_reduction:.1%}")
    print(f"Sensing events ratio: {events_ratio:.2f}")

if __name__ == "__main__":
    main()