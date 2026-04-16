import os
import shutil
import subprocess
import time
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from scipy.ndimage import label as ndlabel
# from scipy.stats import gaussian_kde

# ── User settings ─────────────────────────────────────────────────────────
BASE_INPUT       = "grain_growth_2d.in"
SPPARKS_PATH     = "/home/hice1/vyadav68/scratch/grain_growth/spparks/src/spk_mpi"
N                = 512  # Target grid resolution
TARGET_TIMESTEPS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
RUN_ROOT         = "runs"
OUT_ROOT         = "grain_images"
DIST_ROOT        = "grain_distributions"

TEMP_MIN         = 0.5 
TEMP_MAX         = 1.0

# ── Fixed axis limits for distribution plots ──────────────────────────────
DIST_XLIM = (0.0, 2.5)    
DIST_BINS = 10            

# ── Helper: Boundary Map Generator ────────────────────────────────────────
def get_boundaries(grains_2d):
    boundaries = np.zeros_like(grains_2d, dtype=np.uint8)
    boundaries |= (grains_2d != np.roll(grains_2d, shift=1, axis=0))
    boundaries |= (grains_2d != np.roll(grains_2d, shift=-1, axis=0))
    boundaries |= (grains_2d != np.roll(grains_2d, shift=1, axis=1))
    boundaries |= (grains_2d != np.roll(grains_2d, shift=-1, axis=1))
    return boundaries

# ── 1. Parse dump file ────────────────────────────────────────────────────
def read_dump_all_timesteps(filename, N):
    all_timesteps    = []
    timestep_numbers = []
    current_grains   = []
    current_timestep = None
    reading_data     = False

    with open(filename, "r") as f:
        for line in f:
            if "ITEM: TIMESTEP" in line:
                if current_grains and current_timestep is not None:
                    # 'F' order aligns NumPy with SPPARKS Fortran-style lattice
                    all_timesteps.append(np.array(current_grains).reshape((N, N), order='F'))
                    timestep_numbers.append(current_timestep)
                    current_grains = []
                reading_data     = False
                current_timestep = None

            elif current_timestep is None and line.strip() and not line.startswith("ITEM"):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_timestep = round(float(parts[1]), 2)
                    else:
                        current_timestep = round(float(parts[0]), 2)
                except (ValueError, IndexError):
                    pass

            elif "ITEM: ATOMS" in line:
                reading_data = True

            elif reading_data and line.strip():
                try:
                    parts = line.split()
                    # Grab the 2nd column (grain ID). 1st column is the Site ID.
                    if len(parts) >= 2:
                        current_grains.append(int(parts[1]))
                    else:
                        current_grains.append(int(parts[0]))
                except ValueError:
                    continue

    if current_grains and current_timestep is not None:
        all_timesteps.append(np.array(current_grains).reshape((N, N), order='F'))
        timestep_numbers.append(current_timestep)

    return timestep_numbers, all_timesteps

# ── 2. Grain size distribution ────────────────────────────────────────────
def compute_and_save_distribution(grains_2d, timestep, run_id, temperature):
    unique_ids       = np.unique(grains_2d)
    true_grain_areas = []

    for gid in unique_ids:
        mask                   = (grains_2d == gid)
        labeled_array, n_blobs = ndlabel(mask)
        for blob_idx in range(1, n_blobs + 1):
            area = int(np.sum(labeled_array == blob_idx))
            true_grain_areas.append(area)

    atom_counts = np.array(true_grain_areas, dtype=float)
    radii       = np.sqrt(atom_counts / np.pi)
    mean_R      = radii.mean()
    normalized  = radii / mean_R 

    # ── CSV ──
    csv_dir = os.path.join(DIST_ROOT, f"run_{run_id}")
    os.makedirs(csv_dir, exist_ok=True)
    with open(os.path.join(csv_dir, f"dist_t{timestep}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grain_index", "atom_count", "equiv_radius", "norm_radius"])
        for i, (ac, r, nr) in enumerate(zip(atom_counts, radii, normalized)):
            writer.writerow([i, int(ac), round(float(r), 6), round(float(nr), 6)])

    # ── JSON ──
    num_true_grains = int(len(true_grain_areas))
    record = {
        "run_id":           run_id,
        "temperature":      round(float(temperature), 3),
        "timestep":         timestep,
        "num_spin_ids":     int(len(unique_ids)),
        "num_true_grains":  num_true_grains,
        "mean_radius":      round(float(mean_R), 6),
        "equiv_radii":      sorted(np.round(radii,     6).tolist()),
        "normalized_radii": sorted(np.round(normalized, 6).tolist()),
    }
    with open(os.path.join(csv_dir, f"dist_t{timestep}.json"), "w") as f:
        json.dump(record, f, indent=2)

    print(f"    Target t={timestep}: {len(unique_ids)} spin IDs -> "
          f"{num_true_grains} true grains | mean R={mean_R:.2f}")

    # ── 6. Fixed Plot ──
    # normalized_radii = normalized 
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.hist(normalized_radii, bins=DIST_BINS, range=DIST_XLIM, density=True, 
    #         color="skyblue", edgecolor="white", alpha=0.6, label="Grain Data")

    # x_axis = np.linspace(DIST_XLIM[0], DIST_XLIM[1], 200)
    # if len(normalized_radii) > 1: 
    #     kde = gaussian_kde(normalized_radii)
    #     ax.plot(x_axis, kde(x_axis), color="navy", lw=2, label="Smoothed Trend")

    # ax.axvline(1.0, color='red', linestyle='--', label="Mean (1.0)")
    # ax.set_xlabel("Normalized Radius (R / \u27e8R\u27e9)", fontsize=13)
    # ax.set_ylabel("Probability Density", fontsize=13)
    # ax.set_title(f"Grain Size Distribution (Time {timestep})", fontsize=14)
    # ax.set_xlim(DIST_XLIM)
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(csv_dir, f"dist_t{timestep}.png"),
    #             dpi=150, bbox_inches="tight", facecolor="white")
    # plt.close(fig)
    
    # Return true grain count for the trend plot
    return num_true_grains

# ── 3. Plot grain map images, Export ML Data, & Animate ───────────────────
def plot_and_save_selected_timesteps(timestep_numbers, all_timesteps, run_name, run_id, temperature):
    all_grain_ids = set()
    for ts in all_timesteps:
        all_grain_ids.update(np.unique(ts).tolist())
    unique_ids        = sorted(list(all_grain_ids))
    n_unique          = len(unique_ids)
    grain_id_to_color = {gid: idx for idx, gid in enumerate(unique_ids)}

    cmap_base  = plt.get_cmap("nipy_spectral")
    colors     = [cmap_base(i / max(n_unique, 1)) for i in range(max(n_unique, 1))]
    cmap_fixed = ListedColormap(colors)

    colored_timesteps = []
    for ts in all_timesteps:
        colored = np.zeros_like(ts, dtype=int)
        for gid in unique_ids:
            colored[ts == gid] = grain_id_to_color[gid]
        colored_timesteps.append(colored)

    # Normalize SPPARKS timestamps so the first dump is t=0
    start_time = timestep_numbers[0]
    normalized_times = [ts - start_time for ts in timestep_numbers]

    # tracked_times = []
    # tracked_counts = []
    
    # To store frames for the GIF animation
    # animation_frames = []

    for t in TARGET_TIMESTEPS:
        os.makedirs(os.path.join(OUT_ROOT, f"timestep_{t}"), exist_ok=True)

        closest_idx = min(range(len(normalized_times)), key=lambda i: abs(normalized_times[i] - t))
        actual_physical_time = timestep_numbers[closest_idx]
        
        if abs(normalized_times[closest_idx] - t) > 15.0:  
            print(f"  Warning: Target timestep {t} is out of bounds (closest is {actual_physical_time}). Skipping.")
            continue

        grains   = all_timesteps[closest_idx]
        colored  = colored_timesteps[closest_idx]
        n_grains = len(np.unique(grains))
        out_dir  = os.path.join(OUT_ROOT, f"timestep_{t}")

        # --- A. Save RGB Visualizations ---
        # We want pixels = figsize * dpi. If N=512 and figsize=5.12, dpi must be 100.
        target_dpi = 100
        fig_size_inches = N / target_dpi  # 5.12 inches for N=512
        # Example: run_1_temp_0.752_time_100
        file_base = f"run_{run_id}_temp_{temperature:.3f}_timestep_{t}"

        fig, ax = plt.subplots(figsize=(fig_size_inches, fig_size_inches), dpi=target_dpi)
        
        ax.imshow(
            colored,
            cmap=cmap_fixed,
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=n_unique - 1
        )
        
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        
        # out_path = os.path.join(out_dir, f"{run_name}_time_{t}.png")
        # pad_inches=0 is critical to prevent Matplotlib from adding a border
        rgb_path = os.path.join(out_dir, f"{file_base}_rgb.png")
        plt.savefig(rgb_path, dpi=target_dpi, pad_inches=0, bbox_inches='tight')
        plt.close(fig)
        
        # # Save necessary info for the animation
        # animation_frames.append({
        #     'array': colored,
        #     'time': t,
        #     'grains': n_grains
        # })

        # --- B. Save Raw Array ---
        np.save(os.path.join(out_dir, f"{file_base}_raw.npy"), grains)

        # --- C. Generate & Save Boundary Maps ---
        boundary_map = get_boundaries(grains)
        np.save(os.path.join(out_dir, f"{run_name}_boundaries_{t}.npy"), boundary_map)
       
        fig_b, ax_b = plt.subplots(figsize=(fig_size_inches, fig_size_inches), dpi=target_dpi)
        
        ax_b.imshow(
            boundary_map, 
            cmap='gray_r', 
            origin='lower', 
            interpolation='nearest'
        )
        
        ax_b.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        
        b_path = os.path.join(out_dir, f"{file_base}_boundary.png")
        plt.savefig(b_path, dpi=target_dpi, pad_inches=0, bbox_inches='tight')
        plt.close(fig_b)

        # Compute distributions and store grain count
        # grain_count = compute_and_save_distribution(grains, t, run_id, temperature)
        # tracked_times.append(t)
        # tracked_counts.append(grain_count)

    # # --- D. Plot Grain Count vs Time ---
    # csv_dir = os.path.join(DIST_ROOT, f"run_{run_id}")
    # if tracked_times:
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     ax.plot(tracked_times, tracked_counts, marker='o', linestyle='-', color='purple', linewidth=2)
    #     ax.set_title(f"Grain Count vs Time (Run {run_id}, T={temperature:.2f})", fontsize=14)
    #     ax.set_xlabel("Target Timestep", fontsize=12)
    #     ax.set_ylabel("Number of True Grains", fontsize=12)
    #     ax.grid(True, linestyle='--', alpha=0.6)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(csv_dir, f"grain_count_vs_time_run_{run_id}.png"),
    #                 dpi=150, bbox_inches="tight", facecolor="white")
    #     plt.close(fig)

    # # --- E. Create and Save GIF Animation ---
    # if animation_frames:
    #     fig_anim, ax_anim = plt.subplots(figsize=(6, 6), dpi=100)
        
    #     # Manually add padding to the top of the figure so the title never gets cut off
    #     fig_anim.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.9)
        
    #     # Initialize plot with the first frame
    #     im_anim = ax_anim.imshow(
    #         animation_frames[0]['array'], cmap=cmap_fixed,
    #         interpolation="nearest", origin="lower",
    #         vmin=0, vmax=n_unique - 1
    #     )
    #     ax_anim.set_xlabel("X Position", fontsize=10)
    #     ax_anim.set_ylabel("Y Position", fontsize=10)
    #     ax_anim.grid(False)
    #     ax_anim.set_aspect("equal")

    #     def update(frame_dict):
    #         im_anim.set_array(frame_dict['array'])
    #         # Keep pad at a safe distance
    #         ax_anim.set_title(
    #             f"{run_name} - Time = {frame_dict['time']} \n"
    #             f"({frame_dict['grains']} grains, T={temperature:.2f})",
    #             fontsize=12, fontweight="bold", pad=15
    #         )
    #         return [im_anim, ax_anim.title]

    #     anim = animation.FuncAnimation(
    #         fig_anim, update, frames=animation_frames, 
    #         interval=800, blit=False  # Disabled blit to ensure the title updates cleanly
    #     )
        
    #     # Save as a looping GIF using PillowWriter
    #     gif_path = os.path.join(csv_dir, f"animation_run_{run_id}.gif")
    #     writer = animation.PillowWriter(fps=1.2)  
        
    #     # Removed bbox_inches="tight" to respect the subplots_adjust margins
    #     anim.save(gif_path, writer=writer)
    #     plt.close(fig_anim)
    #     print(f"    Animation saved: {gif_path}")

# ── 4. Single run ─────────────────────────────────────────────────────────
def run_one_sample(run_id):
    seed = np.random.randint(1, 10**9)
    temperature = np.random.uniform(TEMP_MIN, TEMP_MAX)
    print(f"\nRun {run_id}: seed={seed}, temp={temperature:.3f}")

    run_dir = os.path.join(RUN_ROOT, f"run_{run_id}_temp_{temperature:.3f}")
    os.makedirs(run_dir, exist_ok=True)
    
    # UPDATED FOR PACE ICE HPC SLURM
    cmd = [
        "srun", "-n", "4", SPPARKS_PATH, 
        "-in", os.path.abspath(BASE_INPUT),
        "-var", "mySeed", str(seed),
        "-var", "myTemp", str(round(temperature, 3))
    ]
    dump_path = os.path.join(run_dir, "grain_growth.dump")

    shutil.copy(BASE_INPUT, os.path.join(run_dir, BASE_INPUT))

    print(f"  Run {run_id}: simulating (this may take a minute)...")
    process = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)

    if process.returncode != 0:
        print(f"  SPPARKS Error in Run {run_id}:\n{process.stderr}")
        # Note: If `srun` complains about --mpi=pmi2 on your specific PACE node, 
        # you can revert this line back to "mpirun", "-np", "4"
        raise RuntimeError(f"SPPARKS crashed on run {run_id}.")

    if not os.path.exists(dump_path):
        raise FileNotFoundError(f"Dump not found: {dump_path}")

    print(f"  Run {run_id}: simulation complete. Reading dump...")
    ts_nums, ts_data = read_dump_all_timesteps(dump_path, N)
    plot_and_save_selected_timesteps(ts_nums, ts_data, f"run_{run_id}", run_id, temperature)
    print(f"  Run {run_id}: done.")

# ── 5. Main ───────────────────────────────────────────────────────────────
def main():
    # 1. Create directories safely (DO NOT delete them to avoid race conditions on HPC)
    for folder in [RUN_ROOT, OUT_ROOT, DIST_ROOT]:
        os.makedirs(folder, exist_ok=True)

    # 2. Get the specific run_id assigned to this job by the Slurm Array
    # (Defaults to 1 if you test it on your local laptop without Slurm)
    run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

    # 3. Run only that specific sample
    run_one_sample(run_id)

    print(f"\nDataset generation for run {run_id} complete.")
    print(f"  Grain images & ML Arrays -> {OUT_ROOT}/")
    print(f"  Distributions, Animations & Metadata -> {DIST_ROOT}/")

if __name__ == "__main__":
    main()
