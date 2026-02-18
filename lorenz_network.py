import numpy as np
import scipy.integrate as integrate
import networkx as nx
import matplotlib.pyplot as plt

# --- Parameters ---
N = 20          # Number of nodes
K_sw = 4        # Each node is connected to k nearest neighbors in ring topology
P_sw = 0.1      # Probability of rewiring each edge
T_sim = 50.0    # Simulation time
DT = 0.01       # Time step
SIGMA = 10.0
RHO = 28.0
BETA = 8.0/3.0
COUPLING_STRENGTH = 10.0 # Increased coupling strength for better sync chance
NEGATIVE_FRACTION = 0.2 # Fraction of edges with negative weights


# --- Network Generation ---
def generate_stable_network(n, k, p, neg_frac):
    """Generates a network with mixed couplings and checks stability."""
    MAX_ATTEMPTS = 100
    LYAPUNOV_EXP = 0.906  # Approx max Lyapunov exponent for Lorenz (rho=28)
    
    for attempt in range(MAX_ATTEMPTS):
        # Generate Small-World Network (Watts-Strogatz)
        G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=42+attempt) # Change seed if unstable

        # Assign random weights
        for u, v in G.edges():
            if np.random.rand() < neg_frac:
                G[u][v]['weight'] = -np.random.uniform(1.0, 5.0) # Negative weights
            else:
                G[u][v]['weight'] = np.random.uniform(5.0, 15.0) # Positive weights (stronger)

        # Create Adjacency Matrix
        adj = nx.to_numpy_array(G, weight='weight')
        
        # Construct Laplacian-like Coupling Matrix C
        # C_ij = A_ij for i != j
        # C_ii = -sum(A_ik) for k != i
        # This ensures zero row sums: sum_j C_ij = C_ii + sum_{j!=i} C_ij = 0
        C = adj.copy()
        row_sums = np.sum(adj, axis=1)
        np.fill_diagonal(C, -row_sums)
        
        # Check Eigenvalues of C
        eigvals = np.linalg.eigvals(C)
        eigvals = np.sort(np.real(eigvals))
        
        # Stability condition for synchronization:
        # 1. All non-zero eigenvalues must be negative (stable transverse manifold).
        # 2. The magnitude of the "least negative" non-zero eigenvalue (lambda_2)
        #    must be large enough to overcome the chaotic divergence (Lyapunov exponent).
        #    Condition: |lambda_2| * Coupling_Strength > Lyapunov_Exp (approx heuristic)
        #    Actually, in our equation: dot(x) = ... + C * sum(W(xj-xi)) 
        #    The effective coupling is COUPLING_STRENGTH * eigenvalues.
        
        # Filter out the zero eigenvalue (approx)
        non_zero_eigs = eigvals[np.abs(eigvals) > 1e-5]
        lambda_2 = np.max(non_zero_eigs) # The closest to zero (least stable)
        
        # Check 1: Negative Definiteness (on transverse subspace)
        is_negative_definite = np.all(non_zero_eigs < 0)
        
        # Check 2: Coupling Strength Sufficiency (This depends on global scaling C)
        # We can't strictly enforce this in generation without knowing C, but we can report it.
        # We assume global C will be high enough.
        
        if len(non_zero_eigs) == n - 1 and is_negative_definite:
            print(f"Found stable network configuration on attempt {attempt+1}!")
            print(f"Eigenvalues: Max (lambda_2) = {lambda_2:.4f}, Min = {np.min(non_zero_eigs):.4f}")
            print(f"Condition Check: |lambda_2| = {abs(lambda_2):.4f}")
            return G, C, eigvals, lambda_2
        
    print("WARNING: Could not find a strictly stable network configuration. Using last attempt.")
    return G, C, eigvals, -1.0

print("Generating network...")
G, C_matrix, eigenvalues, lambda_2 = generate_stable_network(N, K_sw, P_sw, NEGATIVE_FRACTION)

# Verify Synchronization Condition explicitly
LYAPUNOV_EXP = 0.906
effective_coupling = abs(lambda_2) * COUPLING_STRENGTH
print(f"--- Synchronization Condition Check ---")
print(f"Max Lyapunov Exponent (L_max): ~{LYAPUNOV_EXP}")
print(f"Effective Coupling Strength (|lambda_2| * C): {effective_coupling:.4f}")
if effective_coupling > LYAPUNOV_EXP:
    print(f"✅ Condition satisfied: {effective_coupling:.4f} > {LYAPUNOV_EXP}")
else:
    print(f"⚠️ Condition NOT satisfied: {effective_coupling:.4f} < {LYAPUNOV_EXP} (Expect poor sync)")


# --- Lorenz System with Coupling ---
def lorenz_network_ode(t, state_flat):
    # Reshape state to (N, 3) where columns are x, y, z
    state = state_flat.reshape((N, 3))
    # State vectors
    X = state[:, 0]
    Y = state[:, 1]
    Z = state[:, 2]
    
    # Lorenz dynamics (local)
    dX_local = SIGMA * (Y - X)
    dY_local = X * (RHO - Z) - Y
    dZ_local = X * Y - BETA * Z
    
    # Coupling Term
    # Diffusive coupling on X variable: dot(C, X)
    # Since C is constructed such that row sums are zero, C @ X gives exactly sum C_ij (x_j - x_i)
    coupling_X = COUPLING_STRENGTH * (C_matrix @ X)
    
    dX = dX_local + coupling_X
    dY = dY_local
    dZ = dZ_local
    
    return np.stack([dX, dY, dZ], axis=1).flatten()

# --- Initial Conditions ---
# Random initial state for each node
initial_state = np.random.rand(N, 3) * 20.0 - 10.0
initial_state_flat = initial_state.flatten()

# --- Integration ---
t_span = (0.0, T_sim)
t_eval = np.arange(0.0, T_sim, DT)

print("Starting integration...")
sol = integrate.solve_ivp(
    lorenz_network_ode, 
    t_span, 
    initial_state_flat, 
    t_eval=t_eval, 
    method='RK45',
    rtol=1e-6,
    atol=1e-6
)
print("Integration finished.")

# Reshape solution: (N, 3, time_steps)
states = sol.y.reshape((N, 3, len(sol.t)))
X_all = states[:, 0, :] # (N, time_steps)

# --- Analysis: Synchronization Error ---
# Error = std(X) across nodes at each time step
sync_error = np.std(X_all, axis=0)

# --- Plotting ---
fig = plt.figure(figsize=(14, 10))

# 1. Connectivity Matrix
ax1 = plt.subplot(2, 3, 1)
im = ax1.imshow(C_matrix, cmap='coolwarm', vmin=-np.max(np.abs(C_matrix)), vmax=np.max(np.abs(C_matrix)))
plt.colorbar(im, ax=ax1, label='Coupling Weight')
ax1.set_title('Coupling Matrix $C$')
ax1.set_xlabel('Node Index')
ax1.set_ylabel('Node Index')

# 2. Eigenvalue Spectrum
ax2 = plt.subplot(2, 3, 2)
ax2.plot(np.real(eigenvalues), np.imag(eigenvalues), 'bo', markersize=5)
ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
ax2.set_title('Eigenvalue Spectrum of $C$')
ax2.set_xlabel('Real Part')
ax2.set_ylabel('Imaginary Part')
ax2.grid(True, alpha=0.3)

# 3. Network Graph
ax3 = plt.subplot(2, 3, 3)
pos = nx.spring_layout(G, seed=42)
edges = G.edges(data=True)
pos_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
neg_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='k', ax=ax3)
nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color='g', alpha=0.5, width=1.0, ax=ax3)
nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color='r', alpha=0.8, width=1.5, ax=ax3)
ax3.set_title('Network Structure\n(Green=Pos, Red=Neg)')
ax3.axis('off')

# 4. Time Series
ax4 = plt.subplot(2, 1, 2)
# Plot first 5 nodes
for i in range(min(5, N)):
    ax4.plot(sol.t, X_all[i, :], label=f'Node {i}', alpha=0.7, linewidth=1.0)
# Also plot sync error on twin axis
ax4_right = ax4.twinx()
ax4_right.plot(sol.t, sync_error, 'k--', alpha=0.5, label='Sync Error', linewidth=1.5)
ax4_right.set_ylabel('Sync Error (std dev)', color='k')
ax4.set_title('Time Series & Synchronization Error')
ax4.set_xlabel('Time')
ax4.set_ylabel('X State')
ax4.legend(loc='upper left')

plt.tight_layout()
plt.savefig('lorenz_network_simulation.png', dpi=300)
print("Simulation plot saved to 'lorenz_network_simulation.png'.")

final_sync_error = np.mean(sync_error[-int(len(sync_error)*0.1):])
print(f"Final Average Sync Error (last 10%): {final_sync_error:.4f}")

