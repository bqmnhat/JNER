import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from torch_geometric.nn import GATConv, GCNConv, global_add_pool  # Import PyG's GATConv and GCNConv

class GNNEncoder(nn.Module):
    """
    GNN encoder using PyG's GATConv
    """
    def __init__(self, num_in_features, num_hidden_features, num_of_heads, dropout_prob, alpha):
        super(GNNEncoder, self).__init__()
        self.gat1 = GATConv(num_in_features, num_hidden_features, heads=num_of_heads, dropout=dropout_prob, negative_slope=alpha)
        self.gat2 = GATConv(num_hidden_features * num_of_heads, num_hidden_features, heads=num_of_heads, dropout=dropout_prob, negative_slope=alpha) # Add another GAT layer if needed

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training) # Apply dropout to input features
        h = self.gat1(h, edge_index)
        h = F.elu(h) # Use ELU activation as in original GAT paper (Veličković P. et al., 2018)
        h = F.dropout(h, p=0.6, training=self.training) # Apply dropout after first GAT layer
        h = self.gat2(h, edge_index) # Add second GAT layer
        return h
    
class NodeAttention(nn.Module):
    """Node-level attention module to separate causal and trivial features"""
    def __init__(self, input_dim, dropout_prob=0.6):
        super(NodeAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim // 2, 2)  # 2 outputs: causal and trivial
        )

    def forward(self, node_features):
        # Compute attention scores [batch_size, 2]
        scores = self.mlp(node_features)
        attention = F.softmax(scores, dim=1)
        # Split into causal and trivial attention
        node_c = attention[:, 0] # nodes attention scores causal
        node_t = attention[:, 1] # nodes attention scores trivial
        return node_c, node_t


class EdgeAttention(nn.Module):
    """Edge-level attention for causal and trivial connections"""
    def __init__(self, input_dim, dropout_prob=0.6):
        super(EdgeAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim, 2)  # 2 outputs: causal and trivial
        )

    def forward(self, node_features, edge_index):
        # Get features for each edge
        src_features = node_features[edge_index[0]]  # source features [num_edges, num_dim]
        dst_features = node_features[edge_index[1]]  # destination features [num_edges, num_dim]
        # Concatenate source and destination features
        edge_features = torch.cat([src_features, dst_features], dim=1)  # [num_edges, 2*input_dim]
        # Compute attention scores
        scores = self.mlp(edge_features)  # [num_edges, 2]
        attention = F.softmax(scores, dim=1)
        # Split into causal and trivial attention
        edge_c = attention[:, 0]  # [num_edges]
        edge_t = attention[:, 1]  # [num_edges]
        return edge_c, edge_t
    
class GraphConv(nn.Module): # now using PyG's GCNConv for context/object branches
    """Using PyG's GCNConv for graph convolution layer"""
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.conv = GCNConv(in_features, out_features) # Using PyG's GCNConv

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight=edge_weight) # Pass edge_weight to GCNConv

def scatter_mean_alternative(gated_x, batch):
    """
    Alternative implementation for torch_scatter.scatter_mean
    without using torch_scatter.

    Args:
        gated_x: Tensor of node features (e.g., [N, features]).
        batch: Batch assignment vector (e.g., [N]).

    Returns:
        Tensor of mean node features per batch (e.g., [num_batches, features]).
    """
    if batch is None:
        return gated_x.mean(dim=0, keepdim=True)  # Return mean over all nodes if no batch info

    unique_batches = torch.unique(batch)
    batch_means = []
    for b_idx in unique_batches:
        mask = (batch == b_idx)
        current_batch_nodes = gated_x[mask]
        batch_mean = current_batch_nodes.mean(dim=0) # Mean across nodes within this batch
        batch_means.append(batch_mean)

    return torch.stack(batch_means)

class ReadoutFunction(nn.Module): # Simplified Readout using global_add_pool
    """Readout function for graph-level representations using global_add_pool"""
    def __init__(self, input_dim):
        super(ReadoutFunction, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim) # Linear layer before pooling, if needed

    def forward(self, x, batch=None):
        # Apply linear transformation
        x_linear = self.linear(x)
        # Use global_add_pool for readout
        if batch is not None:
            return global_add_pool(x_linear, batch) # Using global_add_pool from PyG
        return torch.sum(x_linear, dim=0, keepdim=True) # Sum for single graph

class RandomIntervention(nn.Module): # New module for node-level randomization
    """Randomly shuffles node features within each graph in a batch."""
    def __init__(self):
        super(RandomIntervention, self).__init__()

    def forward(self, h_t, batch):
        """
        Args:
            h_t: Trivial node representations [N, feature_dim].
            batch: Batch assignment vector [N].

        Returns:
            shuffled_h_t_pooled: Graph-level trivial representations after node shuffling, [num_graphs, feature_dim].
        """
        if batch is None:
            return h_t.mean(dim=0, keepdim=True) # Fallback for single graph

        shuffled_h_t_nodes = []
        unique_batches = torch.unique(batch)
        for b_idx in unique_batches:
            mask = (batch == b_idx)
            current_batch_h_t = h_t[mask]
            num_nodes_batch = current_batch_h_t.size(0)
            # Generate random permutation indices for nodes within the batch
            perm_indices = torch.randperm(num_nodes_batch)
            shuffled_batch_h_t = current_batch_h_t[perm_indices] # Shuffle node features within the batch
            shuffled_h_t_nodes.append(shuffled_batch_h_t)

        shuffled_h_t = torch.cat(shuffled_h_t_nodes, dim=0) # Concatenate shuffled node features back

        # Pool shuffled node features to get graph-level representation
        shuffled_h_t_pooled = global_add_pool(shuffled_h_t, batch) # Use global_add_pool for consistency

        return shuffled_h_t_pooled
    
class Classifier(nn.Module):
    """Classifier for graph classification"""
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)


class CAL_GAT(nn.Module):
    """Graph Attention Network with Causal Attention Learning (Improved)"""
    def __init__(self, num_in_features, num_hidden_features, num_out_features, num_of_heads=8,
                 dropout_prob=0.6, alpha=0.2, lambda1=0.1, lambda2=0.1):
        super(CAL_GAT, self).__init__()

        # GNN Encoder with PyG's GATConv
        self.gnn_encoder = GNNEncoder(num_in_features, num_hidden_features, num_of_heads, dropout_prob, alpha)

        hidden_dim = num_hidden_features * num_of_heads # Correct hidden_dim to match GNNEncoder output

        # Node and edge attention
        self.node_attention = NodeAttention(hidden_dim, dropout_prob)
        self.edge_attention = EdgeAttention(hidden_dim, dropout_prob)

        # GraphConv layers (now using PyG's GCNConv)
        self.graph_conv_causal = GraphConv(hidden_dim, hidden_dim) # Using the modified GraphConv which is GCNConv now
        self.graph_conv_trivial = GraphConv(hidden_dim, hidden_dim)

        # Readout functions (simplified using global_add_pool)
        self.readout_causal = ReadoutFunction(hidden_dim) # Using simplified ReadoutFunction
        self.readout_trivial = ReadoutFunction(hidden_dim)

        # Random Intervention module
        self.random_intervention = RandomIntervention() # Initialize RandomIntervention module

        # Classifiers
        self.classifier_causal = Classifier(hidden_dim, num_out_features)
        self.classifier_trivial = Classifier(hidden_dim, num_out_features)
        self.classifier_combined = Classifier(hidden_dim * 2, num_out_features)

        # Hyperparameters for loss functions
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, data, eval_random=True): # eval_random is now passed to random_intervention if needed for conditional randomization
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None

        # Step 1: Encode graph with GNN
        H = self.gnn_encoder(x, edge_index)

        # Step 2: Compute node and edge attention scores
        alpha_c, alpha_t = self.node_attention(H)
        beta_c, beta_t = self.edge_attention(H, edge_index)

        # Step 3: Create attended representations
        H_c = H * alpha_c.unsqueeze(1)
        H_t = H * alpha_t.unsqueeze(1)

        edge_attr_c = edge_attr * beta_c.unsqueeze(1) if edge_attr is not None else None
        edge_attr_t = edge_attr * beta_t.unsqueeze(1) if edge_attr is not None else None

        # Step 4: Process through GraphConv layers (GCNConv)
        G_c = self.graph_conv_causal(H_c, edge_index, edge_weight=beta_c) # Pass edge_weight directly to GCNConv
        G_t = self.graph_conv_trivial(H_t, edge_index, edge_weight=beta_t) # Pass edge_weight directly to GCNConv

        # Step 5: Readout functions to get graph-level representations
        h_G_c = self.readout_causal(G_c, batch)
        h_G_t = self.readout_trivial(G_t, batch)

        # Step 6: Predictions (causal and trivial branches)
        z_G_c = self.classifier_causal(h_G_c)
        z_G_t = self.classifier_trivial(h_G_t)

        # Step 7: Causal intervention and combined prediction
        shuffled_h_G_t = self.random_intervention(h_G_t, batch) # Use RandomIntervention module for shuffling
        h_G_combined = h_G_c + shuffled_h_G_t # Combine causal and intervened trivial features
        z_G_prime = self.classifier_combined(h_G_combined)

        return z_G_c, z_G_t, z_G_prime, h_G_c, h_G_t

    def compute_losses(self, z_G_c, z_G_t, z_G_prime, labels, num_classes):
        """Compute CAL losses"""
        if len(labels.shape) == 1 or labels.shape[1] == 1:
            L_sup = F.cross_entropy(z_G_c, labels.view(-1))
        else:
            L_sup = F.binary_cross_entropy_with_logits(z_G_c, labels.float())

        uniform_target = torch.ones(z_G_t.size(0), num_classes).to(z_G_t.device)
        L_unif = F.kl_div(F.log_softmax(z_G_t, dim=1), uniform_target, reduction='batchmean')

        if len(labels.shape) == 1 or labels.shape[1] == 1:
            L_caus = F.cross_entropy(z_G_prime, labels.view(-1))
        else:
            L_caus = F.binary_cross_entropy_with_logits(z_G_prime, labels.float())

        total_loss = L_sup + self.lambda1 * L_unif + self.lambda2 * L_caus
        return total_loss, L_sup, L_unif, L_caus
    
def prepare_graph_data(adjacency_matrix, covariance_features):
    """Convert adjacency matrix and covariance features to PyTorch Geometric format"""
    # Get edges from adjacency matrix (where weight > 0)
    edges = np.where(adjacency_matrix > 0)
    edge_index = torch.tensor(np.vstack((edges[0], edges[1])), dtype=torch.long)
    # Get edge weights from adjacency matrix
    edge_attr = torch.tensor(adjacency_matrix[edges], dtype=torch.float)
    # Process covariance features
    num_nodes = covariance_features.shape[0]
    feature_dim = covariance_features.shape[1]
    # Extract meaningful features from covariance matrices
    node_features = []
    for i in range(num_nodes):
        covar = covariance_features[i]
        # 1. Diagonal elements (variances)
        diag_features = np.diag(covar)
        # 2. Upper triangular part (correlations)
        triu_indices = np.triu_indices(feature_dim, k=1)
        triu_features = covar[triu_indices]
        # 3. Combine features
        node_feat = np.concatenate([diag_features, triu_features])
        node_features.append(node_feat)
    x = torch.tensor(np.stack(node_features), dtype=torch.float)
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def prepare_dataset(adjacency_matrices, covariance_features_list, labels):
    """Prepare dataset for multiple graphs"""
    dataset = []
    for i, (adj, cov) in enumerate(zip(adjacency_matrices, covariance_features_list)):
        data = prepare_graph_data(adj, cov)
        # Add graph label
        data.y = torch.tensor([labels[i]], dtype=torch.long)
        dataset.append(data)
    return dataset


def load_local_data(data_path):
    """
    Load adjacency matrices, covariance features, and labels from local files.
    Assumes files are in .npy format and named as:
    - adj_matrix_graph_i.npy: Adjacency matrix for graph i
    - covar_features_graph_i.npy: Covariance features for graph i
    - labels.npy:  Numpy array of labels for all graphs
    Parameters:
    -----------
    data_path : str
        Path to the directory containing the data files.
    Returns:
    --------
    adjacency_matrices : list
        List of adjacency matrices (numpy arrays).
    covariance_features_list : list
        List of covariance features (numpy arrays).
    labels : numpy.ndarray
        Numpy array of graph labels.
    """
    adjacency_matrices = []
    covariance_features_list = []
    # Load labels
    labels_path = os.path.join(data_path, 'labels.npy')
    labels = np.load(labels_path)
    graph_index = 0 # Start index for graph files
    while True: # Try loading files until no more are found
        adj_matrix_file = os.path.join(data_path, f'adj_matrix_graph_{graph_index}.npy')
        covar_features_file = os.path.join(data_path, f'covar_features_graph_{graph_index}.npy')
        if not os.path.exists(adj_matrix_file) or not os.path.exists(covar_features_file):
            break # Stop loading if files for the current graph index are not found
        adj_matrix = np.load(adj_matrix_file)
        covar_features = np.load(covar_features_file)
        adjacency_matrices.append(adj_matrix)
        covariance_features_list.append(covar_features)
        graph_index += 1 # Increment for next graph
    if not adjacency_matrices: # Check if any data was loaded
        raise FileNotFoundError(f"No adjacency or covariance feature files found in: {data_path}. "
                                f"Make sure files are named as 'adj_matrix_graph_i.npy' and "
                                f"'covar_features_graph_i.npy' and 'labels.npy' are in the directory.")
    return adjacency_matrices, covariance_features_list, labels


# ------------------------------------------------------------------------
# Training and evaluation functions (modified for causal intervention)
# ------------------------------------------------------------------------
def train_cal_gat(model, train_loader, optimizer, device, num_classes):
    """Train the CAL_GAT model for one epoch with causal intervention (integrated in model)"""
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Step 1-7: Forward pass of CAL_GAT model
        # The model's forward pass *now* includes the RandomIntervention module,
        # so it handles shuffling internally.
        z_G_c, z_G_t, z_G_prime, _, _ = model(batch) # Single forward pass, shuffling is inside model

        # Compute losses
        loss, L_sup, L_unif, L_caus = model.compute_losses(
            z_G_c, z_G_t, z_G_prime, batch.y, num_classes
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(train_loader.dataset)


def evaluate_cal_gat(model, loader, device, num_classes):
    """Evaluate the CAL_GAT model (no causal intervention during evaluation)"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass (no shuffled trivial features in eval)
            z_G_c, z_G_t, z_G_prime, h_G_c, h_G_t = model(batch) # shuffled_h_G_t=None by default

            # Compute losses
            loss, _, _, _ = model.compute_losses(
                z_G_c, z_G_t, z_G_prime, batch.y, num_classes
            )
            total_loss += loss.item() * batch.num_graphs

            # Get predictions (using causal branch prediction for evaluation)
            preds = z_G_c.argmax(dim=1).cpu().numpy() # Evaluate based on causal branch, or z_G_prime if you want to evaluate combined.
            labels = batch.y.cpu().numpy().flatten()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader.dataset), accuracy, all_preds, all_labels


# ------------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------------

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    num_classes = 3
    hidden_dim = 32
    heads = 8 # Increased heads as GATConv is used now
    dropout = 0.6 # Increased dropout as in original GAT paper
    alpha = 0.2 # Alpha for LeakyReLU in GATConv
    learning_rate = 0.005
    epochs = 50
    batch_size = 32

    # --- Load data from local files ---
    data_path = 'path/to/your/local/data'
    print(f"Loading data from: {data_path}")
    try:
        adj_matrices, covar_features, labels = load_local_data(data_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 2. Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(adj_matrices, covar_features, labels)

    # Get feature dimensions
    input_dim = dataset[0].x.size(1)
    print(f"Input feature dimension: {input_dim}")

    # 3. Split data into train/val/test
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    print(f"Train set: {len(train_dataset)}, Validation set: {len(val_dataset)}, Test set: {len(test_dataset)}")

    # 4. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 5. Initialize the model
    print("Initializing CAL_GAT model...")
    model = CAL_GAT(
        num_in_features=input_dim,
        num_hidden_features=hidden_dim,
        num_out_features=num_classes,
        num_of_heads=heads,
        dropout_prob=dropout,
        alpha=alpha,
        lambda1=0.5,  # Weight for uniform loss
        lambda2=1.0  # Weight for causal intervention loss
    ).to(device)

    # 6. Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )

    # 7. Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    best_model = None

    for epoch in range(epochs):
        # Train
        train_loss = train_cal_gat(model, train_loader, optimizer, device, num_classes)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_acc, _, _ = evaluate_cal_gat(model, val_loader, device, num_classes)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            print(f"Epoch {epoch+1} - Best validation accuracy improved to: {best_val_acc:.4f}. Saving model...") # Print when best model is saved
        else:
            print(f"Epoch {epoch+1} - Validation accuracy did not improve.") # Print when validation accuracy does not improve

        # --- ADD PRINT STATEMENTS HERE ---
        print(f"Epoch {epoch+1}/{epochs}:") # Print Epoch number
        print(f"  Train Loss:     {train_loss:.4f}") # Print Training Loss
        print(f"  Validation Loss: {val_loss:.4f}")   # Print Validation Loss
        print(f"  Validation Acc:  {val_acc:.4f}")    # Print Validation Accuracy
        print("-" * 30) # Separator for epochs

    # 8. Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(best_model)
    test_loss, test_acc, test_preds, test_labels = evaluate_cal_gat(model, test_loader, device, num_classes)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # 9. Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # 10. Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend() # Add legend to the plot
    plt.show()


if __name__ == '__main__':
    main()