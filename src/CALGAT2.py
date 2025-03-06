import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList, Sequential, ReLU, Dropout
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from functools import partial  # Import partial for CausalGAT
from sklearn.model_selection import StratifiedKFold
import torch_geometric

# --- 0. Utility Functions ---

def matrix_to_array_of_vectors(matrix):
    """
    Convert a NumPy matrix to an array of NumPy vectors where each vector
    is a row of the original matrix with its diagonal element set to 0.
    """
    modified_matrix = matrix.copy()
    min_dim = min(modified_matrix.shape[0], modified_matrix.shape[1])
    for i in range(min_dim):
        modified_matrix[i, i] = 0
    row_vectors = np.empty(modified_matrix.shape[0], dtype=object)
    for i in range(modified_matrix.shape[0]):
        row_vectors[i] = modified_matrix[i, :]
    return row_vectors

# --- 1. Model Components ---

class NodeAttention(torch.nn.Module):
    """Simplified Node Attention Mechanism (Linear + Softmax)"""
    def __init__(self, in_features, dropout_prob):
        super(NodeAttention, self).__init__()
        self.attn_fc = Linear(in_features, 2)  # Linear layer for attention scores
        self.dropout = Dropout(dropout_prob)

    def forward(self, node_embeddings):
        attn_scores = self.attn_fc(node_embeddings) # [Num_nodes_total, 2]
        attn_scores = F.softmax(attn_scores, dim=-1) # Apply softmax for attention weights [Num_nodes_total, 2]
        attn_scores = self.dropout(attn_scores)
        return attn_scores # Returns node attention scores for context and object

class EdgeAttention(torch.nn.Module):
    """Simplified Edge Attention Mechanism (Linear + Softmax)"""
    def __init__(self, in_features, dropout_prob):
        super(EdgeAttention, self).__init__()
        self.attn_fc = Linear(in_features * 2, 2) # Linear layer for edge attention scores (takes concatenated node features)
        self.dropout = Dropout(dropout_prob)

    def forward(self, node_embeddings, edge_index):
        """
        Args:
            node_embeddings:  [Num_nodes_total, feature_dim]
            edge_index: [2, Num_edges]
        Returns:
            Edge attention weights: [Num_edges, 2]
        """
        row, col = edge_index
        edge_feature = torch.cat([node_embeddings[row], node_embeddings[col]], dim=-1) # Concatenate features of source and target nodes [Num_edges, 2*feature_dim]
        attn_scores = self.attn_fc(edge_feature) # [Num_edges, 2]
        attn_scores = F.softmax(attn_scores, dim=-1) # Apply softmax [Num_edges, 2]
        attn_scores = self.dropout(attn_scores)
        return attn_scores # Returns edge attention scores for context and object

class ImprovedGNNEncoder_v2(torch.nn.Module):
    """Improved GNN Encoder with initial GCNConv and BatchNorm"""
    def __init__(self, num_in_features, num_hidden_features, num_of_heads, dropout_prob, alpha):
        super(ImprovedGNNEncoder_v2, self).__init__()

        self.initial_conv = GCNConv(num_in_features, num_hidden_features) # Initial GCNConv layer
        self.bn_initial = BatchNorm1d(num_hidden_features)

        self.conv1 = GATConv(in_channels=num_hidden_features,
                               out_channels=num_hidden_features,
                               heads=num_of_heads,
                               dropout=dropout_prob,
                               negative_slope=alpha) # Use negative_slope
        self.bn1 = BatchNorm1d(num_hidden_features * num_of_heads)
        self.conv2 = GATConv(in_channels=num_hidden_features * num_of_heads,
                               out_channels=num_hidden_features,
                               heads=num_of_heads,
                               dropout=dropout_prob,
                               negative_slope=alpha) # Use negative_slope
        self.bn2 = BatchNorm1d(num_hidden_features * num_of_heads)

    def forward(self, x, edge_index):
        x = F.relu(self.initial_conv(x, edge_index))
        x = self.bn_initial(x)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        return x

class ImprovedClassifier(torch.nn.Module):
    """Improved Classifier with BatchNorm, ReLU, and Dropout"""
    def __init__(self, in_features, hidden_features, num_classes, dropout_prob=0.5):
        super(ImprovedClassifier, self).__init__()
        self.mlp = Sequential(
            Linear(in_features, hidden_features),
            BatchNorm1d(hidden_features),
            ReLU(),
            Dropout(dropout_prob),
            Linear(hidden_features, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class RandomIntervention(torch.nn.Module):
    """Random Intervention Module (Concatenation)"""
    def __init__(self):
        super(RandomIntervention, self).__init__()

    def forward(self, context_output, object_output, eval_random):
        num = context_output.shape[0]
        l = list(range(num))
        if eval_random:
            import random
            random.shuffle(l)
        random_idx = torch.tensor(l)
        # Currently using concatenation - experiment with addition as in online CausalGAT
        combined_output = torch.cat((context_output[random_idx], object_output), dim=1)
        return combined_output

class CAL_GAT(torch.nn.Module):
    """Graph Attention Network with Causal Attention Learning (Improved v2)"""
    def __init__(self, num_in_features, num_hidden_features, num_out_features, num_of_heads=8,
                 dropout_prob=0.6, alpha=0.2):
        super(CAL_GAT, self).__init__()

        # 1. GNN Encoder (ImprovedGNNEncoder_v2)
        self.gnn_encoder = ImprovedGNNEncoder_v2(
            num_in_features=num_in_features,
            num_hidden_features=num_hidden_features,
            num_of_heads=num_of_heads,
            dropout_prob=dropout_prob,
            alpha=alpha
        )

        hidden_dim = num_hidden_features * num_of_heads

        # 2. Node Attention
        self.node_attention = NodeAttention(hidden_dim, dropout_prob)

        # 3. Edge Attention
        self.edge_attention = EdgeAttention(hidden_dim, dropout_prob)

        # 4. Graph Convolution Layers
        self.graph_conv_causal = GCNConv(hidden_dim, hidden_dim) # Causal branch uses GCNConv
        self.graph_conv_trivial = GCNConv(hidden_dim, hidden_dim) # Trivial branch uses GCNConv

        # 5. Readout Functions (Global Mean Pooling)
        self.readout_causal = global_mean_pool
        self.readout_trivial = global_mean_pool

        # 6. Classifiers (ImprovedClassifier)
        classifier_hidden_dim = num_hidden_features
        classifier_dropout_prob = dropout_prob

        self.classifier_causal = ImprovedClassifier(
            in_features=hidden_dim,
            hidden_features=classifier_hidden_dim,
            num_classes=num_out_features,
            dropout_prob=classifier_dropout_prob
        )
        self.classifier_trivial = ImprovedClassifier(
            in_features=hidden_dim,
            hidden_features=classifier_hidden_dim,
            num_classes=num_out_features,
            dropout_prob=classifier_dropout_prob
        )
        self.classifier_combined = ImprovedClassifier(
            in_features=hidden_dim * 2, # Combined classifier takes concatenated features
            hidden_features=classifier_hidden_dim,
            num_classes=num_out_features,
            dropout_prob=classifier_dropout_prob
        )

        # 7. Random Intervention Module
        self.random_intervention = RandomIntervention()

        self.num_classes = num_out_features # Store num_classes in model for loss calculation

    def forward(self, data, eval_random=True, c_weight=0.5, o_weight=1.0, co_weight=1.0):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. GNN Encoder
        node_embeddings = self.gnn_encoder(x, edge_index)

        # 2. Node and Edge Attention
        node_attention_scores = self.node_attention(node_embeddings)
        edge_attention_scores = self.edge_attention(node_embeddings, edge_index)

        # 3. Graph Convolution for Causal and Trivial branches
        causal_embeddings = self.graph_conv_causal(node_embeddings, edge_index, edge_weight=edge_attention_scores[:,0]) # Causal uses edge attention (context)
        trivial_embeddings = self.graph_conv_trivial(node_embeddings, edge_index, edge_weight=edge_attention_scores[:, 1]) # Trivial uses edge attention (object)

        # 4. Readout Functions (Global Mean Pooling)
        pooled_output_causal = self.readout_causal(causal_embeddings, batch)
        pooled_output_trivial = self.readout_trivial(trivial_embeddings, batch)

        # 5. Random Intervention
        combined_output = self.random_intervention(pooled_output_causal, pooled_output_trivial, eval_random)

        # 6. Classifiers
        output_causal = F.log_softmax(self.classifier_causal(pooled_output_causal), dim=-1) # LogSoftmax for NLLLoss
        output_trivial = F.log_softmax(self.classifier_trivial(pooled_output_trivial), dim=-1) # LogSoftmax for NLLLoss
        output_combined = F.log_softmax(self.classifier_combined(combined_output), dim=-1) # LogSoftmax for NLLLoss

        return output_causal, output_trivial, output_combined

# --- 2. Loss Function ---

def compute_losses(output_causal, output_trivial, output_combined, labels, num_classes, c_weight=1.0, o_weight=1.0, co_weight=1.0):
    """Computes the uniform, causal, and combined losses."""
    device = output_causal.device
    labels = labels.to(device)

    # 1. Supervised Loss (Trivial branch - NLLLoss)
    L_sup = F.nll_loss(output_trivial, labels)

    # 2. Uniform Loss (Causal branch - KLDivLoss)
    uniform_target = torch.ones_like(output_causal, dtype=torch.float).to(device) / num_classes
    L_unif = F.kl_div(output_causal, uniform_target, reduction='batchmean')

    # 3. Causal Intervention Loss (Combined branch - NLLLoss)
    L_caus = F.nll_loss(output_combined, labels)

    # Total Loss with weights as arguments
    total_loss = c_weight * L_unif + o_weight * L_sup + co_weight * L_caus

    return total_loss, L_unif, L_sup, L_caus

# --- 3. Data Preparation Function ---

def prepare_graph_data(adjacency_matrix, covariance_features):
    """Convert adjacency matrix and use covariance features for nodes."""
    edges = np.where(adjacency_matrix > 0)
    edge_index = torch.tensor(np.vstack((edges[0], edges[1])), dtype=torch.long)
    edge_attr = torch.tensor(adjacency_matrix[edges], dtype=torch.float)

    # --- Debugging in prepare_graph_data ---
    # print("\n--- Debugging inside prepare_graph_data ---")
    # print("Debug (prepare_graph_data): Type of covariance_features:", type(covariance_features))
    # if covariance_features is not None:
    #     # print("Debug (prepare_graph_data): Length of covariance_features:", len(covariance_features))
    #     if len(covariance_features) > 0:
    #         # print("Debug (prepare_graph_data): Type of first element in covariance_features:", type(covariance_features[0]))
    #         if isinstance(covariance_features[0], np.ndarray):
    #             # print("Debug (prepare_graph_data): Shape of first element in covariance_features:", covariance_features[0].shape)
    #             # print("Debug (prepare_graph_data): dtype of first element in covariance_features:", covariance_features[0].dtype)
    #             # print("Debug (prepare_graph_data): Sample first element of covariance_features:", covariance_features[0][:5])
    #         else:
    #             print("Debug (prepare_graph_data): First element is NOT a numpy array.")
    #     else:
    #         print("Debug (prepare_graph_data): covariance_features is empty list/array.")
    # else:
    #     print("Debug (prepare_graph_data): covariance_features is None.")


    # --- Use Covariance Features as Node Features ---
    # node_features = np.array(covariance_features, dtype=np.float32) # OLD - Causing error
    node_features = np.vstack(covariance_features).astype(np.float32) # NEW - Vertically stack to create 2D array

    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print("Data object __dict__ in prepare_graph_data with covariance features:", data.__dict__) # Debug print
    return data

# --- 4. Data Loading Function ---

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
    labels = np.loadtxt(labels_path)
    graph_index = 0 # Start index for graph files
    while True: # Try loading files until no more are found
        adj_matrix_file = os.path.join(data_path, f'adj_matrix_graph_{graph_index}.npy')
        covar_features_file = os.path.join(data_path, f'covar_features_graph_{graph_index}.npy')
        if not os.path.exists(adj_matrix_file) or not os.path.exists(covar_features_file):
            break # Stop loading if files for the current graph index are not found
        adj_matrix = np.loadtxt(adj_matrix_file)
        covar_features_matrix = np.loadtxt(covar_features_file)

        # print(f"Debug (load_local_data): Shape of covar_features_matrix for graph {graph_index}:", covar_features_matrix.shape)
        # print(f"Debug (load_local_data): dtype of covar_features_matrix for graph {graph_index}:", covar_features_matrix.dtype)
        # print(f"Debug (load_local_data): Sample of covar_features_matrix for graph {graph_index}:\n", covar_features_matrix[:2,:2] if covar_features_matrix.size > 0 else "Matrix is empty") 

        covar_features = matrix_to_array_of_vectors(covar_features_matrix)

        # print(f"Debug (load_local_data): Type of covar_features for graph {graph_index}:", type(covar_features))
        # print(f"Debug (load_local_data): Shape of covar_features for graph {graph_index}:", covar_features.shape)
        # if covar_features.size > 0:
        #     print(f"Debug (load_local_data): Sample element from covar_features for graph {graph_index} (first element):", covar_features[0]) 
        #     if isinstance(covar_features[0], np.ndarray):
        #         print(f"Debug (load_local_data): Shape of the first element in covar_features for graph {graph_index}:", covar_features[0].shape)
        #         print(f"Debug (load_local_data): dtype of the first element in covar_features for graph {graph_index}:", covar_features[0].dtype)
        # else:
        #     print("Debug (load_local_data): covar_features is empty.")

        covariance_features_list.append(covar_features)
        adjacency_matrices.append(adj_matrix)
        graph_index += 1 # Increment for next graph
    if not adjacency_matrices: # Check if any data was loaded
        raise FileNotFoundError(f"No adjacency or covariance feature files found in: {data_path}. "
                                f"Make sure files are named as 'adj_matrix_graph_i.npy' and "
                                f"'covar_features_graph_i.npy' and 'labels.npy' are in the directory.")
    return adjacency_matrices, covariance_features_list, labels

# --- 5. Training and Evaluation Functions ---

def train_cal_gat(model, train_loader, optimizer, device, num_classes, c_weight=1.0, o_weight=1.0, co_weight=1.0):
    """Trains the CAL_GAT model for one epoch."""
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        # Forward pass with loss weights
        output_causal, output_trivial, output_combined = model(
            data, eval_random=True, c_weight=c_weight, o_weight=o_weight, co_weight=co_weight
        )

        # Compute losses with weights
        loss, loss_unif, loss_sup, loss_caus = compute_losses(
            output_causal, output_trivial, output_combined, data.y, num_classes,
            c_weight=c_weight, o_weight=o_weight, co_weight=co_weight
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_cal_gat(model, loader, device, num_classes, c_weight=1.0, o_weight=1.0, co_weight=1.0, eval_random=False):
    """Evaluates the CAL_GAT model on a given dataloader."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output_causal, output_trivial, output_combined = model(
                data, eval_random=eval_random, c_weight=c_weight, o_weight=o_weight, co_weight=co_weight
            )
            loss, _, _, _ = compute_losses(
                output_causal, output_trivial, output_combined, data.y, num_classes,
                c_weight=c_weight, o_weight=o_weight, co_weight=co_weight
            )
            total_loss += loss.item()
            predicted_labels = output_combined.argmax(dim=1)
            correct_predictions += (predicted_labels == data.y).sum().item()
            all_labels.extend(data.y.cpu().numpy())
            all_predicted.extend(predicted_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)
    conf_matrix = confusion_matrix(all_labels, all_predicted)
    return avg_loss, accuracy, conf_matrix, all_predicted

# --- 6. K-Fold Cross-Validation and Main Function ---

def run_cross_validation(model, o_weight, co_weight, num_folds, kf_splits, graph_data_list, batch_size, device, num_classes,
                         input_dim, hidden_dim, heads, dropout, alpha, learning_rate, epochs, fold_num, patience=10): # Added patience parameter
    """
    Executes a single fold of cross-validation for CAL_GAT model with early stopping.

    Args:
        o_weight (float): Weight for orthogonality loss.
        co_weight (float): Weight for contrastive loss.
        num_folds (int): Total number of folds in cross-validation.
        kf_splits (list): Pre-calculated KFold splits.
        graph_data_list (list): List of PyTorch Geometric Data objects.
        batch_size (int): Batch size for DataLoader.
        device (torch.device): 'cuda' or 'cpu'.
        num_classes (int): Number of classes for classification.
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        alpha (float): Alpha for LeakyReLU in GAT.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Maximum number of training epochs.
        fold_num (int): Current fold number (1-indexed for printing).
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping.

    Returns:
        tuple: Final validation loss, final validation accuracy, confusion matrix, and predicted labels for the fold.
    """
    print(f"\n--- Fold {fold_num}/{num_folds} ---")
    train_index, val_index = kf_splits[fold_num-1][0], kf_splits[fold_num-1][1] # Adjust fold_num to be 0-indexed for list access
    train_dataset = [graph_data_list[i] for i in train_index]
    val_dataset = [graph_data_list[i] for i in val_index]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Optimizer and Scheduler ---
    print(f" Initializing optimizer and scheduler for Fold {fold_num}...")
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-10, last_epoch=-1, verbose=False)

    best_val_loss_fold = float('inf') # Initialize best validation loss to infinity for minimization
    best_val_acc_fold = -1.0
    best_model_state_fold = None
    epochs_no_improve = 0 # Counter for epochs with no validation loss improvement

    # --- Training and Validation Loop with Early Stopping ---
    print(f" Starting training and validation for Fold {fold_num} with patience={patience} epochs...")
    for epoch in range(epochs):
        # Train
        train_loss = train_cal_gat(model, train_loader, optimizer, device, num_classes, c_weight=1.0, o_weight=o_weight, co_weight=co_weight)

        # Validate
        val_loss, val_acc, conf_matrix_fold, predicted_labels_fold = evaluate_cal_gat(model, val_loader, device, num_classes, c_weight=1.0, o_weight=o_weight, co_weight=co_weight)

        # Check for validation loss improvement
        if val_loss < best_val_loss_fold:
            best_val_loss_fold = val_loss
            best_val_acc_fold = val_acc
            best_model_state_fold = model.state_dict()
            epochs_no_improve = 0 # Reset counter when validation loss improves
        else:
            epochs_no_improve += 1 # Increment counter if validation loss does not improve

        # Print epoch metrics for each fold
        print(f"  Epoch {epoch+1}/{epochs} - Fold {fold_num}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc_fold:.4f}, No Improvement Epochs: {epochs_no_improve}")


        scheduler.step() # Step the scheduler after each epoch

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"  Early stopping triggered: Validation loss did not improve for {patience} epochs.")
            break # Break out of the epoch loop

    # --- After Epochs (or early stopping) for Fold - Load Best Model & Evaluate ---
    print(f"  Fold {fold_num} Training finished (or stopped early). Loading best model from validation...")
    model.load_state_dict(best_model_state_fold) # Load best model state
    val_loss_final, val_acc_final, conf_matrix_final, predicted_labels_final = evaluate_cal_gat(model, val_loader, device, num_classes, c_weight=1.0, o_weight=o_weight, co_weight=co_weight)

    print(f"  Fold {fold_num} Final Validation Loss: {val_loss_final:.4f}, Final Validation Accuracy: {val_acc_final:.4f}")
    print(f"  Fold {fold_num} Confusion Matrix:\n{conf_matrix_final}")

    return val_loss_final, val_acc_final, conf_matrix_final, predicted_labels_final


def main():
    # --- Hyperparameters (Base Settings) ---
    num_classes = 2
    hidden_dim = 32
    heads = 8
    dropout = 0.8
    alpha = 0.2
    learning_rate = 0.0003
    epochs = 500
    batch_size = 32
    num_folds = 5  # For K-Fold CV
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = 'C:\\MyLab\\UIUC\\Research\\JNER\\preprocessed\\fcm'
    patience = 10 # Early stopping patience

    # --- Hyperparameter Grid for Tuning ---
    o_weight_values = [0.5]  # values to test for o_weight
    co_weight_values = [1.0] # values to test for co_weight

    print("PyTorch version:", torch.__version__)
    print("PyTorch Geometric version:", torch_geometric.__version__)
    print(f"Device: {device}")
    print("Using Covariance Node Features")

    # --- 1. Load and Prepare Data ---
    print(f"Loading data from: {data_path}")
    adjacency_matrices, covariance_features_list, labels = load_local_data(data_path)

    # --- Determine input_dim ---
    if covariance_features_list and covariance_features_list[0] is not None:
        first_graph_cov_features = covariance_features_list[0]
        if len(first_graph_cov_features) > 0:
            input_dim = len(first_graph_cov_features[0])
            print(f"Using Covariance Node Features. Input feature dimension inferred as: {input_dim}")
        else:
            input_dim = 1
            print("Warning: First graph has no node features. Falling back to input_dim=1.")
    else:
        input_dim = 1
        print("Warning: No covariance features loaded. Falling back to input_dim=1.")

    graph_data_list = []
    for adj_matrix, cov_features, label in zip(adjacency_matrices, covariance_features_list, labels):
        data = prepare_graph_data(adj_matrix, cov_features)
        data.y = torch.tensor(int(label), dtype=torch.long)
        graph_data_list.append(data)

    # --- 2. K-Fold Cross-Validation Setup ---
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    kf_splits = list(kf.split(graph_data_list, labels))

    best_avg_val_loss = float('inf') # Initialize with a very high value for loss minimization
    best_avg_val_acc = -1.0
    best_hyperparameters = None
    hyperparameter_tuning_results = {}

    # --- 3. Hyperparameter Grid Search with Cross-Validation ---
    print("\n--- Starting Hyperparameter Grid Search ---")
    for o_weight in o_weight_values:
        for co_weight in co_weight_values:
            print(f"\n--- Evaluating o_weight={o_weight}, co_weight={co_weight} ---")
            k_fold_val_losses = []
            k_fold_val_accuracies = []
            k_fold_confusion_matrices = []
            all_fold_predicted_labels = []

            # --- Initialize Model ---
            print(" Initializing CAL_GAT model...")
            model = CAL_GAT(
                num_in_features=input_dim,
                num_hidden_features=hidden_dim,
                num_out_features=num_classes,
                num_of_heads=heads,
                dropout_prob=dropout,
                alpha=alpha
            ).to(device)
            for i in range(5):
                for fold_num in range(1, num_folds + 1):
                    val_loss_final, val_acc_final, conf_matrix_final, predicted_labels_final = run_cross_validation(
                        model, o_weight, co_weight, num_folds, kf_splits, graph_data_list, batch_size, device, num_classes,
                        input_dim, hidden_dim, heads, dropout, alpha, learning_rate, epochs, fold_num, patience # Pass patience to run_cross_validation
                    )
                    k_fold_val_losses.append(val_loss_final)
                    k_fold_val_accuracies.append(val_acc_final)
                    k_fold_confusion_matrices.append(conf_matrix_final)
                    all_fold_predicted_labels.extend(predicted_labels_final)

                avg_val_accuracy = np.mean(k_fold_val_accuracies)
                std_dev_accuracy = np.std(k_fold_val_accuracies)
                avg_val_loss = np.mean(k_fold_val_losses)

                print(f"\n--- Hyperparameter Combination: o_weight={o_weight}, co_weight={co_weight} ---")
                print(f"Average Validation Accuracy: {avg_val_accuracy:.4f} ± {std_dev_accuracy:.4f}")
                print(f"Average Validation Loss: {avg_val_loss:.4f}")

                hyperparameter_tuning_results[(o_weight, co_weight)] = {
                    'fold_accuracies': k_fold_val_accuracies,
                    'average_accuracy': avg_val_accuracy,
                    'std_dev_accuracy': std_dev_accuracy,
                    'average_loss': avg_val_loss
                }

                if (avg_val_loss < best_avg_val_loss) or ((avg_val_loss == best_avg_val_loss) and (avg_val_accuracy > best_avg_val_acc)):
                    best_avg_val_loss = avg_val_loss
                    best_avg_val_acc = avg_val_accuracy
                    best_hyperparameters = {'o_weight': o_weight, 'co_weight': co_weight}

    # --- 4. Best Hyperparameter Results ---
    print("\n--- Hyperparameter Tuning Complete ---")
    print("Best Hyperparameters:")
    print(f"  o_weight: {best_hyperparameters['o_weight']}")
    print(f"  co_weight: {best_hyperparameters['co_weight']}")
    print(f"Best Average Validation Accuracy: {best_avg_val_acc:.4f}")

    print("\n--- All Hyperparameter Tuning Results ---")
    for hyperparams, results in hyperparameter_tuning_results.items():
        print(f"\nHyperparameters: o_weight={hyperparams[0]}, co_weight={hyperparams[1]}")
        print(f"  Average Validation Accuracy: {results['average_accuracy']:.4f} ± {results['std_dev_accuracy']:.4f}")
        print(f"  Average Validation Loss: {results['average_loss']:.4f}")
        print(f"  Fold Validation Accuracies: {results['fold_accuracies']}")


if __name__ == "__main__":
    main()