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

        # --- Soft Attention as Masking ---

        # 3. Node Feature Weighting using Node Attention
        xc = node_attention_scores[:, 0].view(-1, 1) * node_embeddings # Scale node features by causal node attention
        xo = node_attention_scores[:, 1].view(-1, 1) * node_embeddings # Scale node features by trivial node attention

        # 4. Graph Convolution with Edge Attention as Edge Weights
        causal_embeddings = self.graph_conv_causal(xc, edge_index, edge_weight=edge_attention_scores[:,0]) # Causal branch with causal edge attention
        trivial_embeddings = self.graph_conv_trivial(xo, edge_index, edge_weight=edge_attention_scores[:, 1]) # Trivial branch with trivial edge attention


        # 5. Readout Functions (Global Mean Pooling)
        pooled_output_causal = self.readout_causal(causal_embeddings, batch)
        pooled_output_trivial = self.readout_trivial(trivial_embeddings, batch)

        # 6. Random Intervention
        combined_output = self.random_intervention(pooled_output_causal, pooled_output_trivial, eval_random)

        # 7. Classifiers
        output_causal = F.log_softmax(self.classifier_causal(pooled_output_causal), dim=-1) # LogSoftmax for NLLLoss
        output_trivial = F.log_softmax(self.classifier_trivial(pooled_output_trivial), dim=-1) # LogSoftmax for NLLLoss
        output_combined = F.log_softmax(self.classifier_combined(combined_output), dim=-1) # LogSoftmax for NLLLoss

        return output_causal, output_trivial, output_combined

# --- 2. Loss Function ---

def compute_losses(output_causal, output_trivial, output_combined, labels, num_classes, c_weight=1.0, o_weight=1.0):
    """
    Computes the combined loss, including Uniform Loss for causal branch and NLL Loss for trivial and combined branches.

    Args:
        output_causal (torch.Tensor): Output logits from the causal branch.
        output_trivial (torch.Tensor): Output logits from the trivial branch.
        output_combined (torch.Tensor): Output logits from the combined branch.
        labels (torch.LongTensor): Ground truth labels.
        num_classes (int): Number of classes.
        c_weight (float): Weight for Uniform Loss.
        o_weight (float): Weight for Supervised Loss (trivial branch).

    Returns:
        tuple: Total loss, Uniform Loss, Supervised Loss (trivial), Supervised Loss (combined), and total Supervised Loss.
    """
    device = output_causal.device
    one_hot_target = labels.view(-1)
    uniform_target = torch.ones_like(output_causal, dtype=torch.float).to(device) / num_classes

    # 1. Uniform Loss (KL Divergence) for Causal Branch
    loss_unif = F.kl_div(output_causal, uniform_target, reduction='batchmean')

    # 2. Supervised Loss (NLLLoss) for Trivial Branch
    loss_sup_o = F.nll_loss(output_trivial, one_hot_target)

    # 3. Supervised Loss (NLLLoss) for Combined Branch -  L_sup (weight implicitly 1)
    loss_sup_co = F.nll_loss(output_combined, one_hot_target)

    # 4. Total Loss (Adhering to official equation in CAL paper: L_sup + lambda_1 * L_unif + lambda_2 * L_caus)
    loss = loss_sup_co + c_weight * loss_unif + o_weight * loss_sup_o

    return loss, loss_unif, loss_sup_o, loss_sup_co

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

def train_cal_gat(model, train_loader, optimizer, device, num_classes, c_weight=1.0, o_weight=1.0, eval_random=False): # eval_random as parameter
    """Trains the CAL_GAT model for one epoch."""
    model.train()
    total_loss = 0
    total_loss_unif = 0 # Track Uniform Loss
    total_loss_sup_o = 0 # Track Supervised Loss (trivial)
    total_loss_sup_co = 0 # Track Supervised Loss (combined)
    correct_combined = 0  # Track combined accuracy (optional, but good to monitor)

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        # Forward pass with loss weights and eval_random passed as argument
        output_causal, output_trivial, output_combined = model(
            data, eval_random=eval_random, c_weight=c_weight, o_weight=o_weight,
        )

        # Compute losses using updated compute_losses function
        loss, loss_unif, loss_sup_o, loss_sup_co = compute_losses(
            output_causal, output_trivial, output_combined, data.y, num_classes,
            c_weight=c_weight, o_weight=o_weight
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss_unif += loss_unif.item() # Accumulate Uniform Loss
        total_loss_sup_o += loss_sup_o.item() # Accumulate Supervised Loss (trivial)
        total_loss_sup_co += loss_sup_co.item() # Accumulate Supervised Loss (combined)

        # Optional: Calculate training accuracy for combined output (can be removed if not needed)
        predicted_labels_combined = output_combined.argmax(dim=1)
        correct_combined += (predicted_labels_combined == data.y).sum().item()


    avg_loss = total_loss / len(train_loader)
    avg_loss_unif = total_loss_unif / len(train_loader)
    avg_loss_sup_o = total_loss_sup_o / len(train_loader)
    avg_loss_sup_co = total_loss_sup_co / len(train_loader)
    train_accuracy_combined = correct_combined / len(train_loader.dataset) # Optional accuracy

    return avg_loss, avg_loss_unif, avg_loss_sup_o, avg_loss_sup_co, train_accuracy_combined # Return all losses and accuracy

def evaluate_cal_gat(model, loader, device, num_classes, c_weight=1.0, o_weight=1.0, eval_random=False):
    """Evaluates the CAL_GAT model on a given dataloader."""
    model.eval()
    total_loss = 0.0
    correct_predictions_combined = 0 # Correct predictions for combined output
    correct_predictions_causal = 0   # Correct predictions for causal output
    correct_predictions_trivial = 0  # Correct predictions for trivial output
    all_labels = []
    all_predicted_combined = [] # Predicted labels for combined output

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output_causal, output_trivial, output_combined = model(
                data, eval_random=eval_random, c_weight=c_weight, o_weight=o_weight
            )
            loss, _, _, _ = compute_losses( # Loss calculation still needed for evaluation metrics
                output_causal, output_trivial, output_combined, data.y, num_classes,
                c_weight=c_weight, o_weight=o_weight
            )
            total_loss += loss.item()

            # Predictions for each branch
            predicted_labels_combined = output_combined.argmax(dim=1)
            predicted_labels_causal = output_causal.argmax(dim=1)
            predicted_labels_trivial = output_trivial.argmax(dim=1)

            # Count correct predictions for each branch
            correct_predictions_combined += (predicted_labels_combined == data.y).sum().item()
            correct_predictions_causal += (predicted_labels_causal == data.y).sum().item()
            correct_predictions_trivial += (predicted_labels_trivial == data.y).sum().item()

            all_labels.extend(data.y.cpu().numpy())
            all_predicted_combined.extend(predicted_labels_combined.cpu().numpy()) # Store combined predictions

    avg_loss = total_loss / len(loader)
    accuracy_combined = correct_predictions_combined / len(loader.dataset) # Accuracy for combined
    accuracy_causal = correct_predictions_causal / len(loader.dataset)     # Accuracy for causal branch
    accuracy_trivial = correct_predictions_trivial / len(loader.dataset)   # Accuracy for trivial branch
    conf_matrix = confusion_matrix(all_labels, all_predicted_combined) # Confusion matrix for combined predictions

    return avg_loss, accuracy_combined, accuracy_causal, accuracy_trivial, conf_matrix, all_predicted_combined

# --- 6. K-Fold Cross-Validation and Main Function ---

def run_cross_validation(model, o_weight, c_weight, num_folds, kf_splits, train_val_graph_data_list, batch_size, device, num_classes,
                         input_dim, hidden_dim, heads, dropout, alpha, learning_rate, epochs, fold_num, patience=10, eval_random_training=False, eval_random_validation=False): # Added eval_random parameters
    """
    Executes a single fold of cross-validation for CAL_GAT model.
    Used for training and validation ONLY (not testing).
    Model is expected to be initialized in the outer scope.

    Args:
        ... (rest of the arguments are the same as before)
        eval_random_training (bool): Whether to use random intervention during training.
        eval_random_validation (bool): Whether to use random intervention during validation.

    Returns:
        tuple: Final validation loss, final validation accuracy, confusion matrix, and predicted labels for the fold.
    """
    print(f"\n--- Fold {fold_num}/{num_folds} ---")
    train_index, val_index = kf_splits[fold_num-1][0], kf_splits[fold_num-1][1] # Adjust fold_num to be 0-indexed for list access
    train_dataset = [train_val_graph_data_list[i] for i in train_index] # Use train_val_graph_data_list
    val_dataset = [train_val_graph_data_list[i] for i in val_index]   # Use train_val_graph_data_list

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Optimizer and Scheduler ---
    print(f" Initializing optimizer and scheduler for Fold {fold_num}...")
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8, last_epoch=-1, verbose=False)

    best_val_loss_fold = float('inf') # Initialize best validation loss to infinity for minimization
    best_val_acc_fold = -1.0
    best_model_state_fold = None
    epochs_no_improve = 0 # Counter for epochs with no validation loss improvement

    # --- Training and Validation Loop with Early Stopping ---
    print(f" Starting training and validation for Fold {fold_num} with patience={patience} epochs...")
    for epoch in range(epochs):
        # Train
        train_loss, train_loss_unif, train_loss_sup_o, train_loss_sup_co, train_acc_combined = train_cal_gat( # Get all loss components and combined train accuracy
            model, train_loader, optimizer, device, num_classes, c_weight=1.0, o_weight=o_weight, eval_random=eval_random_training # Pass eval_random_training
        )

        # Validate
        val_loss, val_acc_combined, val_acc_causal, val_acc_trivial, conf_matrix_fold, predicted_labels_fold = evaluate_cal_gat( # Get branch accuracies
            model, val_loader, device, num_classes, c_weight=1.0, o_weight=o_weight, eval_random=eval_random_validation # Pass eval_random_validation
        )

        # Check for validation loss improvement
        if val_loss <= best_val_loss_fold:
            best_val_loss_fold = val_loss
            best_val_acc_fold = val_acc_combined # Best val acc is still combined accuracy
            best_model_state_fold = model.state_dict()
            epochs_no_improve = 0 # Reset counter when validation loss improves
        else:
            epochs_no_improve += 1 # Increment counter if validation loss does not improve

        # Print epoch metrics for each fold - UPDATED PRINT STATEMENT
        print(f"  Epoch {epoch+1}/{epochs} - Fold {fold_num}: "
              f"Train Loss: {train_loss:.4f} (U:{train_loss_unif:.4f}, O:{train_loss_sup_o:.4f}, CO:{train_loss_sup_co:.4f}), " # Print loss components
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc (Combined): {val_acc_combined:.4f}, " # Combined validation accuracy
              f"Val Acc (Causal): {val_acc_causal:.4f}, Val Acc (Trivial): {val_acc_trivial:.4f}, " # Branch validation accuracies
              f"Best Val Acc (Combined): {best_val_acc_fold:.4f}, "
              f"No Improvement Epochs: {epochs_no_improve}"
             )


        scheduler.step() # Step the scheduler after each epoch

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"  Early stopping triggered: Validation loss did not improve for {patience} epochs.")
            break # Break out of the epoch loop

    # --- After Epochs (or early stopping) for Fold - Load Best Model & Evaluate ---
    print(f"  Fold {fold_num} Training finished (or stopped early). Loading best model from validation...")
    model.load_state_dict(best_model_state_fold) # Load best model state
    val_loss_final, val_acc_final, val_acc_causal_final, val_acc_trivial_final, conf_matrix_final, predicted_labels_final = evaluate_cal_gat( # Get branch accuracies in final eval
        model, val_loader, device, num_classes, c_weight=1.0, o_weight=o_weight, eval_random=eval_random_validation # Pass eval_random_validation
    )

    print(f"  Fold {fold_num} Final Validation Loss: {val_loss_final:.4f}, Final Validation Accuracy (Combined): {val_acc_final:.4f}") # Combined accuracy
    print(f"  Fold {fold_num} Final Validation Accuracy (Causal): {val_acc_causal_final:.4f}, Final Validation Accuracy (Trivial): {val_acc_trivial_final:.4f}") # Branch accuracies
    print(f"  Fold {fold_num} Confusion Matrix:\n{conf_matrix_final}")

    return val_loss_final, val_acc_final, conf_matrix_final, predicted_labels_final


def main():
    # --- Hyperparameters (Base Settings) ---
    num_classes = 2
    hidden_dim = 64
    heads = 8
    dropout = 0.7
    alpha = 0.2
    learning_rate = 0.00005
    epochs = 150
    batch_size = 32
    num_folds = 4  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = 'C:\\MyLab\\UIUC\\Research\\JNER\\preprocessed\\fcm'
    patience = 10 # Early stopping patience
    num_test_samples = 12 # Number of samples for test set
    num_cv_runs = 8 # Number of times to repeat cross-validation

    # --- Hyperparameter Grid for Tuning ---
    c_weight_values = [0.5]  # values to test for c_weight
    o_weight_values = [0.5] # values to test for o_weight
    
    test_label_1_count = 10 # Desired count of label 1 in test set
    test_label_0_count = 2 # Desired count of label 0 in test set
    eval_random_training_values = [True, False] # test with and without random intervention during training
    eval_random_validation_values = [False]   # No random intervention during validation (consistent with typical eval)

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

    # --- 2. Create Test Set with Specified Label Distribution ---
    label_1_indices = [i for i, label in enumerate(labels) if label == 1]
    label_0_indices = [i for i, label in enumerate(labels) if label == 0]

    test_indices_label_1 = label_1_indices[:test_label_1_count] # Take first 'test_label_1_count' indices with label 1
    test_indices_label_0 = label_0_indices[:test_label_0_count] # Take first 'test_label_0_count' indices with label 0

    test_indices = sorted(test_indices_label_1 + test_indices_label_0) # Combine and sort test indices
    test_dataset = [graph_data_list[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    train_val_indices = [i for i in range(len(graph_data_list)) if i not in test_indices] # Indices for train/val set
    train_val_graph_data_list = [graph_data_list[i] for i in train_val_indices]
    train_val_labels = [labels[i] for i in train_val_indices]

    # test_dataset = graph_data_list[-num_test_samples:] # Take last num_test_samples for test set # OLD - Removed
    # test_labels = labels[-num_test_samples:] # OLD - Removed
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # No shuffle for test loader
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long) # Convert test labels to tensor

    print(f"Number of training/validation samples: {len(train_val_graph_data_list)}")
    print(f"Number of test samples: {len(test_dataset)} (Label 1: {test_label_1_count}, Label 0: {test_label_0_count})") # Print test set label distribution

    # --- 3. K-Fold Cross-Validation Setup (on train_val_data) ---
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    # kf_splits will now be generated inside the CV loop to allow for different random states if needed in future
    # kf_splits = list(kf.split(train_val_graph_data_list, train_val_labels)) # Use train_val_labels - moved inside CV loop

    best_avg_val_loss = float('inf') # Initialize with a very high value for loss minimization
    best_avg_val_acc = -1.0
    best_hyperparameters = None
    hyperparameter_tuning_results = {}

    # --- 4. Hyperparameter Grid Search with Cross-Validation ---
    print("\n--- Starting Hyperparameter Grid Search ---")
    for c_weight in c_weight_values:
        for o_weight in o_weight_values:
            for eval_random_training in eval_random_training_values: # Loop over eval_random_training
                for eval_random_validation in eval_random_validation_values: # Loop over eval_random_validation

                    print(f"\n--- Evaluating o_weight={o_weight}, eval_random_training={eval_random_training}, eval_random_validation={eval_random_validation} ---") # Print eval_random settings
                    cv_run_val_accuracies = [] # Store validation accuracies for each CV run
                    cv_run_val_losses = []
                    cv_run_test_accuracies = []
                    cv_run_test_losses = []
                    cv_run_val_acc_causal_branches = [] # Store val acc for causal branch
                    cv_run_val_acc_trivial_branches = [] # Store val acc for trivial branch


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


                    for cv_run in range(1, num_cv_runs + 1): # Loop for CV repetitions
                        print(f"\n--- CV Run {cv_run}/{num_cv_runs} ---")
                        k_fold_val_losses = []
                        k_fold_val_accuracies = []
                        k_fold_test_losses = []
                        k_fold_test_accuracies = []
                        k_fold_val_acc_causal_folds = [] # Store val acc causal per fold
                        k_fold_val_acc_trivial_folds = [] # Store val acc trivial per fold
                        k_fold_confusion_matrices = []
                        all_fold_predicted_labels = []

                        # Regenerate k-fold splits for each CV run
                        kf_splits = list(kf.split(train_val_graph_data_list, train_val_labels))

                        for fold_num in range(1, num_folds + 1):
                            val_loss_final, val_acc_final, conf_matrix_final, predicted_labels_final = run_cross_validation(
                                model,
                                o_weight, c_weight, num_folds, kf_splits, train_val_graph_data_list, batch_size, device, num_classes,
                                input_dim, hidden_dim, heads, dropout, alpha, learning_rate, epochs, fold_num, patience, # Pass patience
                                eval_random_training, eval_random_validation # Pass eval_random settings
                            )
                            k_fold_val_losses.append(val_loss_final)
                            k_fold_val_accuracies.append(val_acc_final)
                            k_fold_confusion_matrices.append(conf_matrix_final)
                            all_fold_predicted_labels.extend(predicted_labels_final)

                            # Evaluate on test set AFTER each fold in each CV run
                            print("\n Evaluating on test set after Fold {} of CV run {}...".format(fold_num, cv_run))
                            current_test_loss, current_test_acc, current_test_acc_causal, current_test_acc_trivial, current_test_conf_matrix, current_test_predicted_labels = evaluate_on_test_set( # Get branch test accuracies
                                model, learning_rate, o_weight, c_weight, test_loader, device, num_classes, input_dim, hidden_dim, heads, dropout, alpha, test_labels_tensor,
                                eval_random_test=eval_random_validation # Use same eval_random for test as validation for consistency
                            )
                            k_fold_test_losses.append(current_test_loss) # Record test loss for this fold
                            k_fold_test_accuracies.append(current_test_acc) # Record test accuracy for this fold
                            k_fold_val_acc_causal_folds.append(current_test_acc_causal) # Record test acc causal
                            k_fold_val_acc_trivial_folds.append(current_test_acc_trivial) # Record test acc trivial
                            print(f"  Test Set Loss (Fold {fold_num}): {current_test_loss:.4f}")
                            print(f"  Test Set Accuracy (Combined, Fold {fold_num}): {current_test_acc:.4f}, Test Set Accuracy (Causal, Fold {fold_num}): {current_test_acc_causal:.4f}, Test Set Accuracy (Trivial, Fold {fold_num}): {current_test_acc_trivial:.4f}") # Print branch test accuracies


                        avg_val_accuracy = np.mean(k_fold_val_accuracies)
                        std_dev_accuracy = np.std(k_fold_val_accuracies)
                        avg_val_loss = np.mean(k_fold_val_losses)
                        avg_test_accuracy_cv_run = np.mean(k_fold_test_accuracies) # Avg test acc for this CV run
                        avg_test_loss_cv_run = np.mean(k_fold_test_losses) # Avg test loss for this CV run
                        avg_val_acc_causal_cv_run = np.mean(k_fold_val_acc_causal_folds) # Avg val acc causal for CV run
                        avg_val_acc_trivial_cv_run = np.mean(k_fold_val_acc_trivial_folds) # Avg val acc trivial for CV run


                        cv_run_val_accuracies.append(avg_val_accuracy) # Store avg val acc for this CV run
                        cv_run_val_losses.append(avg_val_loss) # Store avg val loss for this CV run
                        cv_run_test_accuracies.append(avg_test_accuracy_cv_run) # Store avg test acc for this CV run
                        cv_run_test_losses.append(avg_test_loss_cv_run) # Store avg test loss for this CV run
                        cv_run_val_acc_causal_branches.append(avg_val_acc_causal_cv_run) # Store avg val acc causal
                        cv_run_val_acc_trivial_branches.append(avg_val_acc_trivial_cv_run) # Store avg val acc trivial


                    avg_val_accuracy_hparam = np.mean(cv_run_val_accuracies) # Avg val acc across CV runs for this hyperparam set
                    std_dev_accuracy_hparam = np.std(cv_run_val_accuracies) # Std dev val acc across CV runs
                    avg_val_loss_hparam = np.mean(cv_run_val_losses) # Avg val loss across CV runs
                    avg_test_accuracy_hparam = np.mean(cv_run_test_accuracies) # Avg test acc across CV runs
                    avg_test_loss_hparam = np.mean(cv_run_test_losses) # Avg test loss across CV runs
                    avg_val_acc_causal_hparam = np.mean(cv_run_val_acc_causal_branches) # Avg val acc causal across CV runs
                    avg_val_acc_trivial_hparam = np.mean(cv_run_val_acc_trivial_branches) # Avg val acc trivial across CV runs


                    print(f"\n--- Hyperparameter Combination: o_weight={o_weight}, eval_random_training={eval_random_training}, eval_random_validation={eval_random_validation} ---") # Print eval_random settings
                    print(f"Average Validation Accuracy (Combined, CV averaged over {num_cv_runs} runs): {avg_val_accuracy_hparam:.4f} ± {std_dev_accuracy_hparam:.4f}") # Averaged CV accuracy
                    print(f"Average Validation Loss (CV, averaged over {num_cv_runs} runs): {avg_val_loss_hparam:.4f}") # Averaged CV loss
                    print(f"Average Validation Accuracy (Causal Branch, CV averaged over {num_cv_runs} runs): {avg_val_acc_causal_hparam:.4f}") # Avg val acc causal
                    print(f"Average Validation Accuracy (Trivial Branch, CV averaged over {num_cv_runs} runs): {avg_val_acc_trivial_hparam:.4f}") # Avg val acc trivial
                    print(f"Average Test Set Accuracy (Combined, averaged over {num_cv_runs} CV runs, per-fold test eval): {avg_test_accuracy_hparam:.4f}") # Avg Test Accuracy
                    print(f"Average Test Set Loss (averaged over {num_cv_runs} CV runs, per-fold test eval): {avg_test_loss_hparam:.4f}") # Avg Test Loss


                    hyperparameter_tuning_results[(o_weight, c_weight, eval_random_training, eval_random_validation)] = { # Include eval_random settings in hyperparameter key
                        'cv_run_val_accuracies': cv_run_val_accuracies,
                        'average_accuracy': avg_val_accuracy_hparam,
                        'std_dev_accuracy': std_dev_accuracy_hparam,
                        'average_loss': avg_val_loss_hparam,
                        'cv_run_test_accuracies': cv_run_test_accuracies,
                        'average_test_accuracy': avg_test_accuracy_hparam,
                        'average_test_loss': avg_test_loss_hparam,
                        'average_val_acc_causal': avg_val_acc_causal_hparam, # Store avg val acc causal
                        'average_val_acc_trivial': avg_val_acc_trivial_hparam # Store avg val acc trivial
                    }

                    # Best hyperparameter selection still based on combined validation loss/accuracy
                    if (avg_val_loss_hparam < best_avg_val_loss) or ((avg_val_loss_hparam == best_avg_val_loss) and (avg_val_accuracy_hparam > best_avg_val_acc)):
                        best_avg_val_loss = avg_val_loss_hparam
                        best_avg_val_acc = avg_val_accuracy_hparam
                        best_hyperparameters = {'o_weight': o_weight, 'eval_random_training': eval_random_training, 'eval_random_validation': eval_random_validation, # Store eval_random settings
                                                'average_test_accuracy': avg_test_accuracy_hparam,
                                                'average_test_loss': avg_test_loss_hparam,
                                                'average_val_acc_causal': avg_val_acc_causal_hparam, # Store best avg val acc causal
                                                'average_val_acc_trivial': avg_val_acc_trivial_hparam # Store best avg val acc trivial
                                                }

    # --- 5. Best Hyperparameter Results (from CV) ---
    print("\n--- Hyperparameter Tuning Complete (Cross-Validation repeated {} times) ---".format(num_cv_runs))
    print("Best Hyperparameters (based on Validation Accuracy during CV):")
    print(f"  o_weight: {best_hyperparameters['o_weight']}")
    print(f"  c_weight: {best_hyperparameters['c_weight']}")
    print(f"  eval_random_training: {best_hyperparameters['eval_random_training']}") # Print best eval_random settings
    print(f"  eval_random_validation: {best_hyperparameters['eval_random_validation']}") # Print best eval_random settings
    print(f"Best Average Validation Accuracy (Combined, CV averaged over {num_cv_runs} runs): {best_avg_val_acc:.4f}".format(num_cv_runs))
    print(f"Best Average Validation Loss (CV, averaged over {num_cv_runs} runs): {best_avg_val_loss:.4f}".format(num_cv_runs))
    print(f"Average Test Set Accuracy (Combined, for best hyperparameters, averaged over {num_cv_runs} CV runs, per-fold test eval): {best_hyperparameters['average_test_accuracy']:.4f}".format(num_cv_runs)) # Print best avg test accuracy
    print(f"Average Test Set Loss (Combined, for best hyperparameters, averaged over {num_cv_runs} CV runs, per-fold test eval): {best_hyperparameters['average_test_loss']:.4f}".format(num_cv_runs)) # Print best avg test loss
    print(f"Average Validation Accuracy (Causal Branch, for best hyperparameters, CV averaged over {num_cv_runs} runs): {best_hyperparameters['average_val_acc_causal']:.4f}".format(num_cv_runs)) # Print best avg val acc causal
    print(f"Average Validation Accuracy (Trivial Branch, for best hyperparameters, CV averaged over {num_cv_runs} runs): {best_hyperparameters['average_val_acc_trivial']:.4f}".format(num_cv_runs)) # Print best avg val acc trivial


    print("\n--- All Hyperparameter Tuning (CV) Results ---")
    for hyperparams, results in hyperparameter_tuning_results.items():
        print(f"\nHyperparameters: o_weight={hyperparams[0]}, c_weight={hyperparams[1]}, eval_random_training={hyperparams[2]}, eval_random_validation={hyperparams[3]}") # Print eval_random settings
        print(f"  Average Validation Accuracy (Combined, CV averaged over {num_cv_runs} runs): {results['average_accuracy']:.4f} ± {results['std_dev_accuracy']:.4f}".format(num_cv_runs)) # Averaged CV accuracy
        print(f"  Average Validation Loss (CV, averaged over {num_cv_runs} runs): {results['average_loss']:.4f}".format(num_cv_runs)) # Averaged CV loss
        print(f"  Fold Validation Accuracies (per CV run): {results['cv_run_val_accuracies']}") # Show val accuracies per CV run
        print(f"  Average Test Set Accuracy (Combined, averaged over {num_cv_runs} CV runs, per-fold test eval): {results['average_test_accuracy']:.4f}".format(num_cv_runs)) # Avg Test Accuracy
        print(f"  Average Test Set Loss (averaged over {num_cv_runs} CV runs, per-fold test eval): {results['average_test_loss']:.4f}".format(num_cv_runs)) # Avg Test Loss
        print(f"  Avg Test Accuracies per CV run: {results['cv_run_test_accuracies']}") # Show avg test accuracies per CV run
        print(f"  Average Validation Accuracy (Causal Branch, CV averaged over {num_cv_runs} runs): {results['average_val_acc_causal']:.4f}") # Avg val acc causal
        print(f"  Average Validation Accuracy (Trivial Branch, CV averaged over {num_cv_runs} runs): {results['average_val_acc_trivial']:.4f}") # Avg val acc trivial


    # --- 6. Evaluate Best Model on Test Set (Redundant, already evaluated in hyperparameter loop for best params) ---
    print("\n--- Evaluating Best Model on Test Set (Redundant - results already above, averaged over CV runs) ---")
    print(f"Average Test Set Loss (Combined, Best Hyperparameters, averaged over {num_cv_runs} CV runs): {best_hyperparameters['average_test_loss']:.4f}".format(num_cv_runs))
    print(f"Average Test Set Accuracy (Combined, Best Hyperparameters, averaged over {num_cv_runs} CV runs): {best_hyperparameters['average_test_accuracy']:.4f}".format(num_cv_runs))
    print(f"Average Test Set Accuracy (Causal Branch, Best Hyperparameters, averaged over {num_cv_runs} CV runs): {best_hyperparameters['average_val_acc_causal']:.4f}".format(num_cv_runs)) # Print best avg val acc causal
    print(f"Average Test Set Accuracy (Trivial Branch, Best Hyperparameters, averaged over {num_cv_runs} CV runs): {best_hyperparameters['average_val_acc_trivial']:.4f}".format(num_cv_runs)) # Print best avg val acc trivial
    print(f"Test Set Confusion Matrix (Best Hyperparameters):\n...") # Confusion matrix is not averaged across CV runs, and would need to be recomputed if desired for "best" model.


    print("\n--- All Hyperparameter Tuning (CV) Results ---")
    for hyperparams, results in hyperparameter_tuning_results.items():
        print(f"\nHyperparameters: o_weight={hyperparams[0]}, c_weight={hyperparams[1]}")
        print(f"  Average Validation Accuracy (CV, averaged over {num_cv_runs} runs): {results['average_accuracy']:.4f} ± {results['std_dev_accuracy']:.4f}".format(num_cv_runs)) # Averaged CV accuracy
        print(f"  Average Validation Loss (CV, averaged over {num_cv_runs} runs): {results['average_loss']:.4f}".format(num_cv_runs)) # Averaged CV loss
        print(f"  Fold Validation Accuracies (per CV run): {results['cv_run_val_accuracies']}") # Show val accuracies per CV run
        print(f"  Average Test Set Accuracy (averaged over {num_cv_runs} CV runs, per-fold test eval): {results['average_test_accuracy']:.4f}".format(num_cv_runs)) # Avg Test Accuracy
        print(f"  Average Test Set Loss (averaged over {num_cv_runs} CV runs, per-fold test eval): {results['average_test_loss']:.4f}".format(num_cv_runs))     # Avg Test Loss
        print(f"  Avg Test Accuracies per CV run: {results['cv_run_test_accuracies']}") # Show avg test accuracies per CV run


    # --- 6. Evaluate Best Model on Test Set (Redundant, already evaluated in hyperparameter loop for best params) ---
    print("\n--- Evaluating Best Model on Test Set (Redundant - results already above, averaged over CV runs) ---")
    print(f"Average Test Set Loss (Best Hyperparameters, averaged over {num_cv_runs} CV runs): {best_hyperparameters['average_test_loss']:.4f}".format(num_cv_runs))
    print(f"Average Test Set Accuracy (Best Hyperparameters, averaged over {num_cv_runs} CV runs): {best_hyperparameters['average_test_accuracy']:.4f}".format(num_cv_runs))
    print(f"Test Set Confusion Matrix (Best Hyperparameters):\n...") # Confusion matrix is not averaged across CV runs, and would need to be recomputed if desired for "best" model.


def evaluate_on_test_set(model, learning_rate, o_weight, c_weight, test_loader, device, num_classes, input_dim, hidden_dim, heads, dropout, alpha, test_labels_tensor, eval_random_test=False): # Added eval_random_test
    """
    Evaluates the CAL_GAT model on the test set.  This function is called within the hyperparameter tuning loop
    to evaluate the test set performance for each hyperparameter combination's best model (from CV).

    Args:
        o_weight, c_weight, ... (hyperparameters): Hyperparameter values for model initialization.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): 'cuda' or 'cpu'.
        num_classes (int): Number of classes.
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        alpha (float): Alpha for LeakyReLU in GAT.
        test_labels_tensor (torch.Tensor): Tensor of test labels.
        eval_random_test (bool): Whether to use random intervention during test evaluation. # Added eval_random_test

    Returns:
        tuple: Test loss, test accuracy (combined), test accuracy (causal), test accuracy (trivial), confusion matrix, predicted labels. # Updated returns
    """


    # Re-train on the entire train_val dataset with current hyperparameters
    print(" Re-training model on the entire training+validation dataset with current hyperparameters...")
    train_val_dataset = test_loader.dataset # Access the original train_val dataset from the test_loader (which is actually val_loader in the main function's loop but has access to dataset)
    train_val_loader_for_test_eval = DataLoader(train_val_dataset, batch_size=test_loader.batch_size, shuffle=True) # Create loader for train_val dataset

    optimizer_for_test_eval = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3) # Optimizer for final training

    # Evaluate on test set
    print(" Evaluating on test set...")
    test_loss, test_acc_combined, test_acc_causal, test_acc_trivial, test_conf_matrix, test_predicted_labels = evaluate_cal_gat( # Get branch accuracies from evaluation
        model, test_loader, device, num_classes, o_weight=o_weight, c_weight=c_weight, eval_random=eval_random_test # Pass eval_random_test
    )

    return test_loss, test_acc_combined, test_acc_causal, test_acc_trivial, test_conf_matrix, test_predicted_labels # Return branch accuracies


if __name__ == "__main__":
    main()