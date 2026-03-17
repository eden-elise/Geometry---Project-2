import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.font_manager import FontProperties

# --- 0. Configuration ---
# Ensure these paths point to your actual .ttf files
FONT_PATHS = {
    "English": "fonts/Noto_Sans/static/NotoSans_SemiCondensed-Regular.ttf",
    "Arabic": "fonts/Noto_Sans_Arabic/static/NotoSansArabic_SemiCondensed-Regular.ttf",
    "Chinese":  "fonts/Noto_Sans_TC/static/NotoSansTC-Regular.ttf",
}

# --- 1. Resampling and Dataset ---

def resample_points(points, target_len=20):
    points = np.array(points)
    old_steps = np.linspace(0, 1, len(points))
    new_steps = np.linspace(0, 1, target_len)
    f_x = interp1d(old_steps, points[:, 0], kind='linear')
    f_y = interp1d(old_steps, points[:, 1], kind='linear')
    return np.stack([f_x(new_steps), f_y(new_steps)], axis=1).flatten()

class HullDataset(Dataset):
    def __init__(self, json_path, samples_per_lang=3000, target_pts=20):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        lang_groups = {}
        for entry in raw_data:
            lang = entry['language']
            lang_groups.setdefault(lang, []).append(entry)
        
        self.samples, self.labels = [], []
        self.raw_records = []  # FIXED: Now stores original data for plotting
        self.label_to_idx = {lang: i for i, lang in enumerate(lang_groups.keys())}
        self.idx_to_label = {i: lang for lang, i in self.label_to_idx.items()}
        
        for lang, entries in lang_groups.items():
            for entry in entries[:samples_per_lang]:
                self.raw_records.append(entry) 
                self.samples.append(resample_points(entry['hull_points'], target_pts))
                self.labels.append(self.label_to_idx[lang])
        
        # FIXED: Wrap in np.array() to solve UserWarning and speed up loading
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

# --- 2. Probability-Based Model ---

class LanguageClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LanguageClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

# --- 3. Visualization Function ---

def visualize_prediction_gallery(model, test_db, dataset_obj, samples_per_lang=4):
    model.eval()
    
    # We define the column order strictly
    target_langs = ["English", "Arabic", "Chinese"]
    cols = len(target_langs)
    rows = samples_per_lang
    num_total = rows * cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    # Ensure axes is a 2D array even if rows=1
    if rows == 1:
        axes = np.array([axes])

    # 1. Group all test_db indices by their language label
    lang_to_test_indices = {lang: [] for lang in target_langs}
    for idx_in_test_db in range(len(test_db)):
        _, label_tensor = test_db[idx_in_test_db]
        lang_name = dataset_obj.idx_to_label[label_tensor.item()]
        if lang_name in lang_to_test_indices:
            lang_to_test_indices[lang_name].append(idx_in_test_db)

    # 2. Pick random samples for each column
    selected_indices_grid = []
    for r in range(rows):
        row_indices = []
        for lang in target_langs:
            # Pick one random index for this language column
            idx = random.choice(lang_to_test_indices[lang])
            row_indices.append(idx)
        selected_indices_grid.append(row_indices)

    # 3. Plotting
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            test_idx = selected_indices_grid[r][c]
            
            # Data Retrieval
            input_tensor, target_label = test_db[test_idx]
            original_idx = test_db.indices[test_idx]
            raw_record = dataset_obj.raw_records[original_idx]
            
            # Prediction
            with torch.no_grad():
                output = model(input_tensor.unsqueeze(0))
                probs = torch.softmax(output, dim=1)[0]
            
            actual_lang = dataset_obj.idx_to_label[target_label.item()]
            pred_idx = torch.argmax(probs).item()
            pred_lang = dataset_obj.idx_to_label[pred_idx]
            word = raw_record['word']
            hull_pts = np.array(raw_record['hull_points'])
            
            # Plot Text Path
            try:
                font_p = FontProperties(fname=FONT_PATHS[actual_lang])
                from matplotlib.textpath import TextPath
                t_path = TextPath((0, 0), word, size=100, prop=font_p)
                ax.scatter(t_path.vertices[:, 0], t_path.vertices[:, 1], s=0.5, color="gray", alpha=0.2)
            except:
                font_p = None

            # Draw Hull
            poly = Polygon(hull_pts, facecolor='#4A90D9', alpha=0.1, edgecolor='#4A90D9', lw=1.5)
            ax.add_patch(poly)
            ax.scatter(hull_pts[:, 0], hull_pts[:, 1], s=8, color="#E85D4A", alpha=0.6)

            # Confidence Bars (Mini-Legend)
            conf_info = ""
            for j, p in enumerate(probs):
                l_name = dataset_obj.idx_to_label[j]
                bar_str = "#" * int(p.item() * 10)
                mark = "★" if j == pred_idx else " "
                conf_info += f"{mark}{l_name[:2]}: {p.item():.1%} {bar_str}\n"
            
            ax.text(1.02, 0.5, conf_info, transform=ax.transAxes, fontsize=8, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))

            # Styling
            is_correct = (actual_lang == pred_lang)
            ax.set_facecolor("#f9f9f9" if is_correct else "#fff0f0")
            title_color = "green" if is_correct else "red"
            ax.set_title(f"({actual_lang}): '{word}'\nPred: {pred_lang}", 
                         fontproperties=font_p, fontsize=11, color=title_color)
            
            ax.set_aspect("equal")
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(right=0.9, hspace=0.4, wspace=0.6)
    plt.show()
# To call this in your run_experiment():
# visualize_prediction_gallery(model, test_db, dataset)
import matplotlib.pyplot as plt

def plot_model_structure(input_dim, hidden1, hidden2, output_dim):
    # Layer definitions: name, size, and vertical offset for centering
    layers = [
        ("Input\n(Resampled Hull)", input_dim, "skyblue"),
        ("Hidden 1\n(ReLU + Dropout)", hidden1, "lightgreen"),
        ("Hidden 2\n(ReLU)", hidden2, "lightgreen"),
        ("Output\n(Language)", output_dim, "orange")
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Horizontal spacing between layers
    layer_x = [0, 2, 4, 6]
    
    # We will limit how many neurons we actually draw for large layers
    max_visible_neurons = 10 

    for i, (name, size, color) in enumerate(layers):
        # Calculate how many neurons to actually draw
        n_to_draw = min(size, max_visible_neurons)
        y_points = np.linspace(-n_to_draw/2, n_to_draw/2, n_to_draw)
        
        # Draw neurons
        for y in y_points:
            circle = plt.Circle((layer_x[i], y), 0.15, color=color, ec='black', zorder=4)
            ax.add_artist(circle)
        
        # If we truncated the layer, add "..." in the middle
        if size > max_visible_neurons:
            ax.text(layer_x[i], 0, '...', fontsize=20, ha='center', va='center')

        # Label the layer
        ax.text(layer_x[i], n_to_draw/2 + 0.5, f"{name}\n({size} nodes)", 
                ha='center', va='bottom', fontweight='bold')

        # Draw connections to the next layer
        if i < len(layers) - 1:
            next_size = layers[i+1][1]
            next_n_to_draw = min(next_size, max_visible_neurons)
            next_y_points = np.linspace(-next_n_to_draw/2, next_n_to_draw/2, next_n_to_draw)
            
            for y_curr in y_points:
                for y_next in next_y_points:
                    ax.plot([layer_x[i], layer_x[i+1]], [y_curr, y_next], 
                            color='gray', alpha=0.1, lw=0.5, zorder=1)

    ax.set_xlim(-1, 7)
    ax.set_ylim(-max_visible_neurons/2 - 1, max_visible_neurons/2 + 2)
    plt.title("Language Classifier: Neural Network Architecture", fontsize=16)
    plt.show()

# To call it:
# plot_model_structure(input_dim=40, hidden1=128, hidden2=64, output_dim=3)
# --- 4. Main Experiment ---

def run_experiment():
    TARGET_PTS = 20
    dataset = HullDataset('dataset.json', samples_per_lang=3000, target_pts=TARGET_PTS)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_db, test_db = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_db, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=32, shuffle=False)

    model = LanguageClassifier(TARGET_PTS * 2, len(dataset.label_to_idx))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- TRACKING METRICS ---
    loss_history = []
    accuracy_history = [] 

    print(f"Training model on {len(train_db)} samples...")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
        
        # Calculate epoch averages
        avg_loss = total_loss / len(train_loader)
        epoch_acc = (total_correct / total_samples) * 100
        
        loss_history.append(avg_loss)
        accuracy_history.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # --- PLOT LOSS AND ACCURACY ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Error (Loss)
    ax1.plot(range(1, epochs + 1), loss_history, marker='o', color='#E85D4A', linewidth=2)
    ax1.set_title('Training Error (Loss) per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Accuracy (Correct / Total)
    ax2.plot(range(1, epochs + 1), accuracy_history, marker='s', color='#4A90D9', linewidth=2)
    ax2.set_title('Training Accuracy (%) per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # Final Accuracy check on Test Set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"\nFinal Accuracy on Test Set: {100 * correct / total:.2f}%")
    
    # Rest of your visualizations
    plot_model_structure(input_dim=40, hidden1=128, hidden2=64, output_dim=3)
    visualize_prediction_gallery(model, test_db, dataset)


run_experiment()