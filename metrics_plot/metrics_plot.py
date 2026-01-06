import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# 1. Data from the table
# (Model, Method, Dataset): [JSS, err, Chi-sq, err, PCC, err]
data = {
    ('ResNet', 'GradCAM'    , 'CUB'): [0.5978, 0.0000, 0.4372, 0.0000, 0.7250, 0.0000],
    ('ResNet', 'GradCAM'    , 'CXR'): [0.4897, 0.0780, 0.1474, 0.2490, 0.2834, 0.2289],
    ('ResNet', 'ScoreCAM'   , 'CUB'): [0.5625, 0.0000, 0.3451, 0.0000, 0.6941, 0.0000],
    ('ResNet', 'ScoreCAM'   , 'CXR'): [0.6263, 0.0480, 0.5162, 0.1200, 0.3305, 0.2020],
    ('ResNet', 'AblationCAM', 'CUB'): [0.5911, 0.0000, 0.4226, 0.0000, 0.7290, 0.0000],
    ('ResNet', 'AblationCAM', 'CXR'): [0.5408, 0.0650, 0.2975, 0.1840, 0.3051, 0.2330],
    ('Mamba' , 'MambaLRP'   , 'CUB'): [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    ('Mamba' , 'MambaLRP'   , 'CXR'): [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    ('ViT'   , 'ViTLRP'     , 'CUB'): [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    ('ViT'   , 'ViTLRP'     , 'CXR'): [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
}

# Sequence of 10 columns 
# (Model, Method, Dataset)
sequence = [
    ('ResNet', 'GradCAM'    , 'CUB'), 
    ('ResNet', 'GradCAM'    , 'CXR'),
    ('ResNet', 'ScoreCAM'   , 'CUB'), 
    ('ResNet', 'ScoreCAM'   , 'CXR'),
    ('ResNet', 'AblationCAM', 'CUB'), 
    ('ResNet', 'AblationCAM', 'CXR'),
    ('Mamba' , 'MambaLRP'   , 'CUB'), 
    ('Mamba' , 'MambaLRP'   , 'CXR'),
    ('ViT'   , 'ViTLRP'     , 'CUB'), 
    ('ViT'   , 'ViTLRP'     , 'CXR')
]

# 2. Plot setup
fig, ax = plt.subplots(figsize=(14, 7))

# Colors and markers matching the sketch legend
metrics_info = [
    {'name': 'JSS'   , 'marker': 'o', 'color': 'blue'},
    {'name': 'Chi-sq', 'marker': 's', 'color': 'red'},
    {'name': 'PCC'   , 'marker': '^', 'color': 'green'}
]

x_indices = np.arange(len(sequence))

# 3. Plotting only the metric symbols
for i, key in enumerate(sequence):
    vals = data[key]
    for j, m_info in enumerate(metrics_info):
        mean = vals[j*2]
        ax.plot(i, mean, marker=m_info['marker'], color=m_info['color'], 
                markersize=10, markeredgecolor='black', markeredgewidth=1, linestyle='None')

# 4. Styling the Y-axis
ax.set_ylim(-0.05, 1.1)
ax.set_ylabel('Metric Score', fontsize=12)

# Set multiple red ticks on the y-axis
y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
ax.set_yticks(y_ticks)
ax.set_yticklabels([str(t) for t in y_ticks], fontsize=12, color='black')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 5. Adding Hierarchical Labels
# Level 1: Dataset labels (CUB, CXR)
ax.set_xticks(x_indices)
ax.set_xticklabels([k[2] for k in sequence], fontsize=10, color='black')

# Level 2: Method labels
method_centers = [0.5, 2.5, 4.5, 6.5, 8.5]
method_names = ['GradCAM', 'ScoreCAM', 'AblationCAM', 'MambaLRP', 'ViTLRP']
for pos, name in zip(method_centers, method_names):
    ax.text(pos, -0.15, name, ha='center', fontsize=11, color='blue', fontweight='bold')

# Level 3: Model labels (Bottom row)
model_centers = [2.5, 6.5, 8.5]
model_names = ['Resnet', 'Mamba', 'ViT']
for pos, name in zip(model_centers, model_names):
    ax.text(pos, -0.22, name, ha='center', fontsize=15, color='black', fontweight='bold')

# 6. Vertical Separators
ax.axvline(5.5, color='black', linewidth=1.5, alpha=0.6) # Divider between Resnet and Mamba
ax.axvline(7.5, color='black', linewidth=1.5, alpha=0.6) # Divider between Mamba and ViT
for v in [1.5, 3.5]: # Dashed dividers between methods in Resnet
    ax.axvline(v, color='black', linestyle='--', linewidth=1.5, alpha=0.6)

# 7. Legend
legend_elements = [Line2D([0], [0], marker=m['marker'], color='w', label=m['name'],
                          markerfacecolor=m['color'], markersize=10, markeredgecolor='black')
                   for m in metrics_info]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('metrics_plot.png')
plt.show()