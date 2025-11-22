import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colores
color_blue = '#4A90E2'
color_green = '#7ED321'
color_orange = '#F5A623'

# Definir las fases de CRISP-DM
phases = [
    {'name': 'Business\nUnderstanding', 'pos': (5, 8.5), 'color': color_blue},
    {'name': 'Data\nUnderstanding', 'pos': (2, 6.5), 'color': color_green},
    {'name': 'Data\nPreparation', 'pos': (2, 4.5), 'color': color_green},
    {'name': 'Modeling', 'pos': (5, 3), 'color': color_orange},
    {'name': 'Evaluation', 'pos': (8, 4.5), 'color': color_orange},
    {'name': 'Deployment', 'pos': (8, 6.5), 'color': '#BD10E0'}
]

# Dibujar las cajas
boxes = []
for phase in phases:
    box = FancyBboxPatch(
        (phase['pos'][0] - 0.8, phase['pos'][1] - 0.4),
        1.6, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=phase['color'],
        edgecolor='black',
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(box)
    ax.text(phase['pos'][0], phase['pos'][1], phase['name'],
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Dibujar flechas entre fases
arrows = [
    ((5, 8.1), (2.5, 6.9)),  # Business -> Data Understanding
    ((2, 6.1), (2, 4.9)),     # Data Understanding -> Data Preparation
    ((2.5, 4.1), (4.5, 3.4)), # Data Preparation -> Modeling
    ((5.5, 3.4), (7.5, 4.9)), # Modeling -> Evaluation
    ((8, 5.3), (8, 6.1)),     # Evaluation -> Deployment
    ((7.5, 6.9), (5.5, 8.1)), # Deployment -> Business (ciclo)
]

for start, end in arrows:
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->,head_width=0.4,head_length=0.8',
        color='black',
        linewidth=2,
        alpha=0.6
    )
    ax.add_patch(arrow)

# Título
ax.text(5, 9.5, 'CRISP-DM Methodology',
        ha='center', va='center', fontsize=16, fontweight='bold')

# Subtítulo
ax.text(5, 0.5, 'Pipeline del Proyecto de Clasificación de Actividades Humanas',
        ha='center', va='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig('IMAGEN_NECESARIA_crisp_dm_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ Guardado: IMAGEN_NECESARIA_crisp_dm_diagram.png")
plt.close()
