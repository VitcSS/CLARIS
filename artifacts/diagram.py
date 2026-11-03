import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 3.2), dpi=300)
ax.axis('off')

# Definição dos blocos do pipeline
blocks = [
    ("1. Input & Inference", "Black-Box Model $f(x)$\nQuery $x_{\\mathrm{new}}$, $\\mathcal{D}_{\\mathrm{conf}}$", "#E8F0FE"),
    ("2. Latent Projection", "SHAP TreeExplainer\nMapping to $\\mathcal{S}_{\\mathrm{conf}}$ ($s_{\\mathrm{new}}$)", "#D2E3FC"),
    ("3. CBR Vector DB", "FAISS HNSW Index\nAnchors $\\mathcal{A}$ & CFs $\\mathcal{C}$", "#CEEAD6"),
    ("4. Delta Extractor", "$\\mu_{\\mathrm{cf}}$ Centroid Contrast\n$|x_{\\mathrm{new}} - \\mu_{\\mathrm{cf}}|$\nStrict Pydantic Schema", "#FEF7E0"),
    ("5. Factual NLG", "Constrained LLM ($\\tau = 0$)\nDomain Dossier Injection\nAudit Report ($1.00$)", "#FCE8E6")
]

x_start = 0.02
box_width = 0.16
box_height = 0.65
y_start = 0.25

for i, (title, desc, color) in enumerate(blocks):
    x = x_start + i * 0.198
    # Desenho da caixa
    rect = patches.FancyBboxPatch(
        (x, y_start), box_width, box_height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2, edgecolor="#3C4043", facecolor=color
    )
    ax.add_patch(rect)
    
    # Textos dentro do bloco
    ax.text(x + box_width/2, y_start + box_height - 0.12, title,
            ha='center', va='center', fontsize=9.5, fontweight='bold', color="#202124")
    ax.text(x + box_width/2, y_start + box_height/2 - 0.08, desc,
            ha='center', va='center', fontsize=8, color="#3C4043")
    
    # Seta de fluxo entre os blocos
    if i < len(blocks) - 1:
        ax.annotate('', xy=(x + box_width + 0.038, y_start + box_height/2),
                    xytext=(x + box_width, y_start + box_height/2),
                    arrowprops=dict(facecolor='#5F6368', edgecolor='#5F6368', width=1.5, headwidth=6, shrink=0.05))

# Bloco transversal de monitoramento MLOps (inferior)
drift_box = patches.FancyBboxPatch(
    (x_start + 0.198, 0.02), 0.556, 0.16,
    boxstyle="round,pad=0.01,rounding_size=0.02",
    linewidth=1.0, linestyle="--", edgecolor="#EA8600", facecolor="#FFF8E1"
)
ax.add_patch(drift_box)
ax.text(x_start + 0.198 + 0.556/2, 0.10,
        "Continuous MLOps Surveillance: Window Drift Tracking via $\\bar{d}_E$ (Data) & $\\overline{\\cos\\theta}$ (Concept)",
        ha='center', va='center', fontsize=7.5, fontweight='bold', color="#B06000")

# Setas indicando a conexão com o módulo de drift
ax.annotate('', xy=(0.32, 0.18), xytext=(0.32, 0.25),
            arrowprops=dict(arrowstyle="->", color='#EA8600', lw=1.2, linestyle=':'))
ax.annotate('', xy=(0.52, 0.18), xytext=(0.52, 0.25),
            arrowprops=dict(arrowstyle="->", color='#EA8600', lw=1.2, linestyle=':'))

plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig("claris_architecture.pdf", bbox_inches='tight')
plt.savefig("claris_architecture.png", bbox_inches='tight')
print("Figura gerada com sucesso!")