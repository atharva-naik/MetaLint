import matplotlib.pyplot as plt

# Data
pep_counts = {
    '593': 20,
    '634': 17+1,
    '525': 19,
    '616': 20,
    '487': 20,
    '614': 20,
    '530': 22,
    '498': 21,
    '567': 20,
    '584': 20,
    '506': 11+2,
    '655': 15,
    '526': 20,
    '557': 13+2,
    "249": 1,
    # '506_557': 2,
    # '634_249': 1
}

# Titles for each PEP
pep_titles = {
    '593': 'Annotated Types',
    '634': 'Pattern Matching',
    '525': 'Async Generators',
    '616': 'Str Removals',
    '487': 'Class Init Hook',
    '614': 'Decorator Grammar',
    '530': 'Async Comprehensions',
    '498': 'f-Strings',
    '567': 'ContextVars',
    '584': 'Dict Merge',
    '506': 'Secrets Module',
    '655': 'TypedDict Req',
    '526': 'Var Annotations',
    '557': 'Data Classes',
    # '506_557': 'Secrets & Data Classes',
    '249': 'DB API'
}
# Prepare labels and sizes
labels = [f"PEP {pep}: {pep_titles.get(pep, '')}" for pep in pep_counts]
sizes = list(pep_counts.values())

# Create pie chart
fig, ax = plt.subplots(figsize=(12, 12))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 14}  # Set font size for both labels and percentages
)

# Optionally, customize percentage labels further
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_color('black')  # You can change the color if needed

# Set title with larger font size
plt.title('Distribution of Benchmark Instances by PEP', fontsize=20)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')
plt.tight_layout()
plt.savefig("plots/hard_pep_benchmark_dist.png")