# Unipolar3kfinalB.py
# ============================================================
# Split-unipolar trait extraction + embedding fusion + clustering
# FIXES:
#  - Requires `umap-learn` (install: conda install -c conda-forge umap-learn)
#  - Removes wasted extract_batch(texts) call
#  - Saves Step2-compatible CSV with expected column names
# ============================================================

import json, re, os, joblib, numpy as np, pandas as pd
import time
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE

# IMPORTANT: pip install umap-learn  OR  conda install -c conda-forge umap-learn
from umap import UMAP

from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal, chi2_contingency
import seaborn as sns
import httpx
import matplotlib.patches as mpatches

# -------------------- OpenAI client --------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                timeout=httpx.Timeout(120.0))

# -------------------- Paths --------------------
INPUT_FILE = Path(r"C:\Users\vinod\Dropbox\Indra - Research Folder\Data\Trait\Masked\maskedalldata.JSON")
RESULTS_FOLDER = Path(r"C:\Users\vinod\Dropbox\Indra - Research Folder\Results\Trait Paper3")
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

CACHE_FILE = RESULTS_FOLDER / "trait_cache_split_unipolar_k3_final.pkl"

# -------------------- Trait schema --------------------
TRAITS = ['U', 'Crowd_Love', 'Crowd_Aversion', 'Walk_Love', 'Walk_Aversion', 'S', 'Scenic_Love', 'Scenic_Aversion', 'R', 'K']
trait_labels = ['Urgency', 'Crowd Love', 'Crowd Aversion', 'Walk Love', 'Walk Aversion', 'Safety', 'Scenic Love', 'Scenic Aversion', 'Reliability', 'Cost']

# -------------------- Load texts --------------------
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    raw = f.read()

try:
    data = json.loads(raw)
except Exception:
    data = [json.loads(m) for m in re.findall(r'\{[^{}]*"content"[^{}]*\}', raw) if m.count('{') == m.count('}')]

texts = [(item.get('content') or '').strip()[:800] for item in data if item.get('content')]
df = pd.DataFrame({'content': texts})
print(f"Loaded {len(df)} reviews")

# -------------------- Few-shot --------------------
few_shot = """
Examples (match exactly):
1. "Crowds are hell, avoid at all costs" → {"U":0.6,"Crowd_Love":0.0,"Crowd_Aversion":1.0,"Walk_Love":0.1,"Walk_Aversion":0.4,"S":0.7,"Scenic_Love":0.2,"Scenic_Aversion":0.8,"R":0.7,"K":0.3}
2. "Love walking scenic paths even if longer" → {"U":0.1,"Crowd_Love":0.3,"Crowd_Aversion":0.2,"Walk_Love":1.0,"Walk_Aversion":0.0,"S":0.2,"Scenic_Love":1.0,"Scenic_Aversion":0.0,"R":0.2,"K":0.0}
3. "Get me there fast, hate detours" → {"U":0.95,"Crowd_Love":0.1,"Crowd_Aversion":0.5,"Walk_Love":0.0,"Walk_Aversion":0.9,"S":0.3,"Scenic_Love":0.0,"Scenic_Aversion":1.0,"R":0.95,"K":0.4}
4. "Empty quiet ride is perfect" → {"U":0.3,"Crowd_Love":0.9,"Crowd_Aversion":0.0,"Walk_Love":0.4,"Walk_Aversion":0.1,"S":0.1,"Scenic_Love":0.7,"Scenic_Aversion":0.0,"R":0.5,"K":0.0}
5. "Delays unacceptable, too expensive" → {"U":0.9,"Crowd_Love":0.2,"Crowd_Aversion":0.3,"Walk_Love":0.1,"Walk_Aversion":0.2,"S":0.4,"Scenic_Love":0.0,"Scenic_Aversion":0.8,"R":1.0,"K":0.9}
"""

prompt = f"""
Output ONLY a JSON object with a single key "scores" that contains an array of exactly {{N}} trait objects, one for each review in the order provided.
Each object in the array must have exactly these keys: {', '.join(TRAITS)}, with values as floats from 0.0 to 1.0.
No extra text, no explanations, no additional keys — strictly {{"scores": [ {{"U": 0.5, "Crowd_Love": 0.3, ...}}, ... ] }}.
Use the full 0-1 range; for strong aversions, use 0.8-1.0.

Trait definitions:
U: 0–1 urgency
Crowd_Love: 0–1 enjoys crowds
Crowd_Aversion: 0–1 dislikes crowds
Walk_Love: 0–1 enjoys walking
Walk_Aversion: 0–1 dislikes walking
S: 0–1 safety concern
Scenic_Love: 0–1 values scenic routes
Scenic_Aversion: 0–1 prefers fastest over scenic
R: 0–1 reliability demand
K: 0–1 cost sensitivity

For pairs (Love/Aversion), usually one high, other low; both high if ambivalent.

{few_shot}

Reviews:
"""

def extract_batch(texts_batch):
    numbered = "\n".join(str(i+1) + '. "' + t.replace('"', '\\"') + '"' for i, t in enumerate(texts_batch))
    system_prompt = "You are a helpful assistant that outputs only valid JSON as specified in the user prompt. Do not add any additional text."

    user_prompt = prompt.replace("{N}", str(len(texts_batch))) + numbered

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=4096
            )

            content = resp.choices[0].message.content.strip()
            data_out = json.loads(content)
            results = data_out.get("scores", [])

            if not isinstance(results, list) or len(results) != len(texts_batch):
                raise ValueError(f"Invalid scores: got {len(results)} items, expected {len(texts_batch)}")

            for res in results:
                if set(res.keys()) != set(TRAITS):
                    raise ValueError("Invalid trait object keys")
                for v in res.values():
                    if not isinstance(v, (float, int)) or not (0 <= float(v) <= 1):
                        raise ValueError("Invalid trait object values")

            # normalize to float
            cleaned = []
            for res in results:
                cleaned.append({k: float(res[k]) for k in TRAITS})
            return cleaned

        except Exception as e:
            print(f"API error on attempt {attempt+1}: {str(e)}")
            if 'content' in locals():
                print(f"Raw response: {content[:500]}...")
            if attempt < 4:
                time.sleep(4 ** attempt)
            else:
                print("Max retries reached — using defaults")
                return [{t: 0.3 for t in TRAITS} for _ in texts_batch]

# -------------------- Trait extraction (cached) --------------------
if CACHE_FILE.exists():
    print("Loading cached split-unipolar traits...")
    trait_matrix = joblib.load(CACHE_FILE)
else:
    trait_matrix = []
    batch_size = 3
    print("Extracting split-unipolar traits...")

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df['content'].iloc[i:i+batch_size].tolist()
        scores = extract_batch(batch)
        for s in scores:
            arr = np.array([float(s.get(t, 0.3)) for t in TRAITS], dtype=float)
            trait_matrix.append(arr)

    joblib.dump(trait_matrix, CACHE_FILE)
    print("Split-unipolar cache saved")

df_traits = pd.DataFrame(trait_matrix, columns=TRAITS)
df_traits.columns = trait_labels

# -------------------- Filter weak samples --------------------
trait_var = df_traits.std(axis=1)
mask = trait_var > 0.20
df = df.loc[mask].copy()
df_traits = df_traits.loc[mask].copy()
print(f"Filtered to {len(df)} high-signal samples")

df_traits_scaled = pd.DataFrame(StandardScaler().fit_transform(df_traits),
                                columns=trait_labels,
                                index=df_traits.index)

# -------------------- Embeddings --------------------
def embed_batch(text_list):
    embeds = []
    batch_size_embed = 100
    for i in range(0, len(text_list), batch_size_embed):
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=text_list[i:i+batch_size_embed]
        )
        embeds.extend([e.embedding for e in resp.data])
    return np.array(embeds, dtype=float)

semantic = embed_batch(df['content'].tolist())
semantic_reduced = PCA(n_components=100, random_state=42).fit_transform(semantic)
fused = np.hstack([semantic_reduced, 1.5 * df_traits_scaled.values])

X_final = UMAP(n_components=50, random_state=42).fit_transform(fused)

# -------------------- Clustering --------------------
print("Evaluating clusters with Agglomerative...")
best_sil = -1
best_k = 3
best_labels = None

for k in [3]:
    clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = clusterer.fit_predict(X_final)
    sil = silhouette_score(X_final, labels)
    dbi = davies_bouldin_score(X_final, labels)
    ch = calinski_harabasz_score(X_final, labels)
    print(f"k={k} → Silhouette: {sil:.4f}, DBI: {dbi:.4f}, CH: {ch:.4f}")
    if sil > best_sil:
        best_sil = sil
        best_k = k
        best_labels = labels

print(f"\nBest k = {best_k} (Silhouette = {best_sil:.4f})")

df['cluster'] = best_labels

cluster_means = df_traits.groupby(df['cluster']).mean()
cluster_means.round(4).to_csv(RESULTS_FOLDER / "cluster_centroids_split_unipolar_k3.csv")

# -------------------- Save full outputs --------------------
df_full = df.copy()
df_full[trait_labels] = df_traits.loc[df.index, trait_labels].values

# Original (human label) version
df_full.to_csv(RESULTS_FOLDER / "full_data_k3.csv", index=False)
joblib.dump(df_full, RESULTS_FOLDER / "full_data_k3.pkl")

# Step2-compatible column names
rename_for_step2 = {
    "Urgency": "Urgency",
    "Crowd Love": "Crowd_Love",
    "Crowd Aversion": "Crowd_Aversion",
    "Walk Love": "Walk_Love",
    "Walk Aversion": "Walk_Aversion",
    "Safety": "Safety_Sensitivity",
    "Scenic Love": "Scenic_Love",
    "Scenic Aversion": "Scenic_Aversion",
    "Reliability": "Reliability_Expectation",
    "Cost": "Cost_Sensitivity",
}

df_full_step2 = df_full.rename(columns=rename_for_step2)
df_full_step2.to_csv(RESULTS_FOLDER / "full_data_k3_step2.csv", index=False)
print("Saved Step2-compatible file:", RESULTS_FOLDER / "full_data_k3_step2.csv")

# -------------------- t-SNE Plot --------------------
X_tsne = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(X_final)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
persona_names = ["Efficient Minimalist", "Safety-Conscious", "Scenic Walker",
                 "Pragmatic Commuter", "Comfort-Seeker", "Crowd-Lover", "Cost-Driven"][:best_k]

for i in range(best_k):
    m = (df['cluster'] == i).values
    ax.scatter(X_tsne[m, 0], X_tsne[m, 1],
               c=colors[i], label=f"{persona_names[i]} (n={m.sum()})",
               s=140, alpha=0.85, edgecolor='k')

x_min, x_max = X_tsne[:, 0].min(), X_tsne[:, 0].max()
y_min, y_max = X_tsne[:, 1].min(), X_tsne[:, 1].max()
padding = 0.1 * max(x_max - x_min, y_max - y_min)

ax.set_xlim(x_min - padding, x_max + padding)
ax.set_ylim(y_min - padding, y_max + padding)
ax.legend(fontsize=20, loc='lower right')
ax.set_title(f"t-SNE – Split Unipolar Schema\nSilhouette = {best_sil:.3f} (k={best_k})", fontsize=22, fontweight='bold')
ax.tick_params(labelsize=16)
ax.set_xlabel("t-SNE Dimension 1", fontsize=18, fontweight='bold')
ax.set_ylabel("t-SNE Dimension 2", fontsize=18, fontweight='bold')
fig.tight_layout()
fig.savefig(RESULTS_FOLDER / "tSNE_split_unipolar_k3.png", dpi=600)
print("t-SNE plot saved")

# -------------------- Silhouette Plot --------------------
sample_sil = silhouette_samples(X_final, best_labels)

fig_sil = plt.figure(figsize=(10, 7))
ax_sil = fig_sil.add_subplot(111)

y_lower = 10
for i in range(best_k):
    cluster_sil = sample_sil[best_labels == i]
    cluster_sil.sort()
    y_upper = y_lower + cluster_sil.shape[0]
    color = colors[i]
    ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=color, edgecolor=color, alpha=0.7)
    ax_sil.text(-0.05, y_lower + 0.5 * cluster_sil.shape[0], str(i))
    y_lower = y_upper + 10

ax_sil.set_title(f"Silhouette Plot for k={best_k}", fontsize=18, fontweight='bold')
ax_sil.set_xlabel("Silhouette coefficient values", fontsize=14, fontweight='bold')
ax_sil.set_ylabel("Cluster label", fontsize=14, fontweight='bold')
ax_sil.axvline(x=best_sil, color="red", linestyle="--")
ax_sil.set_yticks([])
handles = [mpatches.Patch(color=colors[i], label=persona_names[i]) for i in range(best_k)]
ax_sil.legend(handles=handles, loc='lower right', fontsize=10)
fig_sil.tight_layout()
fig_sil.savefig(RESULTS_FOLDER / "silhouette_plot_k3.png", dpi=600)
print("Silhouette plot saved")

# -------------------- Correlation Matrix --------------------
corr_matrix = df_traits.corr()
fig_corr = plt.figure(figsize=(12, 10))
ax_corr = fig_corr.add_subplot(111)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, annot_kws={"size": 12})
ax_corr.set_title("Trait Correlation Matrix", fontsize=18, fontweight='bold')
ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right', fontsize=12)
ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0, fontsize=12)
fig_corr.tight_layout()
fig_corr.savefig(RESULTS_FOLDER / "trait_correlation_matrix.png", dpi=600)
print("Trait correlation matrix saved")

# -------------------- Tables + stats (kept as you had) --------------------
def get_distinctive_traits(row):
    numeric_row = row[trait_labels] if isinstance(row, pd.Series) else row[trait_labels]
    traits_sorted = numeric_row.sort_values(ascending=False).head(3)
    return ', '.join([f"{trait} ({value:.2f})" for trait, value in traits_sorted.items()])

cluster_means['Distinctive Traits'] = cluster_means.apply(get_distinctive_traits, axis=1)

interpretations = [
    "High Reliability, Urgency, Crowd Aversion – Efficient Minimalists: prioritize time and reliability, avoid crowding and detours.",
    "High Crowd Love, Walk Love, Reliability – Active Commuters: tolerate crowds, enjoy short walking, still value reliability.",
    "High Reliability, Scenic Aversion, Urgency – Direct Travelers: prefer direct routes, time-sensitive, dislike scenic detours."
]
cluster_means['Interpretation'] = interpretations[:len(cluster_means)]

table_df = cluster_means[['Distinctive Traits', 'Interpretation']].reset_index()
table_df.columns = ['Cluster', 'Distinctive Traits', 'Interpretation']

print("\nTABLE III")
print("Behavioral Cluster Profiles and Dominant Traits")
print(table_df.to_string(index=False))

# Table IV
corr_flat = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
top_corr = corr_flat[(corr_flat < 1) & (corr_flat > 0.5)].head(3)

table_corr = pd.DataFrame({
    'Trait 1': top_corr.index.get_level_values(0),
    'Trait 2': top_corr.index.get_level_values(1),
    'r': top_corr.values
})
table_corr['Interpretation'] = ['High coupling', 'Strong positive link', 'Strong association']

print("\nTABLE IV")
print("Top Pairwise Trait Correlations")
print(table_corr.to_string(index=False))

# Table V
df_full['Distinctive Traits'] = df_traits.apply(get_distinctive_traits, axis=1)
representative = df_full.groupby('cluster').apply(lambda g: g.sample(1, random_state=42), include_groups=False).reset_index()
representative['Dominant Traits'] = representative.apply(lambda row: get_distinctive_traits(cluster_means.loc[row['cluster']]), axis=1)
representative = representative[['cluster', 'content', 'Dominant Traits']]
representative.columns = ['Cluster', 'Example Text (truncated)', 'Dominant Traits']
representative['Example Text (truncated)'] = representative['Example Text (truncated)'].str[:100] + '...'

print("\nTABLE V")
print("Representative Texts and Dominant Traits")
print(representative.to_string(index=False))

# Stats
print("\nStatistical Significance")
for trait in trait_labels:
    groups = [df_traits[df['cluster'] == i][trait] for i in range(best_k) if len(df_traits[df['cluster'] == i][trait]) > 0]
    if len(groups) > 1 and all(len(g) > 0 for g in groups):
        anova_p = f_oneway(*groups).pvalue
        kruskal_p = kruskal(*groups).pvalue
    else:
        anova_p = np.nan
        kruskal_p = np.nan
    print(f"{trait}: ANOVA p = {anova_p:.4f}, Kruskal-Wallis p = {kruskal_p:.4f}")

# Chi-square example
df_full['U_bin'] = pd.cut(df_full['Urgency'], bins=2, labels=['Low', 'High'])
df_full['R_bin'] = pd.cut(df_full['Reliability'], bins=2, labels=['Low', 'High'])
contingency = pd.crosstab(df_full['U_bin'], df_full['R_bin'])

if contingency.shape == (2, 2) and contingency.values.min() > 0:
    chi2, p, dof, ex = chi2_contingency(contingency)
else:
    p = np.nan

print(f"Chi-Square for U-R: p = {p:.4f}")
print("\nDone!")
