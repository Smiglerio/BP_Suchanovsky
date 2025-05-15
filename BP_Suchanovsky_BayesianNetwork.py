import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/BP_BayesReadyData.csv")

columns_of_interest = ["ghi_pyr", "ghi_rsi", "dni", "dhi",
                       "air_temperature", "relative_humidity",
                       "wind_speed", "wind_from_direction",
                       "Energy", "Efficiency_Temp", "Real Energy", "Error"]

df_cleaned = df.dropna(subset=columns_of_interest).copy()

def discretize_column(column, bins=3):
    return pd.cut(df_cleaned[column], bins=bins, labels=False, duplicates="drop")
for col in columns_of_interest:
    if df_cleaned[col].dtype in ['float64', 'int64']:
        df_cleaned[col] = discretize_column(col)
def classify_energy(val):
    if val < 215:
        return 'nizka'
    elif val <= 315:
        return 'stredna'
    else:
        return 'vysoka'
df_cleaned['Trieda'] = df['Real Energy'].apply(classify_energy)
edges = [
    ('ghi_pyr', 'Efficiency_Temp'),('air_temperature', 'Efficiency_Temp'),
    ('relative_humidity', 'Efficiency_Temp'),('wind_speed', 'Efficiency_Temp'),
    ('wind_from_direction','Efficiency_Temp'),('ghi_pyr', 'Energy'),('dni','ghi_pyr'),
    ('dhi','ghi_pyr'),('Efficiency_Temp', 'Real Energy'),('Energy', 'Real Energy'),('ghi_pyr', 'Trieda'),
    ('dni', 'Trieda'),('dhi', 'Trieda'),('Efficiency_Temp', 'Trieda'),('Energy', 'Trieda'),
    ('Real Energy', 'Trieda'),('Energy', 'Error'),('Real Energy', 'Error')]
train_data, test_data = train_test_split(df_cleaned, test_size=0.2, random_state=42)
model = BayesianNetwork(edges)
model.fit(train_data, estimator=MaximumLikelihoodEstimator)
G = nx.DiGraph()
G.add_edges_from(edges)
if nx.is_directed_acyclic_graph(G):
    print("Sieť je acyklická (platná Bayesovská sieť).")
else:
    print("Sieť obsahuje cyklus! Nie je platná Bayesovská sieť.")
plt.figure(figsize=(24, 18))
pos = graphviz_layout(G,prog='neato')
if 'wind_from_direction' in pos:
    x, y = pos['wind_from_direction']
    pos['wind_from_direction'] = (x+15, y+20)
if 'air_temperature' in pos:
    x, y = pos['air_temperature']
    pos['air_temperature'] = (x, y+20)
if 'wind_speed' in pos:
    x, y = pos['wind_speed']
    pos['wind_speed'] = (x+30, y+30)
if 'Error' in pos:
    x, y = pos['Error']
    pos['Error'] = (x-60, y+60)
if 'dni' in pos:
    x, y = pos['dni']
    pos['dni'] = (x, y+10)

nx.draw(G, pos, with_labels=True, node_size=5000, node_color="skyblue",
        edge_color="black", font_size=12, font_weight='bold')
for node in model.nodes():
    try:
        cpd = model.get_cpds(node)
        lines = str(cpd).split('\n')
        cpt_text = '\n'.join(lines[1:6])
        x, y = pos[node]
        plt.text(x - 18, y - 11, cpt_text,
                 fontsize=10, fontweight='bold', family='monospace',
                 ha='left', va='top',
                 bbox=dict(facecolor='white', edgecolor='black',
                           boxstyle='round,pad=0.2', alpha=0.9))
    except Exception as e:
        print(f" CPT error @ {node}: {e}")

plt.title("Bayesovská sieť s CPT tabuľkami", fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.savefig("output/bayesova_siet_final.png", dpi=1200)
plt.show()

inference = VariableElimination(model)
filtered_test_data = test_data.dropna(subset=['wind_speed', 'air_temperature', 'ghi_pyr','relative_humidity','wind_from_direction','dni','dhi']).copy()
def predict_class(row):
    try:
        q = inference.query(variables=['Trieda'],evidence = {
    'ghi_pyr': row['ghi_pyr'],
    'air_temperature': row['air_temperature'],
    'wind_speed': row['wind_speed'],
    'relative_humidity': row['relative_humidity'],
    'wind_from_direction': row['wind_from_direction'],
    'dni': row['dni'],
    'dhi': row['dhi']
        })
        return q.values.argmax()
    except:
        return None

filtered_test_data['Predicted'] = filtered_test_data.apply(predict_class, axis=1)
filtered_test_data.dropna(subset=['Predicted'], inplace=True)

label_map = {'nizka': 0, 'stredna': 1, 'vysoka': 2}
filtered_test_data['Actual'] = filtered_test_data['Trieda'].map(label_map)
filtered_test_data['Predicted'] = filtered_test_data['Predicted'].astype(int)

accuracy = accuracy_score(filtered_test_data['Actual'], filtered_test_data['Predicted'])
precision = precision_score(filtered_test_data['Actual'], filtered_test_data['Predicted'], average='weighted')
conf_matrix = confusion_matrix(filtered_test_data['Actual'], filtered_test_data['Predicted'])

precision_per_class = precision_score(
    filtered_test_data['Actual'],
    filtered_test_data['Predicted'],
    average=None,
    labels = list(label_map.values())
)

print(f"Precision pre 'nizka' : {precision_per_class[0]:.4f}")
print(f"Precision pre 'stredna': {precision_per_class[1]:.4f}")
print(f"Precision pre 'vysoka' : {precision_per_class[2]:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['nizka', 'stredna', 'vysoka'],
            yticklabels=['nizka', 'stredna', 'vysoka'])
plt.xlabel("Predikovaná Trieda")
plt.ylabel("Skutočná Trieda")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("output/confMatrix_final.png", dpi=1200)
plt.show()



