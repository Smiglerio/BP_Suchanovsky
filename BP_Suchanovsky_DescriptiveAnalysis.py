import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import sklearn.linear_model

file_path = "data/solar-measurementspakistanlahorewb-esmapqc.csv"
df = pd.read_csv(file_path)
df_cleaned = df[df["comments"] != "Power Supply Failure"].copy()
df_cleaned.drop(columns=["comments"], inplace=True)
columns_of_interest = ["ghi_pyr", "ghi_rsi", "dni", "dhi",
                       "air_temperature", "relative_humidity",
                       "wind_speed", "wind_from_direction"]
df_cleaned = df_cleaned.dropna(subset=columns_of_interest)

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 5))
fig.suptitle("Metriky Žiarenia", fontsize=16, fontweight='bold')

radiation_metrics = ["ghi_pyr", "ghi_rsi", "dni", "dhi"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for ax, col, color in zip(axes.flatten(), radiation_metrics, colors):
    sns.histplot(df_cleaned[col], bins=30, kde=True, color=color, ax=ax)
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel("Hodnota")
    ax.set_ylabel("Početnosť")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("output/MetrikyZiarenia.png", dpi=1200)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Enviromentálne Atribúty", fontsize=16, fontweight='bold')

environmental_attributes = ["air_temperature", "wind_speed", "relative_humidity", "wind_from_direction"]
colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

for ax, col, color in zip(axes.flatten(), environmental_attributes, colors):
    sns.histplot(df_cleaned[col], bins=30, kde=True, color=color, ax=ax)
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel("Hodnota")
    ax.set_ylabel("Početnosť")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("output/EnviromentalneAtributy.png", dpi=1200)
plt.show()

# Vypočítanie korelačných matíc
pearson_corr = df_cleaned[columns_of_interest].corr(method="pearson")
spearman_corr = df_cleaned[columns_of_interest].corr(method="spearman")

# Zvýšenie veľkosti obrázka pre korelačné matice
fig, axes = plt.subplots(2, 1, figsize=(18, 8))  # Zväčšenie figsize pre lepšiu čitateľnosť
fig.suptitle("Korelačné Matice", fontsize=16, fontweight='bold')

# Pearson korelačná matica
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0], linewidths=0.5, cbar_kws={'label': 'Korelačný koeficient'})
axes[0].set_title("Pearson Korelácia", fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=12)  # Otočenie názvov na osi X
axes[0].tick_params(axis='y', rotation=0, labelsize=12)  # Názvy na osi Y

# Spearman korelačná matica
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1], linewidths=0.5, cbar_kws={'label': 'Korelačný koeficient'})
axes[1].set_title("Spearman Korelácia", fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45, labelsize=12)  # Otočenie názvov na osi X
axes[1].tick_params(axis='y', rotation=0, labelsize=12)  # Názvy na osi Y

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Usporiadanie, aby sa tituly nezakrývali
plt.savefig("output/korelacnaMaticaBezAtribuvPDA.png", dpi=1200)
plt.show()

scatter_pairs = [("dni", "ghi_pyr"), ("dhi", "ghi_pyr"), ("dni", "ghi_rsi"),
                 ("dhi", "ghi_rsi"), ("relative_humidity", "air_temperature"), ("dni", "dhi")]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Vizualizácia Vybraných Atribútov s LOESS", fontsize=16, fontweight='bold')
for ax, (x, y) in zip(axes.flatten(), scatter_pairs):
    df_subset = df_cleaned[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    lower_bound, upper_bound = df_subset[x].quantile([0.01, 0.99])
    df_subset = df_subset[(df_subset[x] > lower_bound) & (df_subset[x] < upper_bound)]
    df_subset = df_subset.drop_duplicates(subset=[x])
    if df_subset.shape[0] > 10:
        print(f"Spracovanie {x} vs {y}: {df_subset.shape[0]} bodov")
        try:
            loess_result = lowess(df_subset[y], df_subset[x], frac=0.2)
            x_loess, y_loess = loess_result[:, 0], loess_result[:, 1]
            ax.scatter(df_subset[x], df_subset[y], alpha=0.2, label="Dáta")
            ax.plot(x_loess, y_loess, color='red', linewidth=2, linestyle='-', label="LOESS Krivka")
        except Exception as e:
            print(f"Chyba pri spracovaní {x} vs {y}: {e}")
    else:
        print(f"Preskočené {x} vs {y} - nedostatok dát")
    ax.set_title(f"{x} vs {y}", fontsize=12, fontweight='bold')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("output/LOESS_regresion_2D.png", dpi=1200)
plt.show()

triplets = [("dni", "ghi_pyr", "air_temperature"),
            ("dhi", "ghi_pyr", "wind_speed"),
            ("dni", "ghi_rsi", "relative_humidity"),
            ("dhi", "ghi_rsi", "wind_from_direction"),
            ("relative_humidity", "air_temperature", "wind_speed")]

fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={"projection": "3d"})
fig.suptitle("Lineárna regresia pre všetky trojice atribútov", fontsize=16, fontweight='bold')
axes = axes.flatten()
colors = ['r', 'g', 'b', 'c', 'm']

for i, (x_col, y_col, z_col) in enumerate(triplets):
    df_subset = df_cleaned[[x_col, y_col, z_col]].dropna()
    X = df_subset[[x_col, y_col]].values
    y = df_subset[z_col].values

    # Rozdelenie na trénovaciu a testovaciu množinu
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
    y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]

    # Tréning lineárnej regresie
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ax = axes[i]
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, marker='.', color=colors[i], label=f'{z_col} vs {x_col}, {y_col}')

    # Regresná rovina
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    xs, ys = np.meshgrid(xs, ys)
    zs = model.intercept_ + model.coef_[0] * xs + model.coef_[1] * ys
    ax.plot_surface(xs, ys, zs, alpha=0.3, color=colors[i])

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f"{z_col} vs {x_col}, {y_col}")

fig.delaxes(axes[-1])
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("output/linear_regression_3d.png", dpi=1200)
plt.show()
