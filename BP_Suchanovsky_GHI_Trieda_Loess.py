import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns

df = pd.read_csv("data/BP_BayesReadyData.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ghi_pyr', 'Error', 'Real Energy'])

def classify_energy(val):
    if val < 215:
        return 'nizka'
    elif val <= 315:
        return 'stredna'
    else:
        return 'vysoka'

df['Trieda'] = df['Real Energy'].apply(classify_energy)
triedy = ['nizka', 'stredna', 'vysoka']
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
fig.suptitle("LOESS regresia: ghi_pyr vs Error podľa triedy", fontsize=16, fontweight='bold')

for ax, trieda in zip(axes, triedy):
    df_subset = df[df['Trieda'] == trieda][['ghi_pyr', 'Error']].dropna()
    print(f"{trieda} – dostupných riadkov: {len(df_subset)}")

    if df_subset.shape[0] > 10:
        x = df_subset['ghi_pyr'].astype(float)
        y = df_subset['Error'].astype(float)

        try:
            frac = 0.3 if trieda == 'vysoka' else 0.5 if trieda == 'stredna' else 0.7
            loess_result = lowess(y, x, frac=frac, return_sorted=True)
            x_loess, y_loess = loess_result[:, 0], loess_result[:, 1]

            ax.scatter(x, y, alpha=0.3, label='Dáta', s=10)
            ax.plot(x_loess, y_loess, color='red', linewidth=2, label='LOESS krivka')

        except Exception as e:
            print(f" Chyba pri LOESS pre triedu {trieda}: {e}")
    else:
        print(f" Trieda {trieda} – nedostatok dát")

    ax.set_title(f"Trieda: {trieda}", fontsize=12, fontweight='bold')
    ax.set_xlabel("ghi_pyr")
    ax.set_ylabel("Error")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("output/lOESS_GHI_pyrTrieda.png", dpi=1200)
plt.show()

for trieda in triedy:
    df_subset = df[df['Trieda'] == trieda]
    rmse = np.sqrt(np.mean(df_subset['Error']**2))
    print(f" RMSE pre triedu '{trieda}': {rmse:.4f}")

rmse_total = np.sqrt(np.mean(df['Error']**2))
print(f" RMSE bez klasifikácie: {rmse_total:.4f}")

columns_of_interest = ["ghi_pyr", "ghi_rsi", "dni", "dhi",
                       "air_temperature", "relative_humidity",
                       "wind_speed", "wind_from_direction",
                       "Energy", "Efficiency_Temp", "Real Energy", "Error"]

pearson_corr = df[columns_of_interest].corr(method="pearson")
spearman_corr = df[columns_of_interest].corr(method="spearman")

fig, axes = plt.subplots(2, 1, figsize=(18, 8))
fig.suptitle("Korelačné Matice", fontsize=16, fontweight='bold')

sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0], linewidths=0.5, cbar_kws={'label': 'Korelačný koeficient'})
axes[0].set_title("Pearson Korelácia", fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=12)
axes[0].tick_params(axis='y', rotation=0, labelsize=12)

sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1], linewidths=0.5, cbar_kws={'label': 'Korelačný koeficient'})
axes[1].set_title("Spearman Korelácia", fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45, labelsize=12)
axes[1].tick_params(axis='y', rotation=0, labelsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("output/korrMaticeVrataneAtributovPDA.png", dpi=1200)
plt.show()
