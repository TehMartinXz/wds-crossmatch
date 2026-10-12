import pandas as pd

ruta_mio = "./outputs/components_crossmatch_bayesian.csv"
ruta_colega = "./referencia_extended/components_crossmatch_bayesian.csv"

df_mio = pd.read_csv(ruta_mio)
df_colega = pd.read_csv(ruta_colega)

merged = pd.merge(
    df_mio, 
    df_colega, 
    on=["wds_id", "component"], 
    suffixes=('_mio', '_colega')
)

mio_ids = pd.to_numeric(merged["assigned_source_id_mio"], errors="coerce")
colega_ids = pd.to_numeric(merged["assigned_source_id_colega"], errors="coerce")

match_mask = (mio_ids == colega_ids) | (mio_ids.isna() & colega_ids.isna())
discrepancies = merged[~match_mask]

print(f"Total de discrepancias a revisar: {len(discrepancies)}")
# Mostramos las columnas clave para evaluar los casos borde
print(discrepancies[['wds_id', 'component', 'assigned_source_id_mio', 'assigned_source_id_colega', 'best_posterior_prob_mio', 'best_posterior_prob_colega']].head(10))