import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("global_country_boundaries.gpkg")

fig, ax = plt.subplots(figsize=(32, 16))
gdf[gdf.source == "nuts"].plot(ax=ax, color="tab:blue", edgecolor="black", linewidth=0.2)
gdf[gdf.source == "admin"].plot(ax=ax, color="tab:orange", edgecolor="black", linewidth=0.2)
ax.set_title(f"{len(gdf)} regions - blue=NUTS (Europe), orange=Natural Earth (rest of world)")
plt.savefig("global_country_boundaries_preview.png", dpi=600, bbox_inches="tight")
