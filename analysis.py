import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium

# ============================================================
# PHASE 1 : DATA SETUP AND AGGREGATION
# ============================================================

# Load Dataset
df = pd.read_csv("National Family Health Survey (NFHS-5) 2019-20.csv")

# Rename Columns
df.columns = [
    "SNo",
    "Indicator_Code",
    "Indicators",
    "Sub_Indicators",
    "Urban",
    "Rural",
    "Total",
    "NFHS4_Total",
    "State"
]

# Remove Extra Header Row
df = df.iloc[1:].reset_index(drop=True)

# Required Indicators
required_indicators = [

    "Women who are overweight or obese (BMI ≥25.0 kg/m2) (%)",

    "Men who are overweight or obese (BMI ≥25.0 kg/m2) (%)",

    "Children age 6-59 months who are anaemic (<11.0 g/dl) (%)",

    "All women age 15-49 years who are anaemic (%)",

    "Blood sugar level - high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%) - Women",

    "Blood sugar level - high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%) - Men",

    "Elevated blood pressure (Systolic ≥140 mm of Hg and/or Diastolic ≥90 mm of Hg) or taking medicine to control blood pressure (%) - Women",

    "Elevated blood pressure (Systolic ≥140 mm of Hg and/or Diastolic ≥90 mm of Hg) or taking medicine to control blood pressure (%) - Men"

]

# Filter
filtered_df = df[df["Sub_Indicators"].isin(required_indicators)]

# Pivot
pivot_df = filtered_df.pivot_table(
    index="State",
    columns="Sub_Indicators",
    values="Total",
    aggfunc="first"
).reset_index()

# Rename Columns
pivot_df.rename(columns={

    "All women age 15-49 years who are anaemic (%)":"Anaemia_Women",

    "Children age 6-59 months who are anaemic (<11.0 g/dl) (%)":"Anaemia_Children",

    "Women who are overweight or obese (BMI ≥25.0 kg/m2) (%)":"Obesity_Women",

    "Men who are overweight or obese (BMI ≥25.0 kg/m2) (%)":"Obesity_Men",

    "Blood sugar level - high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%) - Women":"BloodSugar_Women",

    "Blood sugar level - high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%) - Men":"BloodSugar_Men",

    "Elevated blood pressure (Systolic ≥140 mm of Hg and/or Diastolic ≥90 mm of Hg) or taking medicine to control blood pressure (%) - Women":"Hypertension_Women",

    "Elevated blood pressure (Systolic ≥140 mm of Hg and/or Diastolic ≥90 mm of Hg) or taking medicine to control blood pressure (%) - Men":"Hypertension_Men"

}, inplace=True)

# Convert Numeric
numeric_cols = pivot_df.columns.drop("State")

for col in numeric_cols:
    pivot_df[col] = pd.to_numeric(pivot_df[col], errors="coerce")

pivot_df[numeric_cols] = pivot_df[numeric_cols].fillna(
    pivot_df[numeric_cols].mean()
)

# Gender Obesity Gap
pivot_df["Gender_Obesity_Gap"] = (
    pivot_df["Obesity_Women"] -
    pivot_df["Obesity_Men"]
)

print("\nPHASE 1 COMPLETED")
print(pivot_df.head())


# ============================================================
# PHASE 2 : EXPLORATORY DATA ANALYSIS
# ============================================================

# ---------------- Correlation ----------------

correlation = pivot_df["Anaemia_Children"].corr(
    pivot_df["Obesity_Women"]
)

print("\nPearson Correlation =", round(correlation,3))


# ---------------- Bar Chart ----------------

top10 = pivot_df.sort_values(
    by="Hypertension_Women",
    ascending=False
).head(10)

x = np.arange(len(top10))
width = 0.35

plt.figure(figsize=(12,6))

plt.bar(
    x-width/2,
    top10["Hypertension_Men"],
    width,
    label="Men"
)

plt.bar(
    x+width/2,
    top10["Hypertension_Women"],
    width,
    label="Women"
)

plt.xticks(
    x,
    top10["State"],
    rotation=45,
    ha="right"
)

plt.xlabel("States")
plt.ylabel("Hypertension (%)")
plt.title("Top 10 States: Hypertension in Men vs Women")
plt.legend()

plt.tight_layout()

plt.savefig("Hypertension_BarChart.png")

plt.show()


# ---------------- Scatter Plot ----------------

plt.figure(figsize=(10,8))

plt.scatter(
    pivot_df["Anaemia_Women"],
    pivot_df["Obesity_Women"],
    s=70
)

for i in range(len(pivot_df)):
    plt.text(
        pivot_df["Anaemia_Women"].iloc[i],
        pivot_df["Obesity_Women"].iloc[i],
        pivot_df["State"].iloc[i][:3],
        fontsize=8
    )

plt.xlabel("Percentage of Anaemic Women (%)")
plt.ylabel("Percentage of Obese Women (%)")
plt.title("Anaemia vs Obesity among Women")

plt.grid(True)

plt.tight_layout()

plt.savefig("Scatter_Plot.png")

plt.show()


# ============================================================
# PHASE 3 : FOLIUM MAP
# ============================================================

india_map = folium.Map(
    location=[20.5937,78.9629],
    zoom_start=5
)
pivot_df["State"] = pivot_df["State"].replace({
    "Andaman & Nicobar Islands": "Andaman and Nicobar",
    "Dadra & Nagar Haveli and Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "NCT of Delhi": "Delhi",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Odisha": "Orissa"
})

# Choropleth
folium.Choropleth(

    geo_data="india_state.geojson",

    data=pivot_df,

    columns=["State","Obesity_Women"],

    key_on="feature.properties.NAME_1",

    fill_color="YlOrRd",

    fill_opacity=0.7,

    line_opacity=0.3,

    legend_name="Women Obesity (%)"

).add_to(india_map)

# Basic state markers (using approximate map center)
# ============================================================
# STATE CAPITAL COORDINATES
# ============================================================

state_capitals = {

    "Andaman and Nicobar":[11.6234,92.7265],
    "Andhra Pradesh":[15.9129,79.7400],
    "Arunachal Pradesh":[27.0844,93.6053],
    "Assam":[26.1433,91.7898],
    "Bihar":[25.5941,85.1376],
    "Chandigarh":[30.7333,76.7794],
    "Chhattisgarh":[21.2514,81.6296],
    "Dadra and Nagar Haveli and Daman and Diu":[20.3974,72.8328],
    "Delhi":[28.6139,77.2090],
    "Goa":[15.4909,73.8278],
    "Gujarat":[23.2156,72.6369],
    "Haryana":[30.7333,76.7794],
    "Himachal Pradesh":[31.1048,77.1734],
    "Jammu and Kashmir":[34.0837,74.7973],
    "Jharkhand":[23.3441,85.3096],
    "Karnataka":[12.9716,77.5946],
    "Kerala":[8.5241,76.9366],
    "Ladakh":[34.1526,77.5770],
    "Lakshadweep":[10.5667,72.6417],
    "Madhya Pradesh":[23.2599,77.4126],
    "Maharashtra":[19.0760,72.8777],
    "Manipur":[24.8170,93.9368],
    "Meghalaya":[25.5788,91.8933],
    "Mizoram":[23.7271,92.7176],
    "Nagaland":[25.6751,94.1086],
    "Odisha":[20.2961,85.8245],
    "Puducherry":[11.9416,79.8083],
    "Punjab":[30.7333,76.7794],
    "Rajasthan":[26.9124,75.7873],
    "Sikkim":[27.3389,88.6065],
    "Tamil Nadu":[13.0827,80.2707],
    "Telangana":[17.3850,78.4867],
    "Tripura":[23.8315,91.2868],
    "Uttar Pradesh":[26.8467,80.9462],
    "Uttarakhand":[30.3165,78.0322],
    "West Bengal":[22.5726,88.3639]

}
# ============================================================
# CIRCLE MARKERS
# ============================================================

for _, row in pivot_df.iterrows():

    state = row["State"]

    if state in state_capitals:

        folium.CircleMarker(

            location=state_capitals[state],

            radius=8,

            color="blue",

            fill=True,

            fill_color="red",

            fill_opacity=0.8,

            tooltip=state,

           popup=folium.Popup(f"""
<h3>{state}</h3>

<table style="width:220px">
<tr><td><b>Blood Sugar (Women)</b></td><td>{row['BloodSugar_Women']}%</td></tr>
<tr><td><b>Blood Sugar (Men)</b></td><td>{row['BloodSugar_Men']}%</td></tr>
<tr><td><b>Anaemia (Women)</b></td><td>{row['Anaemia_Women']}%</td></tr>
<tr><td><b>Anaemia (Children)</b></td><td>{row['Anaemia_Children']}%</td></tr>
<tr><td><b>Obesity (Women)</b></td><td>{row['Obesity_Women']}%</td></tr>
<tr><td><b>Obesity (Men)</b></td><td>{row['Obesity_Men']}%</td></tr>
</table>
""", max_width=300)

        ).add_to(india_map)
folium.LayerControl().add_to(india_map)

india_map.save("India_Health_Dashboard.html")

print("\nPHASE 3 COMPLETED")
print("Dashboard saved as India_Health_Dashboard.html")

# Correlation Matrix
correlation_matrix = pivot_df[
    [
        "Anaemia_Children",
        "Anaemia_Women",
        "Obesity_Women",
        "Obesity_Men",
        "BloodSugar_Women",
        "BloodSugar_Men",
        "Hypertension_Women",
        "Hypertension_Men",
        "Gender_Obesity_Gap"
    ]
].corr()

print(correlation_matrix.head(3))