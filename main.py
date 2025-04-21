import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm

# Leer datos
dataset = pd.read_csv('manufacturing.csv')

# Extraer X y y con los nombres de las columnas
X = dataset[['Material Fusion Metric']].values
y = dataset['Quality Rating'].values

# Escalar X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Regresión Lineal**
lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y)

# Predicciones para regresión lineal
y_pred_lin = lin_reg.predict(X_scaled)

# Calcular métricas para regresión lineal
r2_lin = r2_score(y, y_pred_lin)  # R²
r_lin = np.sqrt(r2_lin) if lin_reg.coef_[0] > 0 else -np.sqrt(r2_lin)  # R
mae_lin = mean_absolute_error(y, y_pred_lin)  # MAE

# Calcular p-valor con statsmodels
X_sm = sm.add_constant(X_scaled)  # Añadir intercepto
model_lin = sm.OLS(y, X_sm).fit()
p_value_lin = model_lin.pvalues[1]  # P-valor del coeficiente

# **Regresión Polinómica**
degree = 10
poly_reg = PolynomialFeatures(degree=degree)
X_poly = poly_reg.fit_transform(X_scaled)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Predicciones para regresión polinómica
y_pred_poly = lin_reg_2.predict(X_poly)

# Calcular métricas para regresión polinómica
r2_poly = r2_score(y, y_pred_poly)  # R²
r_poly = np.corrcoef(y, y_pred_poly)[0, 1]  # R (correlación entre y y predicciones)
mae_poly = mean_absolute_error(y, y_pred_poly)  # MAE

# Calcular p-valor con statsmodels (usamos F-test para el modelo general)
X_poly_sm = sm.add_constant(X_poly)
model_poly = sm.OLS(y, X_poly_sm).fit()
p_value_poly = model_poly.f_pvalue  # P-valor del F-test

# **Imprimir resultados**
print("Regresión Lineal:")
print(f"R²: {r2_lin:.4f}")
print(f"R: {r_lin:.4f}")
print(f"MAE: {mae_lin:.4f}")
print(f"P-valor: {p_value_lin:.4f}")

print("\nRegresión Polinómica (grado 10):")
print(f"R²: {r2_poly:.4f}")
print(f"R: {r_poly:.4f}")
print(f"MAE: {mae_poly:.4f}")
print(f"P-valor (F-test): {p_value_poly:.4f}")

# **Visualizar regresión lineal**
plt.scatter(X_scaled, y, color='red')
plt.plot(X_scaled, y_pred_lin, color='blue')
plt.title('Linear Regression (Scaled)')
plt.xlabel('Material Fusion Metric (scaled)')
plt.ylabel('Quality Rating')
plt.show()

# **Visualizar regresión polinómica**
X_grid = np.arange(min(X_scaled), max(X_scaled), 0.01).reshape(-1, 1)
X_grid_poly = poly_reg.transform(X_grid)
plt.scatter(X_scaled, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(X_grid_poly), color='blue')
plt.title('Polynomial Regression (Scaled, degree=10)')
plt.xlabel('Material Fusion Metric (scaled)')
plt.ylabel('Quality Rating')
plt.show()