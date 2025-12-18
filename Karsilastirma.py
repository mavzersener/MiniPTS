pip install catboost

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Kütüphane Kontrolleri
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import catboost as cb
except ImportError:
    cb = None

def train_and_evaluate_speed(df, target_col='AvgSpeed'):
    print(f"\n--- HIZ TAHMİN ANALİZİ BAŞLIYOR: {target_col} ---")

    # 1. Veri Hazırlığı
    data = df.copy()
    data['minute'] = data.index.minute
    data['hour'] = data.index.hour
    # Lag Features (Geçmiş değerler)
    for i in range(1, 4):
        data[f'lag_{i}'] = data[target_col].shift(i)
    data.dropna(inplace=True)

    # 2. Eğitim/Test Ayrımı (%80 - %20)
    split_point = int(len(data) * 0.8)
    train = data.iloc[:split_point]
    test = data.iloc[split_point:]

    features = ['minute', 'hour', 'lag_1', 'lag_2', 'lag_3']
    X_train = train[features]
    y_train = train[target_col]
    X_test = test[features]
    y_test = test[target_col]

    # SVM için Ölçekleme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. MODELLERİN TANIMLANMASI (Güncellendi)
    models = {
        # SVM: Süre kısıtlaması ile (max_iter=3000)
        'SVM': SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.1, max_iter=3000, cache_size=2000)
    }

    if xgb:
        # --- GÜNCELLENEN KISIM: OPTİMİZE XGBOOST ---
        # Grid Search'ten gelen en iyi parametreler:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=500,       # Grid Search sonucu
            learning_rate=0.01,     # Grid Search sonucu (Daha hassas öğrenme)
            max_depth=3,            # Grid Search sonucu (Daha sığ ağaç, ezberlemeyi önler)
            subsample=0.8,          # Grid Search sonucu
            colsample_bytree=0.8,   # Grid Search sonucu
            n_jobs=-1               # Tüm işlemcileri kullan
        )

    if lgb:
        # LightGBM: Hızlı olduğu için varsayılan ayarlara yakın, biraz ağaç sayısı yüksek
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, verbose=-1, n_jobs=-1)

    if cb:
        # CatBoost: Dengeli ayarlar
        models['CatBoost'] = cb.CatBoostRegressor(n_estimators=500, learning_rate=0.05, depth=6, verbose=0, thread_count=-1)

    results = []

    # 4. Eğitim Döngüsü
    for name, model in models.items():
        print(f"Model Eğitiliyor: {name}...")

        # Süre Ölçümü Başlat
        start_wall = time.time()
        start_cpu = time.process_time()

        try:
            if name == 'SVM':
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
        except Exception as e:
            print(f"Hata ({name}): {e}")
            continue

        # Süre Ölçümü Bitir
        elapsed_wall = time.time() - start_wall
        elapsed_cpu = time.process_time() - start_cpu

        # Negatif tahmin düzeltme
        pred = np.maximum(pred, 0)

        # Metrikler
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        # MAPE
        mask = y_test != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_test[mask] - pred[mask]) / y_test[mask])) * 100
        else:
            mape = np.nan

        results.append({
            'Algoritma': name,
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'MAPE (%)': round(mape, 2),
            'R2 Score': round(r2, 4),
            'Wall Time (sn)': round(elapsed_wall, 2), # Bekleme süresi
            'CPU Time (sn)': round(elapsed_cpu, 2)    # İşlemci yükü
        })

    return pd.DataFrame(results).sort_values(by='MAPE (%)')

# --- ANA PROGRAM ---
print("Veri işleniyor...")

try:
    # Dosya okuma
    df = pd.read_csv('simpleModified.csv', delimiter=';')
    start_date = pd.Timestamp('2024-01-01')
    df['Datetime'] = start_date + pd.to_timedelta(df['SimTime'], unit='s')
    df.set_index('Datetime', inplace=True)

    # Toplulaştırma
    df_agg = df.resample('1T').agg({'Hiz': 'mean'}).rename(columns={'Hiz': 'AvgSpeed'})
    df_agg['AvgSpeed'] = df_agg['AvgSpeed'].interpolate(method='time').fillna(method='bfill')

    # Analizi Çalıştır
    results_speed = train_and_evaluate_speed(df_agg, 'AvgSpeed')

    print("\n--- TRAFİK HIZI (SPEED) KARŞILAŞTIRMALI SONUÇLAR ---")
    print(results_speed.to_string(index=False))

    # CSV Kaydet
    results_speed.to_csv('Hiz_Tahmin_Optimize_Sonuclar.csv', index=False)
    print("\nSonuçlar kaydedildi.")

except Exception as e:
    print(f"Bir hata oluştu: {e}")

# Önce kütüphaneyi kurmadıysanız:
# !pip install catboost xgboost lightgbm

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Grafik Stili
plt.style.use('seaborn-v0_8-darkgrid') # veya 'ggplot'

# Kütüphane Kontrolleri
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import catboost as cb
except ImportError:
    cb = None

def train_and_evaluate_speed(df, target_col='AvgSpeed'):
    print(f"\n--- HIZ TAHMİN ANALİZİ BAŞLIYOR: {target_col} ---")

    # 1. Veri Hazırlığı
    data = df.copy()
    data['minute'] = data.index.minute
    data['hour'] = data.index.hour
    # Lag Features (Geçmiş değerler)
    for i in range(1, 4):
        data[f'lag_{i}'] = data[target_col].shift(i)
    data.dropna(inplace=True)

    # 2. Eğitim/Test Ayrımı (%80 - %20)
    split_point = int(len(data) * 0.8)
    train = data.iloc[:split_point]
    test = data.iloc[split_point:]

    features = ['minute', 'hour', 'lag_1', 'lag_2', 'lag_3']
    X_train = train[features]
    y_train = train[target_col]
    X_test = test[features]
    y_test = test[target_col]

    # SVM için Ölçekleme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. MODELLERİN TANIMLANMASI
    models = {
        'SVM': SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.1, max_iter=3000, cache_size=2000)
    }

    if xgb:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.01, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1
        )
    if lgb:
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, verbose=-1, n_jobs=-1)
    if cb:
        models['CatBoost'] = cb.CatBoostRegressor(n_estimators=500, learning_rate=0.05, depth=6, verbose=0, thread_count=-1)

    results = []
    predictions_dict = {} # Grafikler için tahminleri burada saklayacağız

    # 4. Eğitim Döngüsü
    for name, model in models.items():
        print(f"Model Eğitiliyor: {name}...")

        start_wall = time.time()
        start_cpu = time.process_time()

        try:
            if name == 'SVM':
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
        except Exception as e:
            print(f"Hata ({name}): {e}")
            continue

        elapsed_wall = time.time() - start_wall
        elapsed_cpu = time.process_time() - start_cpu

        # Negatif tahmin düzeltme ve Kaydetme
        pred = np.maximum(pred, 0)
        predictions_dict[name] = pred # Tahmini sakla

        # Metrikler
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        mask = y_test != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_test[mask] - pred[mask]) / y_test[mask])) * 100
        else:
            mape = np.nan

        results.append({
            'Algoritma': name,
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'MAPE (%)': round(mape, 2),
            'R2 Score': round(r2, 4),
            'Wall Time (sn)': round(elapsed_wall, 2),
            'CPU Time (sn)': round(elapsed_cpu, 2)
        })

    # --- 5. GRAFİK ÇİZİMİ (YENİ EKLENEN KISIM) ---
    print("Grafikler oluşturuluyor...")
    plt.figure(figsize=(15, 7))

    # Gerçek Veriyi Çiz
    plt.plot(y_test.index, y_test.values, label='Gerçek Veri (Real)', color='black', linewidth=2.5, alpha=0.8)

    # Modelleri Çiz
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1'] # Renk paleti
    for i, (name, preds) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(y_test.index, preds, label=f'{name} (R2: {r2_score(y_test, preds):.2f})',
                 linestyle='--', linewidth=1.5, color=color, alpha=0.9)

    #plt.title(f'Trafik Hız Tahmini Karşılaştırması\n({target_col})', fontsize=14)
    plt.xlabel('Zaman', fontsize=12)
    plt.ylabel('Hız (km/s)', fontsize=12)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results).sort_values(by='MAPE (%)')

# --- ANA PROGRAM ---
print("Veri işleniyor...")

try:
    # Dosya okuma
    df = pd.read_csv('simpleModified.csv', delimiter=';')
    start_date = pd.Timestamp('2024-01-01')
    df['Datetime'] = start_date + pd.to_timedelta(df['SimTime'], unit='s')
    df.set_index('Datetime', inplace=True)

    # Toplulaştırma
    df_agg = df.resample('1T').agg({'Hiz': 'mean'}).rename(columns={'Hiz': 'AvgSpeed'})
    df_agg['AvgSpeed'] = df_agg['AvgSpeed'].interpolate(method='time').fillna(method='bfill')

    # Analizi Çalıştır
    results_speed = train_and_evaluate_speed(df_agg, 'AvgSpeed')

    print("\n--- TRAFİK HIZI (SPEED) KARŞILAŞTIRMALI SONUÇLAR ---")
    print(results_speed.to_string(index=False))

    # CSV Kaydet
    results_speed.to_csv('Hiz_Tahmin_Optimize_Sonuclar.csv', index=False)
    print("\nSonuçlar kaydedildi.")

except Exception as e:
    print(f"Bir hata oluştu: {e}")