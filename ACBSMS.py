import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. CSV 파일 불러오기
suicide_rate_usa_df = pd.read_csv('suicide_rate_USA.csv')
social_media_impact_df = pd.read_csv('social_media_impact_on_suicide_rates.csv')

# 2. 데이터 전처리
# 자살률 데이터에서 Nation이 'USA'인 행을 선택하고, Total 열만 추출
usa_suicide_rates = suicide_rate_usa_df[suicide_rate_usa_df['Nation'] == 'USA'].iloc[:, 1::3].astype(float).values[0]  # Total suicide rates for each year (2010-2019)

# social-media 데이터에서 연도별 BTSX의 Suicide Rate % change, Twitter user change, Facebook user change를 필터링
filtered_data = social_media_impact_df[social_media_impact_df['sex'] == 'BTSX']
filtered_years = filtered_data['year'].values
filtered_twitter_user_changes = filtered_data['Twitter user count % change since 2010'].values
filtered_facebook_user_changes = filtered_data['Facebook user count % change since 2010'].values

# 3. **Groupby를 사용한 데이터 그룹화 후 간단한 통계 분석**
# 그룹화 기준: sex
grouped_social_media = social_media_impact_df.groupby('sex')

# Suicide Rate % change, Twitter user count % change, Facebook user count % change의 평균, 최대값, 표준편차 계산
grouped_stats = grouped_social_media[['Suicide Rate % change since 2010', 
                                      'Twitter user count % change since 2010', 
                                      'Facebook user count % change since 2010']].agg(['mean', 'max', 'std'])

# Groupby 결과 출력
print("**Groupby 통계 분석 결과**")
print(grouped_stats)

# 4. **그래프 1: 연도별 자살률 변화 추이 (2010-2019)**
plt.figure(figsize=(10, 6))
plt.plot(range(2010, 2020), usa_suicide_rates, marker='o', linestyle='-', color='b', label='Total Suicide Rate')
plt.title('Total Suicide Rate in the USA (2010-2019)')
plt.xlabel('Year')
plt.ylabel('Suicide Rate')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(2010, 2020))
plt.legend()
plt.show()

# 5. **그래프 2: 트위터 사용자 변화와 자살률 변화 간의 관계 시각화 (산점도)**
plt.figure(figsize=(10, 6))
sns.scatterplot(x=filtered_twitter_user_changes, y=usa_suicide_rates, hue=range(2010, 2020), palette='coolwarm', s=100)
plt.title('Relationship between Twitter User Changes and Suicide Rates (2010-2019)')
plt.xlabel('Twitter User Change % (since 2010)')
plt.ylabel('Suicide Rate in the USA')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 6. **그래프 3: 페이스북 사용자 변화와 자살률 변화 간의 관계 시각화 (산점도)**
plt.figure(figsize=(10, 6))
sns.scatterplot(x=filtered_facebook_user_changes, y=usa_suicide_rates, hue=range(2010, 2020), palette='viridis', s=100)
plt.title('Relationship between Facebook User Changes and Suicide Rates (2010-2019)')
plt.xlabel('Facebook User Change % (since 2010)')
plt.ylabel('Suicide Rate in the USA')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 7. **머신러닝 모델 학습**
# 데이터 통합
# 독립 변수 (X): 트위터 사용자 변화율, 페이스북 사용자 변화율
# 종속 변수 (y): 자살률
X = np.column_stack((filtered_twitter_user_changes, filtered_facebook_user_changes))
y = usa_suicide_rates

# 데이터셋 분할 (80% 학습용, 20% 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 회귀 모델 사용 (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# 8. **모델 예측**
y_pred = model.predict(X_test)

# 9. **모델 평가**
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 모델 성능 지표 출력
print("\n**모델 평가 결과**")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-Squared (R2): {r2:.4f}")

# 10. **예측 결과 시각화**
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Suicide Rate')
plt.ylabel('Predicted Suicide Rate')
plt.title('Actual vs Predicted Suicide Rates (Linear Regression)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()