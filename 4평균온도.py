# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 1. CSV 파일 읽기
# df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')
#
# # 2. 새로운 헤더 설정
# header_english = [
#     '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
#     '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
#     '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
#     '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
#     '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
#     '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
#     '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
#     '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
#     '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
#     '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
# ]
# df.columns = header_english
#
# # X와 Y 설정
# X2 = df['4.Average Temperature (℃)']
# Y2 = df['2.Total Transported Patients']
#
# # result 디렉토리 생성
# output_dir = './result'
# os.makedirs(output_dir, exist_ok=True)
#
# # Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
# plt.figure(figsize=(10, 6))
# sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')
#
# # 그래프에 제목 추가
# plt.title("Average Temperature vs Number of Heatstroke Patients with Regression Line")
# plt.xlabel("Average Temperature (℃)")
# plt.ylabel("Number of Heatstroke Patients")
# plt.grid(True)
#
# # 결과를 파일로 저장
# plt.savefig(os.path.join(output_dir, 'Average_Temperature_vs_Heatstroke_Patients_with_Regression.png'))
# plt.show()


# 로지스틱 회귀 그래프

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# X와 Y 설정
X2 = df['4.Average Temperature (℃)']
Y2 = df['2.Total Transported Patients']

# Y를 이진 변수로 변환 (예: 열사병 환자 수가 50명 이상이면 1, 그렇지 않으면 0)
# 도메인 지식: 열사병과 같은 질병의 경우, 환자 수가 특정 수치를 넘으면 상황이 심각하다고 판단할 수 있습니다.
# 50명이라는 기준은 이러한 상황을 반영하여, 병원이나 응급 서비스가 대응해야 하는 임계점을 설정할 수 있습니다.
# 데이터의 자연적 구분
# 구간의 구분: 환자 수가 매우 적거나 많은 경우, 두 집단의 차이가 뚜렷하게 나지 않을 수 있습니다.
# 50명을 기준으로 설정하면, 환자 수가 적은 그룹과 많은 그룹을 명확히 구분할 수 있습니다.
# 이는 로지스틱 회귀와 같은 모델에서 두 집단을 효과적으로 비교하고 분석하는 데 도움이 됩니다.
threshold = 50
Y_binary = (Y2 >= threshold).astype(int)

# X에 상수항 추가 (로지스틱 회귀 모델에 필요)
X2 = sm.add_constant(X2)

# 로지스틱 회귀 모델 적합
model = sm.Logit(Y_binary, X2)
result = model.fit()

# 예측값 계산
predictions = result.predict(X2)

# 결과를 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X2['4.Average Temperature (℃)'], y=Y_binary, color='blue', label='Observed')
sns.lineplot(x=X2['4.Average Temperature (℃)'], y=predictions, color='red', label='Logistic Regression Fit')

# 그래프에 제목 추가
plt.title("Average Temperature vs Number of Heatstroke Patients with Logistic Regression Line")
plt.xlabel("Average Temperature (℃)")
plt.ylabel("Probability of Heatstroke (Binary Outcome)")
plt.grid(True)
plt.legend()

# 결과를 파일로 저장
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'Average_Temperature_vs_Heatstroke_Patients_with_Logistic_Regression.png'))
plt.show()
