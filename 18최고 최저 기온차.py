import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
X2 = df['18.Temperature Difference (High-Low)']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')

# 그래프에 제목 추가
plt.title("Temperature Differencevs vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Temperature Difference (High-Low)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Temperature_Difference.png'))
plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
X2 = df['18.Temperature Difference (High-Low)']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# 히스토그램 그리기
plt.figure(figsize=(14, 6))

# 서브플롯 생성
plt.subplot(1, 2, 1)
sns.histplot(X2, kde=True, color='blue', bins=30)
plt.title("Histogram of Temperature Difference (High-Low)")
plt.xlabel("Temperature Difference (High-Low)")
plt.ylabel("Frequency")
plt.grid(True)

plt.subplot(1, 2, 2)
sns.histplot(Y2, kde=True, color='green', bins=30)
plt.title("Histogram of Number of Heatstroke Patients")
plt.xlabel("Number of Heatstroke Patients")
plt.ylabel("Frequency")
plt.grid(True)

# 그래프에 제목 추가
plt.suptitle("Histograms of Temperature Difference and Number of Heatstroke Patients")

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Histograms_Temperature_Difference_and_Heatstroke_Patients.png'))
plt.show()
#왼쪽 그래프는 온도 차이(하루 최고온도와 최저온도 간의 차이)의 분포를 보여주고, 오른쪽 그래프는 열사병 환자 수의 분포를 보여줍니다.
# 이 두 그래프를 나란히 보여줌으로써, 온도 차이와 열사병 환자 수 간의 잠재적인 상관관계를 쉽게 파악할 수 있습니다.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
X = df['18.Temperature Difference (High-Low)']
Y = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# 히스토그램 그리기
plt.figure(figsize=(10, 6))

# 히스토그램 그리기
sns.histplot(x=X, y=Y, bins=30, cmap="Blues", cbar=True)

# 그래프 제목 및 축 레이블 설정
plt.title("Histogram of Heatstroke Patients by Temperature Difference")
plt.xlabel("Temperature Difference (High-Low)")
plt.ylabel("Number of Heatstroke Patients")

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Heatstroke_Patients_by_Temperature_Difference.png'))
plt.show()

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
# X2 = df['18.Temperature Difference (High-Low)']
# Y2 = df['2.Total Transported Patients']
#
# # result 디렉토리 생성
# output_dir = './result'
# os.makedirs(output_dir, exist_ok=True)
#
# # 히스토그램과 선그래프 그리기
# plt.figure(figsize=(10, 6))
#
# # 히스토그램 (온도차이)
# sns.histplot(X2, kde=False, color='blue', bins=30, label='Temperature Difference (High-Low)')
#
# # y축을 오른쪽에 추가 (열사병 환자 수 분포 선그래프)
# plt.twinx()
# sns.kdeplot(Y2, color='red', linewidth=2, label='Heatstroke Patients Density')
#
# # x축 범위 0~50으로 제한
# plt.xlim(0, 50)
#
# # 그래프 제목 및 축 레이블 설정
# plt.title("Temperature Difference and Heatstroke Patients Distribution (Zoomed: 0-50)")
# plt.xlabel("Temperature Difference (High-Low)")
# plt.ylabel("Frequency / Density")
#
# # 범례 추가
# plt.legend(loc="upper right")
#
# # 결과를 파일로 저장
# plt.savefig(os.path.join(output_dir, 'Temperature_Difference_vs_Heatstroke_Patients_Zoomed.png'))
# plt.show()

