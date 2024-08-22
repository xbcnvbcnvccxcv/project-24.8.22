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
X2 = df['3.Highest Temperature (℃)']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')

# 그래프에 제목 추가
plt.title("Highest Temperature vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Highest Temperature (℃)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Highest_Temperature.png'))
plt.show()

