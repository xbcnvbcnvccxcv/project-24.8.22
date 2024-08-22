import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

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
# X2 = df['3최고기온']
# Y2 = df['2이송 인원']

#데이터에 사용할 컬럼(속성) 이름 지정
column_names = ['1.Date', '3.Highest Temperature (℃)', '4.Average Temperature (℃)','9.Average Humidity (%)',
                '10.Total Precipitation (mm)', '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)',
                '20.Discomfort Index', '39.Transported Patients on Previous Day']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)


df[column_names].plot(kind='density', figsize=(12, 10), subplots=True, layout=(3, 3), sharex=False)

# 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X2, Y2, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)

#그래프에 제목 추가
# plt.title("Highest Temperature vs Number of Heatstroke Patients")
# plt.xlabel("Highest Temperature (℃)")
# plt.ylabel("Number of Heatstroke Patients")
## 범래 추가
# plt.legend()
# #그래프 표시
# plt.grid(True)
plt.savefig(os.path.join(output_dir, 'density_plots.png'))
plt.show()

