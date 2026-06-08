import csv
import matplotlib.pyplot as plt

rows, time, gb = [], [], []

with open('logs/fdk_filter_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['views']) == 512 and int(row['channels']) == 512:
            rows.append(int(row['rows']))
            time.append(float(row['time']))
            gb.append(int(row['bytes']) / 1e9)

paired = sorted(zip(rows, time, gb))
rows, time, gb = zip(*paired)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(rows, time, marker='o', markersize=3)
ax1.set_xlabel('Number of Rows')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Time vs Rows (views=512, channels=512)')
ax1.grid(True, alpha=0.3)

ax2.plot(rows, gb, marker='o', markersize=3, color='orange')
ax2.set_xlabel('Number of Rows')
ax2.set_ylabel('GB')
ax2.set_title('GB vs Rows (views=512, channels=512)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logs/fdk_filter_views512_channels512.png', dpi=150)
print(f"Saved to logs/fdk_filter_views512_channels512.png ({len(rows)} data points)")
