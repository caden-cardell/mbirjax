import csv
import matplotlib.pyplot as plt

views, time, gb = [], [], []

with open('output/fdk_filter_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['rows']) == 512 and int(row['channels']) == 512:
            views.append(int(row['views']))
            time.append(float(row['time']))
            gb.append(int(row['bytes']) / 1024**3)

paired = sorted(zip(views, time, gb))
views, time, gb = zip(*paired)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(views, time, marker='o', markersize=3)
ax1.set_xlabel('Number of Views')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Time vs Views (rows=512, channels=512)')
ax1.grid(True, alpha=0.3)

ax2.plot(views, gb, marker='o', markersize=3, color='orange')
ax2.set_xlabel('Number of Views')
ax2.set_ylabel('GB')
ax2.set_title('GB vs Views (rows=512, channels=512)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logs/fdk_filter_rows512_channels512.png', dpi=150)
print(f"Saved to logs/fdk_filter_512_512.png ({len(views)} data points)")
