import csv
import matplotlib.pyplot as plt

def load_data(path):
    views, time, gb = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['rows']) == 512 and int(row['channels']) == 512:
                views.append(int(row['views']))
                time.append(float(row['time']))
                gb.append(int(row['bytes']) / 1024**3)
    paired = sorted(zip(views, time, gb))
    return zip(*paired)

datasets = [
    ('old', load_data('logs/fdk_filter_results.csv')),
    ('new', load_data('../fdk-filter-new/logs/fdk_filter_results.csv')),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for label, (views, time, gb) in datasets:
    ax1.plot(views, time, marker='o', markersize=3, label=label)
    ax2.plot(views, gb, marker='o', markersize=3, label=label)

ax1.set_xlabel('Number of Views')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Time vs Views (rows=512, channels=512)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Number of Views')
ax2.set_ylabel('GB')
ax2.set_title('GB vs Views (rows=512, channels=512)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logs/fdk_filter_rows512_channels512.png', dpi=150)
print("Saved to logs/fdk_filter_rows512_channels512.png")
