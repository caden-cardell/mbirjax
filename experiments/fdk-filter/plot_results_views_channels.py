import csv
import matplotlib.pyplot as plt

def load_data(path):
    rows, time, gb = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['views']) == 512 and int(row['channels']) == 512:
                rows.append(int(row['rows']))
                time.append(float(row['time']))
                gb.append(int(row['bytes']) / 1024**3)
    paired = sorted(zip(rows, time, gb))
    return zip(*paired)

datasets = [
    ('old', load_data('logs/fdk_filter_results.csv')),
    ('new', load_data('../fdk-filter-new/logs/fdk_filter_results.csv')),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for label, (rows, time, gb) in datasets:
    ax1.plot(rows, time, marker='o', markersize=3, label=label)
    ax2.plot(rows, gb, marker='o', markersize=3, label=label)

ax1.set_xlabel('Number of Rows')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Time vs Rows (views=512, channels=512)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Number of Rows')
ax2.set_ylabel('GB')
ax2.set_title('GB vs Rows (views=512, channels=512)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logs/fdk_filter_views512_channels512.png', dpi=150)
print("Saved to logs/fdk_filter_views512_channels512.png")