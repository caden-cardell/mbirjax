import csv
import matplotlib.pyplot as plt

def load_data(filter_fn):
    sino_gb, time, gb = [], [], []
    with open('logs/fdk_filter_results.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            v, r, c = int(row['views']), int(row['rows']), int(row['channels'])
            if filter_fn(v, r, c):
                sino_gb.append(v * r * c * 4 / 1e9)
                time.append(float(row['time']))
                gb.append(int(row['bytes']) / 1e9)
    paired = sorted(zip(sino_gb, time, gb))
    sino_gb, time, gb = zip(*paired)
    return sino_gb, time, gb

datasets = [
    ('dynamic views',    load_data(lambda v, r, c: r == 512 and c == 512)),
    ('dynamic rows',     load_data(lambda v, r, c: v == 512 and c == 512)),
    ('dynamic channels', load_data(lambda v, r, c: v == 512 and r == 512)),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for label, (sino_gb, time, gb) in datasets:
    ax1.plot(sino_gb, time, marker='o', markersize=3, label=label)
    ax2.plot(sino_gb, gb, marker='o', markersize=3, label=label)

ax1.set_xlabel('Sinogram Size (GB)')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Time vs Sinogram Size')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Sinogram Size (GB)')
ax2.set_ylabel('GB')
ax2.set_title('GB vs Sinogram Size')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logs/fdk_filter_combined.png', dpi=150)
print("Saved to logs/fdk_filter_combined.png")
