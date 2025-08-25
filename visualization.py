#@title Visualization

# --- Time per batch ---
plt.figure(figsize=(10,4))
plt.plot(flash_stats['time_per_batch'], label='Time per batch (s)')
plt.xlabel('Batch')
plt.ylabel('Seconds')
plt.title('Batch Time per Batch')
plt.legend()
plt.show()

# --- Peak memory per batch ---
plt.figure(figsize=(10,4))
plt.plot(flash_stats['peak_memory_mb'], label='Peak memory (MB)', color='orange')
plt.xlabel('Batch')
plt.ylabel('Memory (MB)')
plt.title('Peak GPU Memory Usage per Batch')
plt.legend()
plt.show()

# --- Average statistics ---
avg_time = sum(flash_stats['time_per_batch']) / len(flash_stats['time_per_batch'])
avg_mem = sum(flash_stats['peak_memory_mb']) / len(flash_stats['peak_memory_mb'])

# Print overall average runtime and memory usage
print(f"Average time per batch: {avg_time:.6f} sec")
print(f"Average peak memory usage: {avg_mem:.2f} MB")
