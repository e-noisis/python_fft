import csv
import numpy as np
import matplotlib.pyplot as plt

# Open the CSV file
with open('20K_80Hz_8KSPs.csv', 'r') as file:
    # Read the CSV file
    csv_reader = csv.reader(file)
    # Extract all rows from the CSV
    data = list(csv_reader)
# Get the number of rows in the data
data_length = len(data)
# Extract the 5th column
column_5 = [float(row[4]) for row in data]
# Define the sampling rate
sampling_rate = 8000  # 8 KSPs (Kilo Samples Per Second)
# Generate the frequency axis for FFT
fft_freqs = np.fft.fftfreq(data_length, 1 / sampling_rate)
positive_freqs_mask = fft_freqs >= 0
# Calculate FFT of the 5th column
fft_result = np.fft.fft(column_5)
# Calculate amplitude in dB
amplitude_dB = 20 * np.log10(np.abs(fft_result) / np.max(np.abs(fft_result)))
# Find the indices of the first 10 peak amplitudes
peak_indices = np.argsort(np.abs(fft_result))[-10:]
peak_frequencies = fft_freqs[peak_indices]
peak_amplitudes = amplitude_dB[peak_indices]

# Print the peak values to the terminal
print("Top 10 Peak Frequencies and Amplitudes:")
print(f"{'Frequency (Hz)':>15} | {'Amplitude (dB)':>15}")
print("-" * 33)
for freq, amp in zip(peak_frequencies, peak_amplitudes):
    print(f"{freq:>15.2f} | {amp:>15.2f}")
    
# Plot FFT of the 5th column
plt.figure(figsize=(12, 6))
plt.plot(fft_freqs[positive_freqs_mask], amplitude_dB[positive_freqs_mask])
plt.title('FFT of Reference Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.ylim(-140, 0)  # Set y-axis range from -140 dB to 0 dB
plt.grid(True)

"""
# Add table for peak values
cell_text = [['Frequency (Hz)', 'Amplitude (dB)']]
for freq, amp in zip(peak_frequencies, peak_amplitudes):
    cell_text.append([f'{freq:.2f}', f'{amp:.2f}'])
table = plt.table(cellText=cell_text,
                  colLabels=None,
                  cellLoc='center',
                  loc='right',
                  bbox=[1.2, 0.1, 0.3, 0.8],
                  edges='open')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.grid(True)
plt.show()
"""

col_labels=['Fr[Hz]','A[dB]']
row_labels=['1','2','3','4','5','6','7','8','9','10']
table_vals=[[11,12],[21,22],[31,32],[1,2],[3,4],[5,6],[7,8],[9,0],[34,54],[78,44]]
# the rectangle is where I want to place the table
the_table = plt.table(cellText=table_vals,
                  colWidths = [0.1]*3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='upper right')
#plt.text(12,3.4,'Table Title',size=8)
plt.show()