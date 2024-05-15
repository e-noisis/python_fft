import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

# Calculate amplitude in dB for positive frequencies only
positive_fft_result = fft_result[positive_freqs_mask]
amplitude_dB = 20 * np.log10(np.abs(positive_fft_result) / np.max(np.abs(positive_fft_result)))

# Get the threshold from the user
threshold = float(input("Enter the amplitude threshold (in dB): "))

# Find peaks in the FFT result above the threshold with prominence and height filtering
peaks, properties = find_peaks(amplitude_dB, height=threshold, prominence=1, distance=20)

# Get the 10 highest peaks above the threshold
if len(peaks) < 10:
    print(f"Found only {len(peaks)} peaks above the threshold.")
    top_peaks_indices = np.argsort(amplitude_dB[peaks])[-len(peaks):]
else:
    top_peaks_indices = np.argsort(amplitude_dB[peaks])[-10:]
top_peaks_freqs = fft_freqs[positive_freqs_mask][peaks][top_peaks_indices]
top_peaks_amps = amplitude_dB[peaks][top_peaks_indices]

# Sort peaks by frequency for better readability in the table
sorted_indices = np.argsort(top_peaks_freqs)
top_peaks_freqs = top_peaks_freqs[sorted_indices]
top_peaks_amps = top_peaks_amps[sorted_indices]

# Print the top 10 peak values to the terminal
print("Top Peak Frequencies and Amplitudes above the threshold:")
print(f"{'Frequency (Hz)':>15} | {'Amplitude (dB)':>15}")
print("-" * 33)
for freq, amp in zip(top_peaks_freqs, top_peaks_amps):
    print(f"{freq:>15.2f} | {amp:>15.2f}")

# Plot FFT of the 5th column for positive frequencies
plt.figure(figsize=(12, 6))
plt.plot(fft_freqs[positive_freqs_mask], amplitude_dB)
plt.title('FFT of Reference Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.ylim(-140, 0)  # Set y-axis range from -140 dB to 0 dB

# Highlight the top 10 peaks in the plot
plt.plot(top_peaks_freqs, top_peaks_amps, 'ro')  # Red dots on peaks

# Add table for peak values
cell_text = [['Frequency (Hz)', 'Amplitude (dB)']]
for freq, amp in zip(top_peaks_freqs, top_peaks_amps):
    cell_text.append([f'{freq:.2f}', f'{amp:.2f}'])

# Format the table to be placed within the plot
table = plt.table(cellText=cell_text[1:],  # Exclude header from cellText
                  colLabels=cell_text[0],  # Use the first row as colLabels
                  cellLoc='center',
                  loc='right',
                  bbox=[1.2, 0.1, 0.3, 0.8],
                  edges='open')

table.auto_set_font_size(False)
table.set_fontsize(10)

# Display the plot with the table
plt.grid(True)
plt.show()
