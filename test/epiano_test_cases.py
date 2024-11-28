# test_epiano.py

import numpy as np
import matplotlib.pyplot as plt
from asset.epiano import EPiano
from scipy.io.wavfile import write

def test_epiano():
    sample_rate = 44100
    duration = 4.0  # 总持续时间，单位秒

    # 创建 EPiano 实例
    epiano = EPiano(sample_rate=sample_rate, max_polyphony=16)

    # 定义音符序列（MIDI 音符编号）及其开始时间和持续时间
    note_sequence = [
        {'note': 60, 'velocity': 1.0, 'start_time': 0.0, 'duration': 0.5},  # C4
        {'note': 64, 'velocity': 1.0, 'start_time': 0.5, 'duration': 0.5},  # E4
        {'note': 67, 'velocity': 1.0, 'start_time': 1.0, 'duration': 0.5},  # G4
        {'note': 72, 'velocity': 1.0, 'start_time': 1.5, 'duration': 0.5},  # C5
        {'note': 76, 'velocity': 1.0, 'start_time': 2.0, 'duration': 0.5},  # E5
        {'note': 79, 'velocity': 1.0, 'start_time': 2.5, 'duration': 0.5},  # G5
        {'note': 84, 'velocity': 1.0, 'start_time': 3.0, 'duration': 1.0},  # C6
    ]

    # 准备生成音频
    num_samples = int(sample_rate * duration)
    audio = np.zeros(num_samples)
    chunk_size = 1024
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    t = np.linspace(0, duration, num_samples)[:-1]

    # 处理音符事件
    note_events = []
    for note_info in note_sequence:
        start_sample = int(note_info['start_time'] * sample_rate)
        end_sample = start_sample + int(note_info['duration'] * sample_rate)
        note_events.append({
            'note': note_info['note'],
            'velocity': note_info['velocity'],
            'start_sample': start_sample,
            'end_sample': end_sample
        })

    # 按块生成音频
    current_sample = 0
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_samples)
        chunk_duration = (end_idx - start_idx) / sample_rate
        chunk_t = t[start_idx:end_idx]

        # 处理音符事件
        for event in note_events:
            if event['start_sample'] == start_idx:
                epiano.note_on(event['note'], event['velocity'])
            if event['end_sample'] == start_idx:
                epiano.note_off(event['note'])

        # 生成音频块
        audio_chunk = epiano.generate_audio(duration=chunk_duration)
        audio[start_idx:end_idx-1] = audio_chunk

        current_sample += (end_idx - start_idx)

    # 归一化音频
    audio /= np.max(np.abs(audio))

    # 保存为 WAV 文件
    scaled = np.int16(audio * 32767)
    write('epiano_output.wav', sample_rate, scaled)

    # 绘制音频片段
    plt.figure(figsize=(10, 4))
    plt.plot(t[:5000], audio[:5000])
    plt.title("EPiano Output")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
