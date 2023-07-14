import json
import matplotlib.pyplot as plt
import numpy as np
import ast


def process_og_gz_arith_baseline():
    with open("stats1000.json") as read_file:
        data = json.load(read_file)
        og = []
        arith_baseline = []
        gz = []
        for batch in data:
            for art in batch:
                og.append(art[0])
                arith_baseline.append(art[1])
                gz.append(art[2])
    return np.array(og), np.array(arith_baseline), np.array(gz)


def process_stats_1000():
    og, gz, neural_varint, neural_arith = [], [], [], []
    with open("stats1000.txt", "r") as f:
        for line in f:
            line = line[line.find("["):]
            x = ast.literal_eval(line)
            for t in x:
                og.append(t[0])
                gz.append(t[1])
                neural_varint.append(t[2])
                neural_arith.append(t[3])
    return np.array(og), np.array(gz), np.array(neural_varint), np.array(neural_arith)


def process_arith_neural():
    neural_arith = []
    with open("saving_progress.txt", "r") as f:
        for line in f:
            x = ast.literal_eval(line)
            for t in x:
                neural_arith.append(t[0])
    neural_arith = np.array(neural_arith)
    return neural_arith


def get_cropped_og():
    with open("cropped_arts.json") as read_file:
        data = json.load(read_file)
        return np.array(data)


def get_compression_ratio(neural_varint, og, neural_arith, arith_baseline):
    # gz_baseline = og / gz_baseline
    neural_varint = og / neural_varint
    arith_baseline = og / arith_baseline
    neural_arith = og / neural_arith
    return neural_varint, neural_arith, arith_baseline


def graph_data(gz_baseline, neural_varint, neural_arith, arith_baseline):
    plt.hist(gz_baseline, color='y', label='GZIP_Baseline', alpha=0.7)
    plt.hist(arith_baseline, color='g', label='Arith_Baseline', alpha=0.7)
    plt.hist(neural_arith, color='b', label='Neural_Arith', alpha=0.7)
    plt.hist(neural_varint, color='red', label='Neural_Varint', alpha=0.7)
    plt.xlabel('Compression Ratio (uncompressed/compressed size in bytes)')
    plt.ylabel('Number of documents')
    plt.legend()
    plt.show()


og, gz_baseline, neural_varint, neural_arith = process_stats_1000()
gz_baseline = og / gz_baseline
og = get_cropped_og()

og1, arith_baseline, gz_baseline2 = process_og_gz_arith_baseline()
neural_varint, neural_arith, arith_baseline = get_compression_ratio(neural_varint, og, neural_arith, arith_baseline)

print("GZIP mean, min, max, std: ", np.mean(gz_baseline), np.min(gz_baseline), np.max(gz_baseline), np.std(gz_baseline))
print("Arith baseline mean, min, max, std: ", np.mean(arith_baseline), np.min(arith_baseline), np.max(arith_baseline),
      np.std(arith_baseline))
print("Neural varint mean, min, max, std: ", np.mean(neural_varint), np.min(neural_varint), np.max(neural_varint),
      np.std(neural_varint))
print("Neural airth mean, min, max, std: ", np.mean(neural_arith), np.min(neural_arith), np.max(neural_arith),
      np.std(neural_arith))

graph_data(gz_baseline, neural_varint, neural_arith, arith_baseline)
