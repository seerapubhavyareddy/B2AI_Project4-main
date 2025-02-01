import os

# Base directory containing folds
base_dir = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/MachineLearning/Data_PreProcessing/rp"

# Output file to save results
output_file = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/MachineLearning/DataFormatting/rp_summary.txt"

def count_labels(file_path):
    """Counts occurrences of 0s and 1s in a given file."""
    count_0, count_1 = 0, 0
    with open(file_path, 'r') as file:
        for line in file:
            label = line.strip().split()[-1]
            if label == '0':
                count_0 += 1
            elif label == '1':
                count_1 += 1
    return count_0, count_1

def process_folds(base_dir):
    """Processes each fold and generates label counts."""
    results = []
    for fold in sorted(os.listdir(base_dir)):
        fold_path = os.path.join(base_dir, fold)
        if os.path.isdir(fold_path):
            fold_result = [f"{fold.capitalize()}"]
            for data_type in ['train.txt', 'test.txt']:
                file_path = os.path.join(fold_path, data_type)
                if os.path.exists(file_path):
                    count_0, count_1 = count_labels(file_path)
                    fold_result.append(f"{data_type.split('.')[0].capitalize()}:")
                    fold_result.append(f"    0 - {count_0}")
                    fold_result.append(f"    1 - {count_1}")
            results.append("\n".join(fold_result))
    return results

def save_results(results, output_file):
    """Saves the processed results to a file."""
    with open(output_file, 'w') as file:
        file.write("\n\n".join(results))
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Process folds and generate results
    results = process_folds(base_dir)

    # Save results to a file
    save_results(results, output_file)
