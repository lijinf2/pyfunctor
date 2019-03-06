import csv

def csv_readlines(data_path, encoder = 'utf-8', delimit = ','):
    reader = csv.reader(open(data_path, 'rt', encoding = encoder), delimiter = delimit)
    dataset = []
    for row in reader:
        dataset.append(row)
    return dataset
   
def csv_writelines(output_path, dataset, encoder = 'utf-8', delimit = ','):
    writer = csv.writer(open(output_path, 'w', encoding = encoder), delimiter = delimit)
    for row in dataset:
        writer.writerow(row)
