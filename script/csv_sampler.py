import os
import sys

dir_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dir_path, "../"))
from csv_handler import CSV_Handler

assert(len(sys.argv) == 4)
input_path = sys.argv[1]
num_sample = int(sys.argv[2])
output_path = sys.argv[3]

handler = CSV_Handler(input_path)
handler.csv_shuf(num_sample, output_path)
