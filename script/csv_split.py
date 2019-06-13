import os
import sys

dir_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dir_path, "../"))
from csv_handler import CSV_Handler

assert(len(sys.argv) >= 3)
input_path = sys.argv[1]
percentage = float(sys.argv[2])

first_output_path = input_path + "_train"
if len(sys.argv) >= 4:
    first_output_path = sys.argv[3]

second_output_path = input_path + "_dev"
if len(sys.argv) >= 5:
    second_output_path = sys.argv[4]

handler = CSV_Handler(input_path)
handler.csv_split(percentage, first_output_path, second_output_path)
