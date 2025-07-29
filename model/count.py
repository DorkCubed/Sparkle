def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return len(lines)

file_path = '/home/satvik/spark/spark/direction/direction.txt'  
line_count = count_lines_in_file(file_path)
print(f"The number of lines in the file is: {line_count}")
