def compare_file(file_a, file_b):
    with open(file_a) as f:
        a = f.read()
    with open(file_b) as f:
        b = f.read()
    a_paras = a.split('\n')
    b_paras = b.split('\n')

    for i in range(len(a_paras)):
        a_spaces = a_paras[i].split(' ')
        b_spaces = b_paras[i].split(' ')
        a_spaces = [i for i in a_spaces if i != '']
        b_spaces = [i for i in b_spaces if i != '']
        if len(a_spaces) != len(b_spaces):
            print(len(a_spaces), len(b_spaces))
            print(f'Files are different at line {i + 1}')

    print('Files are identical')


compare_file(r"C:\Users\Artemis\Downloads\text\packets\2013-10-21_capture-1-only-dns-0001 (1).txt",
             r"C:\Users\Artemis\Downloads\text\fields\2013-10-21_capture-1-only-dns-0001 (1).txt")
