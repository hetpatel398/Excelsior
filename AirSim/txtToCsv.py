flag=True
with open('data.txt') as infile, open('data_tmp.csv', 'w') as outfile:
    for line in infile:
        if flag:
            '''outfile.write(" ".join(line.split()[:-1]).replace(' ', ','))
            outfile.write(',center,right,left')
            outfile.write("\n") # trailing comma shouldn't matter'''
            flag=False
            continue
        else:
            outfile.write(" ".join(line.split()).replace(' ', ','))
            outfile.write("\n") # trailing comma shouldn't matter
			
with open('data_tmp.csv') as infile, open('data.csv', 'w') as outfile:
    for line in infile:
        outfile.write(" ".join(line.split(';')).replace(' ', ','))

import os
os.remove('./data_tmp.csv')