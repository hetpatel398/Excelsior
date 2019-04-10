'''


PLEASE run this script after Copying it to the Documents Folder


'''

import os
import shutil
folders_list=os.listdir('./Airsim/')
if not os.path.exists('./images'):
    os.makedirs('./images')
if not os.path.exists('./data'):
    os.makedirs('./data')



def txtToCsv(i):
    flag=True
    with open(i) as infile, open('data_tmp.csv', 'w') as outfile:
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
    			
    with open('data_tmp.csv') as infile, open('./data/merge.csv', 'a+') as outfile:
        for line in infile:
            outfile.write(" ".join(line.split(';')).replace(' ', ','))
            
    os.remove('./data_tmp.csv')

for i in folders_list:
    images=os.listdir('./Airsim/%s/images'%(i))
    txtToCsv('./Airsim/%s/airsim_rec.txt'%(i))
    for j in images:
        shutil.copyfile('./Airsim/%s/images/%s'%(i,j), './images/%s'%(j))
        
