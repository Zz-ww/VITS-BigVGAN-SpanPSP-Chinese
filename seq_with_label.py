import re
import numpy as np

import math
import os.path
import subprocess
import tempfile

import nltk


def remove_top(a):
    a = a.replace('(TOP (S ', '')
    a = a[::-1].replace('))', '', 1)[::-1]
    return a

def replace_n(a, i, j, num):
    a = a.replace(i, '*', num)
    a = a.replace('*', i, num-1)
    a = a.replace('*', j)
    return a

def replace1(a):
    a = re.sub('\n', '', a)
    for i in range(len(a)):
        num_left = 0
        num_right = 0
        flag = 0
        for j in range(len(a)):          
            if a[j] == '(' :
                num_left += 1
                if a[j+1] == 'S' and flag == 0:
                    b = a[j+1]
                    flag = 1
                if a[j+1] == '#' and flag == 0:
                    b = a[j+1] + a[j+2]
                    flag = 1
                    
            elif a[j] == ')' :
                num_right += 1
                if num_right == num_left and a[j-1] == ')':
                    a = replace_n(a, ')', b, num_left)
                    a = a.replace('('+b, '', 1)
                    break
    return a

def replace2(a):
    s = re.sub('\n', '', a)
    s = re.sub('#', '', s)

    compileX = re.compile(r'\d+')
    num_result = compileX.findall(s)
    for i in num_result:
        if i != '1':
            s = re.sub(i, '#'+ str(max(i)), s, 1)
    s = re.sub('#1', '##', s)
    s = re.sub('1','#1', s)
    s = re.sub('##','#1',s)
    
    s = s.replace('(n ', '')
    s = s.replace(')', '')

    punctuation_list = ['，','。','、','；','：','？','！','“','”','‘','’','—','…','（','）','《','》']
    for punc in  punctuation_list:
        # s = re.sub('('+ punc, '', s)
        s = s.replace('('+ punc, '')
    
    s = re.sub(' ', '', s)

    return s


def output_file(temp_data_path, output_data_path):

    line_sen_list = []

    with open(temp_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sen_token_list = []
           
            if line != '\n' and line != '':
               
                ss = line
                sss = remove_top(ss)
                sss = replace1(sss)
                sss = replace2(sss)
                sen_token_list.append(sss+ '\n')
                line_sen_list.append(''.join(sen_token_list))
        f.close()
    
    with open(output_data_path,'w+', encoding='utf-8') as o:
      o.write(''.join(line_sen_list))
      o.close()



def output_seq(tem_data_path):

    line_sen_list = []

    with open(tem_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sen_token_list = []
            if line != '\n' and line != '':
                ss = line
                sss = remove_top(ss)
                sss = replace1(sss)
                sss = replace2(sss)
                sen_token_list.append(sss+ '\n')
                line_sen_list.append(''.join(sen_token_list))
        f.close()
    return line_sen_list




################################################################################################################
def output(output_path, predicted_trees):
    for predicted_tree in predicted_trees:
        assert isinstance(predicted_tree, nltk.Tree)

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    predicted_path = os.path.join(temp_dir.name, "predicted.txt")

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            # print(tree)
            outfile.write("{}\n".format(tree.pformat(margin=1e100)))

    output_file(predicted_path, output_path)
