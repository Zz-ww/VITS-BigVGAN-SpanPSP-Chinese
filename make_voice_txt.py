import random
from pinyin_dict import pinyin_dict

def gen_duration_from_textgrid(inputdir, output, output1):
    with open(inputdir, 'r') as  f:
        for line in f.readlines():
            try:
                line = line.strip()
                if 'baker' in line:
                    zw_index = 0
                    all = 'sil'
                    single_name = line.split(' ')[0]
                    single_zw = line.split(' ')[2]
                    continue
                py_list = line.split(' ')
                for single in single_zw:
                    if single == '#':
                        all = all[:-2]
                        all += single
                    elif single.isdigit():
                        all += single
                    else:
                        pyname = pinyin_dict.get(py_list[zw_index][:-1])
                        all += ' ' + pyname[0] + ' ' + pyname[1] + py_list[zw_index][-1] + ' ' + '#0'
                        zw_index += 1
                all = './baker_waves/{}.wav|'.format(single_name) + all + ' ' + 'sil' + ' ' + 'eos'
                with open(output, 'a') as  f:
                    f.write(all + '\n')
                if random.randint(0, 50) == 0:
                    with open(output1, 'a') as  w:
                        w.write(all + '\n')
            except Exception as e:
                print(single_name)
                print(e)

def main():
    inputdir = '/mnt/2t/home/zhengbowen/vits_chinese/zlj_checkout_0802_update_all.txt'
    output = '/home/zhengbowen/vits_chinese/filelists/baker_train.txt'
    output1 = '/home/zhengbowen/vits_chinese/filelists/baker_valid.txt'
    gen_duration_from_textgrid(inputdir, output, output1)

if __name__ == "__main__":
    main()
