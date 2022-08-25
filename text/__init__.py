from text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text.split()]
    # sequence=[]
    # for symbol in cleaned_text.split():
    #   sequence.append(_symbol_to_id[symbol])
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


if __name__ == '__main__':
    with open('/mnt/2t/home/zhengbowen/vits_chinese/filelists/baker_train_0810.txt', 'r') as  f:
        for line in f.readlines():
            try:
                name = line.strip().split('|')[0]
                line = line.strip().split('|')[1]
                # line_list=line.split(' ')
                cleaned_text_to_sequence(line)
            except Exception as e:
                print(name, e)
