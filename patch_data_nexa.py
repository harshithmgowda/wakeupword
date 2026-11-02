path = '/root/openwakeword-training/openwakeword/openwakeword/data.py'
with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

target = '    word_phones = []\n    input_text_phones = [pronouncing.phones_for_word(i) for i in input_text.split()]'
replacement = '''    pronouncing.init_cmu()
    if 'nexa' not in pronouncing.lookup:
        pronouncing.lookup['nexa'] = ['N EH1 K S AH0']
    word_phones = []
    input_text_phones = [pronouncing.phones_for_word(i) for i in input_text.split()]'''

assert target in code, "Target not found in data.py"
code = code.replace(target, replacement)
with open(path, 'w', encoding='utf-8') as f:
    f.write(code)

print("Successfully injected nexa pronunciation into openwakeword/data.py")

