line = open('sp500.txt').read()
for line in open('sp500.txt'):
    words = line.strip().split()
    print('$' + words[1], words[2], sep='\t', file=open('sp500-keywords.txt', 'a'))

