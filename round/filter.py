for line in open('tick-201611.txt'):
    if ',20161101,' in line:
        # if i % 1000 == 0:
        if ',20161102,' in line:
            break
        #	print(i)
        print(line.strip())
