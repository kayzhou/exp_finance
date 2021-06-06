from mongo_handler import get_coll_stock_v2
import matplotlib.pyplot as plt
import pendulum
import datetime
import numpy as np


def get_emotion():
    dts = []
    sum_mood = []
    moods_0 = []
    moods_1 = []
    moods_3 = []
    moods_4 = []

    coll = get_coll_stock_v2()
    data = coll.find({}).sort('_id')

    count = 0
    date_set = set()

    for d in data:
        # print(d)
        dt = d['datetime'][: 10]
        if not dt.endswith('24'):
            continue
        count += 1
        # dt = pendulum.strptime(dt[: 8], '%Y%m%d')
        dt = datetime.datetime.strptime(dt[: 8], '%Y%m%d')
        # print(dt.weekday())
        # if dt.weekday() in [5, 6]:
        #     continue
        mood = [ d['mood']['0'],d['mood']['1'], d['mood']['2'], d['mood']['3'], d['mood']['4'] ]
        print(dt, mood)
        date_set.add(dt)
        dts.append(dt.strftime('%Y%m%d'))
        moods_0.append('%.4f' % np.log10( (mood[0] + 1) / (mood[2] + 1) ))
        moods_1.append('%.4f' % np.log10( (mood[1] + 1) / (mood[2] + 1) ))
        moods_3.append('%.4f' % np.log10( (mood[3] + 1) / (mood[2] + 1) ))
        moods_4.append('%.4f' % np.log10( (mood[4] + 1) / (mood[2] + 1) ))
        # print(dt, sum(mood))
        sum_mood.append(str(sum(mood)))

    return dts, moods_0, moods_1, moods_3, moods_4, sum_mood
    # print(count, len(date_set))
    # # plt.plot(dts, moods_0, label='disgust / joy')
    # # plt.plot(dts, moods_1, label='anger / joy')
    # # plt.plot(dts, moods_3, label='sadness / joy')
    # # plt.plot(dts, moods_4, label='fear / joy')
    # plt.plot(dts, sum_mood)

    # plt.legend()
    # plt.show()
    # plt.close()


def union_stock_emotion():

    dts, moods_0, moods_1, moods_3, moods_4, sum_mood = get_emotion()
    for line in open('../data/stock-data.csv'):
        words = line.strip().split(',')
        dt = words[0]
        print(dt)
        ix = dts.index(dt)

        words.extend([moods_0[ix], moods_1[ix], moods_3[ix], moods_4[ix], sum_mood[ix]])
        # print(words)
        print(','.join(words), file=open('../data/stock-emotions.csv', 'a'))


if __name__ == '__main__':
    # get_emotion()
    union_stock_emotion()

