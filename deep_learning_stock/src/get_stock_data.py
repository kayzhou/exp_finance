from mongo_handler import get_coll_sh


def get_data():
    coll = get_coll_sh()
    data = coll.find({}).sort('date')
    for d in data:
        dt = d['date']
        if dt >= '20150101':
            print(dt, d['close'], d['open'], d['high'], d['low'], d['vol'], d['amt'], sep=',', file=open('../data/stock-data.csv', 'a'))


if __name__ == '__main__':
    get_data()