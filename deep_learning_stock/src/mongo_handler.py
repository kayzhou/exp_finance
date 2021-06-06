import pymongo as pm


def get_coll_stock_v2():
    db = pm.MongoClient('192.168.1.222', 47541).get_database('stock')
    # db.authenticate('root', '!Wode@207')
    coll = db.get_collection('stock_v2')
    return coll


def get_coll_sh():
    db = pm.MongoClient('192.168.1.222', 47541).get_database('k1day')
    # db.authenticate('root', '!Wode@207')
    coll = db.get_collection('999999.SH')
    return coll


if __name__ == '__main__':
    coll = get_coll_stock_v2()
    data = coll.find({})
    for d in data:
        # print(d)
        dt = d['datetime'][: 10]
        if not dt.endswith('24'):
            continue
        dt = dt[: 8]
        mood = [ d['mood']['0'],d['mood']['1'], d['mood']['2'], d['mood']['3'], d['mood']['4'] ] 
        print(dt, mood)