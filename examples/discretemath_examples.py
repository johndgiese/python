import discretemath as d
reload(d)

# print are there numbers with more than one inverese?

for ts in xrange(3, 1000):
    mt = d.mtable(ts)

    # remove headers
    mt_no_headers = [row[1:] for row in mt][1:]

    for row_num, row in enumerate(mt_no_headers):
        ones = reduce(lambda x, y: x + y, [i == 1 for i in row])
        if ones > 1:
            print("{} has > 1 inverses in the ring of {}".format(row_num, ts))

