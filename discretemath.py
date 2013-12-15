
def mtable(n):
    """A multiplication table for a ring of natural numbers."""

    table = []
    headers = [None]
    headers.extend(range(n))
    table.append(headers)
    for x in xrange(n):
        row = [x]
        for y in xrange(n):
            row.append(x*y % n)
        table.append(row)

    return table


def print_table(table):
    """Print a table (nested list) of ints."""

    flat = []
    [flat.extend(row) for row in table]
    decimal_places = [len(str(i)) for i in flat if type(i) == int]
    cell_width = reduce(lambda a, b: max(a, b), decimal_places)

    print_num = lambda i: (str(i) if type(i) == int else '').rjust(cell_width)
    
    for row in table:
         print(" ".join([print_num(i) for i in row]))
        
