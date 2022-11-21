import sys
from itertools import count

def find23(n: int) -> int:
    if n < 10:
        return n & ~1
    m = 1
    for b in count(0):
        x = 3 ** b
        if n < x:
            if x-n < abs(n-m):
                m = x
            break
        for a in count(0):
            y = x << a
            if abs(n-y) < abs(n-m):
                m = y
            if n < y:
                break
    return m

FUNCS = {
    'find23': [find23, int],
}

def main():
    assert len(sys.argv), "utils.py {func} [arg0] [arg1] ... [arg{n - 1}]"
    try:
        f_sig = FUNCS[sys.argv[1]]
    except KeyError:
        print(f"unknown function {repr(sys.argv[1])}")
    else:
        assert len(sys.argv) == len(f_sig) + 1, f"Function '{sys.argv[1]} requires {len(f_sig) - 1}'"
        print(f_sig[0](*[parse(s) for parse, s in zip(f_sig[1:], sys.argv[2:])]))

if __name__ == '__main__':
    main()
