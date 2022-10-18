import re 
import sys
from typing import List
regex = re.compile('[.,@_!#$%^&*<>?/\|}{~:]')

def comb_inputs(s1:List[str], s2:List[str], max_len:int=1024) -> List[str]:
    ## 2k is of MDS (doesn matter since they cut to 500)
    assert max_len in [512, 1024, 2000], 'Only these three lengths are allowed. I am imposing this restricting based on what models expect'
    half_len = max_len//2
    comb = []
    for ii, jj in zip(s1, s2):
        li = len(ii.split())
        lj = len(jj.split())
        if li + lj <= max_len:
            pass
        else:
            if li >= half_len and lj >= half_len:
                ii = ' '.join(ii.split()[:half_len])
                jj = ' '.join(jj.split()[:half_len])
            elif li >= half_len:
                ii = ' '.join(ii.split()[:max_len - lj])
            elif lj >= half_len:
                jj = ' '.join(jj.split()[:max_len - li])
            else:
                print('Something wrong.')
                sys.exit(0)

        if regex.search(ii[-1]) == None:
            ii += "."

        comb.append(ii + ' ' + jj)
    return comb