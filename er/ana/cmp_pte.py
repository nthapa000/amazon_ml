import sys, polars as pl, numpy as np
W='/scratch/m25csa019/er_final'
a=pl.read_parquet(f'{W}/pte_{sys.argv[1]}.parquet'); b=pl.read_parquet(f'{W}/pte_{sys.argv[2]}.parquet')
m=a.join(b,on=['id1','id2'],suffix='_b')
print('rows',a.height,b.height,m.height,'max|dp|',float((m['p']-m['p_b']).abs().max()))
