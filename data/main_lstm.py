import tushare as ts
import os
ts.set_token('41e9c5741a087491d6617061f68121ad0c45920577251efd99f1efa0')
pro = ts.pro_api()
df = pro.daily(ts_code='000859.SH',start_date='2016-01-01',end_date='2018-06-01')
# os.mkdir("stocks1")
df.to_csv('./stocks1/'+'000859.SH'+'.csv',columns=['trade_date','open','high','low','close','vol'])
