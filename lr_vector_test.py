import linear_reg_bt as LR

lrbt = LR.LRVectorBacktester("2010-1-1", "2019-12-31", 10000, 0.0)

lrbt.run_strategy("2010-1-1", "2019-12-31","2010-1-1", "2019-12-31", lags=5)
lrbt.plot_results()
