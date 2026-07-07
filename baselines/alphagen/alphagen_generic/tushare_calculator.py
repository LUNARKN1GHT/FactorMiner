"""跟 alphagen_qlib.calculator.QLibStockDataCalculator 逻辑完全一致（鸭子类型复用）——
唯一区别是喂的是 TushareStockData 不是真的 qlib 数据，换个名字避免误导。"""
from alphagen_qlib.calculator import QLibStockDataCalculator


class TushareStockDataCalculator(QLibStockDataCalculator):
    pass
