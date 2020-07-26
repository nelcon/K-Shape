
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.font_manager import FontProperties

font=FontProperties(fname='/Library/Fonts/Songti.ttc',size=10)


price = [0.21273551422878356, 0.24787492146324197]
"""
绘制水平条形图方法barh
参数一：y轴
参数二：x轴
"""
fig = plt.figure(figsize=(6, 2.2))
ax = fig.add_subplot(111)
plt.barh(0, 51.29896154818086, height=0.06, color='steelblue', alpha=0.8)
plt.barh(0.12, 28.462002058479104, height=0.06, color='crimson', alpha=0.8)
plt.yticks([0,0.12], ['未提取基线', '提取基线'],FontProperties=font)
plt.xlabel("SSE值",FontProperties=font)
plt.title("是否提取基线对SSE值的影响",FontProperties=font)

plt.tight_layout()
plt.show()

