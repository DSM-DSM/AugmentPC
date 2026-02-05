import numpy as np

from fdr_control.multi_test import HybridMultiTest

# p_values = [0.00001, 0.0245, 0.03, 0.8, 0.937]
# p_values = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.0245, 0.03, 0.8, 0.354, 0.4325, 0.12836, 0.12379, 0.00824,
#             0.05564, 0.937]
np.random.seed(42)
p_values = np.random.rand(10).tolist()
p_values.extend((np.random.rand(10) / 10).tolist())
print(np.array(p_values))
alpha = 0.15
hmt = HybridMultiTest(alpha=alpha)
reject2 = hmt.BHProcedure(p_values)[0]
reject3 = hmt.BCProcedure(p_values)[0]
reject5 = hmt.HybAdaProcedure(p_values)[0]
reject6 = hmt.FastHybAdaProcedure(p_values)[0]

print('*' * 80)
print('Original Rejection:')
print(np.array(p_values) < 0.05)
print('*' * 80)
print('BHProcedure:')
print(reject2)
print('*' * 80)
print('BCProcedure:')
print(reject3)
print('*' * 80)
print('HybAdaProcedure:')
print(reject5)
print('*' * 80)
print('FastHybAdaProcedure:')
print(reject6)
print('*' * 80)
