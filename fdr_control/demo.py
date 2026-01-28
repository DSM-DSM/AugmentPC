from fdr_control.multi_test import HybridMultiTest

# p_values = [0.00001, 0.0245, 0.03, 0.8, 0.937]
p_values = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.0245, 0.03, 0.8, 0.354, 0.4325, 0.12836, 0.12379, 0.00824,
            0.05564, 0.937]
alpha = 0.2
hmt = HybridMultiTest(alpha=alpha)

reject2 = hmt.BHProcedure(p_values)[0]
reject3 = hmt.BCProcedure(p_values)[0]
# reject4 = hmt.StoreyProcedure(p_values, 0.3)[0]

reject5 = hmt.HybAdaProcedure(p_values)[0]
reject6 = hmt.FastHybAdaProcedure(p_values)[0]

print(reject2)
print(reject3)

print(reject5)
print(reject6)
