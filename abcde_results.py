# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:56:12 2023

@author: Lena Papailiou
"""
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import random

metrics = ['Intent_Accuracy', 'Nextstep_Accuracy', 'Action_Accuracy', 'Value_Accuracy', 'Joint_Accuracy',
           'Recall_at_1', 'Recall_at_5', 'Recall_at_10', 'Turn_Accuracy', 'Cascading_Score']


groundtruth_bert = [ # abcde3
{'Intent_Accuracy_groundtruth_cds': 0.5213, 'Nextstep_Accuracy_groundtruth_cds': 0.7932, 'Action_Accuracy_groundtruth_cds': 0.6135, 'Value_Accuracy_groundtruth_cds': 0.2248, 'Joint_Accuracy_groundtruth_cds': 0.1526, 'Recall_at_1_groundtruth_cds': 0.087, 'Recall_at_5_groundtruth_cds': 0.2916, 'Recall_at_10_groundtruth_cds': 0.4109, 'Turn_Accuracy_groundtruth_cds': 0.0718, 'Cascading_Score_groundtruth_cds': 0.0415},
{'Intent_Accuracy_groundtruth_cds': 0.7287, 'Nextstep_Accuracy_groundtruth_cds': 0.7994, 'Action_Accuracy_groundtruth_cds': 0.6907, 'Value_Accuracy_groundtruth_cds': 0.3316, 'Joint_Accuracy_groundtruth_cds': 0.2282, 'Recall_at_1_groundtruth_cds': 0.1174, 'Recall_at_5_groundtruth_cds': 0.3571, 'Recall_at_10_groundtruth_cds': 0.479, 'Turn_Accuracy_groundtruth_cds': 0.1304, 'Cascading_Score_groundtruth_cds': 0.0592},
{'Intent_Accuracy_groundtruth_cds': 0.8522, 'Nextstep_Accuracy_groundtruth_cds': 0.8043, 'Action_Accuracy_groundtruth_cds': 0.7052, 'Value_Accuracy_groundtruth_cds': 0.3759, 'Joint_Accuracy_groundtruth_cds': 0.2575, 'Recall_at_1_groundtruth_cds': 0.1446, 'Recall_at_5_groundtruth_cds': 0.3993, 'Recall_at_10_groundtruth_cds': 0.5165, 'Turn_Accuracy_groundtruth_cds': 0.1801, 'Cascading_Score_groundtruth_cds': 0.0744},
{'Intent_Accuracy_groundtruth_cds': 0.9012, 'Nextstep_Accuracy_groundtruth_cds': 0.8094, 'Action_Accuracy_groundtruth_cds': 0.7137, 'Value_Accuracy_groundtruth_cds': 0.4147, 'Joint_Accuracy_groundtruth_cds': 0.2806, 'Recall_at_1_groundtruth_cds': 0.1583, 'Recall_at_5_groundtruth_cds': 0.4085, 'Recall_at_10_groundtruth_cds': 0.5317, 'Turn_Accuracy_groundtruth_cds': 0.2036, 'Cascading_Score_groundtruth_cds': 0.081},
{'Intent_Accuracy_groundtruth_cds': 0.9222, 'Nextstep_Accuracy_groundtruth_cds': 0.811, 'Action_Accuracy_groundtruth_cds': 0.7323, 'Value_Accuracy_groundtruth_cds': 0.4539, 'Joint_Accuracy_groundtruth_cds': 0.3077, 'Recall_at_1_groundtruth_cds': 0.1709, 'Recall_at_5_groundtruth_cds': 0.4191, 'Recall_at_10_groundtruth_cds': 0.5396, 'Turn_Accuracy_groundtruth_cds': 0.2231, 'Cascading_Score_groundtruth_cds': 0.085},
{'Intent_Accuracy_groundtruth_cds': 0.9216, 'Nextstep_Accuracy_groundtruth_cds': 0.8101, 'Action_Accuracy_groundtruth_cds': 0.7408, 'Value_Accuracy_groundtruth_cds': 0.5127, 'Joint_Accuracy_groundtruth_cds': 0.3419, 'Recall_at_1_groundtruth_cds': 0.1532, 'Recall_at_5_groundtruth_cds': 0.3978, 'Recall_at_10_groundtruth_cds': 0.512, 'Turn_Accuracy_groundtruth_cds': 0.221, 'Cascading_Score_groundtruth_cds': 0.0886},
{'Intent_Accuracy_groundtruth_cds': 0.9214, 'Nextstep_Accuracy_groundtruth_cds': 0.8127, 'Action_Accuracy_groundtruth_cds': 0.7296, 'Value_Accuracy_groundtruth_cds': 0.5518, 'Joint_Accuracy_groundtruth_cds': 0.3584, 'Recall_at_1_groundtruth_cds': 0.1727, 'Recall_at_5_groundtruth_cds': 0.4291, 'Recall_at_10_groundtruth_cds': 0.5416, 'Turn_Accuracy_groundtruth_cds': 0.2357, 'Cascading_Score_groundtruth_cds': 0.0901},
{'Intent_Accuracy_groundtruth_cds': 0.9247, 'Nextstep_Accuracy_groundtruth_cds': 0.8186, 'Action_Accuracy_groundtruth_cds': 0.7526, 'Value_Accuracy_groundtruth_cds': 0.5738, 'Joint_Accuracy_groundtruth_cds': 0.3741, 'Recall_at_1_groundtruth_cds': 0.1924, 'Recall_at_5_groundtruth_cds': 0.4535, 'Recall_at_10_groundtruth_cds': 0.5658, 'Turn_Accuracy_groundtruth_cds': 0.2525, 'Cascading_Score_groundtruth_cds': 0.0955},
{'Intent_Accuracy_groundtruth_cds': 0.9248, 'Nextstep_Accuracy_groundtruth_cds': 0.8214, 'Action_Accuracy_groundtruth_cds': 0.7614, 'Value_Accuracy_groundtruth_cds': 0.587, 'Joint_Accuracy_groundtruth_cds': 0.3851, 'Recall_at_1_groundtruth_cds': 0.2093, 'Recall_at_5_groundtruth_cds': 0.4792, 'Recall_at_10_groundtruth_cds': 0.5883, 'Turn_Accuracy_groundtruth_cds': 0.2664, 'Cascading_Score_groundtruth_cds': 0.1008},
{'Intent_Accuracy_groundtruth_cds': 0.9248, 'Nextstep_Accuracy_groundtruth_cds': 0.8259, 'Action_Accuracy_groundtruth_cds': 0.7639, 'Value_Accuracy_groundtruth_cds': 0.589, 'Joint_Accuracy_groundtruth_cds': 0.3859, 'Recall_at_1_groundtruth_cds': 0.2157, 'Recall_at_5_groundtruth_cds': 0.4889, 'Recall_at_10_groundtruth_cds': 0.5967, 'Turn_Accuracy_groundtruth_cds': 0.2711, 'Cascading_Score_groundtruth_cds': 0.1023}
        ]

groundtruth_albert = [ # abcde1
{'Intent_Accuracy_groundtruth_cds': 0.3682, 'Nextstep_Accuracy_groundtruth_cds': 0.7884, 'Action_Accuracy_groundtruth_cds': 0.5652, 'Value_Accuracy_groundtruth_cds': 0.2491, 'Joint_Accuracy_groundtruth_cds': 0.1648, 'Recall_at_1_groundtruth_cds': 0.0759, 'Recall_at_5_groundtruth_cds': 0.2579, 'Recall_at_10_groundtruth_cds': 0.3741, 'Turn_Accuracy_groundtruth_cds': 0.0496, 'Cascading_Score_groundtruth_cds': 0.0287},
{'Intent_Accuracy_groundtruth_cds': 0.6134, 'Nextstep_Accuracy_groundtruth_cds': 0.804, 'Action_Accuracy_groundtruth_cds': 0.6748, 'Value_Accuracy_groundtruth_cds': 0.2976, 'Joint_Accuracy_groundtruth_cds': 0.2029, 'Recall_at_1_groundtruth_cds': 0.1114, 'Recall_at_5_groundtruth_cds': 0.3498, 'Recall_at_10_groundtruth_cds': 0.4799, 'Turn_Accuracy_groundtruth_cds': 0.1062, 'Cascading_Score_groundtruth_cds': 0.0508},
{'Intent_Accuracy_groundtruth_cds': 0.7817, 'Nextstep_Accuracy_groundtruth_cds': 0.8073, 'Action_Accuracy_groundtruth_cds': 0.7099, 'Value_Accuracy_groundtruth_cds': 0.3122, 'Joint_Accuracy_groundtruth_cds': 0.2143, 'Recall_at_1_groundtruth_cds': 0.1328, 'Recall_at_5_groundtruth_cds': 0.3839, 'Recall_at_10_groundtruth_cds': 0.5084, 'Turn_Accuracy_groundtruth_cds': 0.1427, 'Cascading_Score_groundtruth_cds': 0.0611},
{'Intent_Accuracy_groundtruth_cds': 0.8169, 'Nextstep_Accuracy_groundtruth_cds': 0.8158, 'Action_Accuracy_groundtruth_cds': 0.7152, 'Value_Accuracy_groundtruth_cds': 0.3333, 'Joint_Accuracy_groundtruth_cds': 0.2276, 'Recall_at_1_groundtruth_cds': 0.1502, 'Recall_at_5_groundtruth_cds': 0.4108, 'Recall_at_10_groundtruth_cds': 0.5314, 'Turn_Accuracy_groundtruth_cds': 0.1647, 'Cascading_Score_groundtruth_cds': 0.0677},
{'Intent_Accuracy_groundtruth_cds': 0.8382, 'Nextstep_Accuracy_groundtruth_cds': 0.8187, 'Action_Accuracy_groundtruth_cds': 0.7443, 'Value_Accuracy_groundtruth_cds': 0.3756, 'Joint_Accuracy_groundtruth_cds': 0.2581, 'Recall_at_1_groundtruth_cds': 0.1701, 'Recall_at_5_groundtruth_cds': 0.4412, 'Recall_at_10_groundtruth_cds': 0.5585, 'Turn_Accuracy_groundtruth_cds': 0.1848, 'Cascading_Score_groundtruth_cds': 0.074},
{'Intent_Accuracy_groundtruth_cds': 0.7486, 'Nextstep_Accuracy_groundtruth_cds': 0.7563, 'Action_Accuracy_groundtruth_cds': 0.6695, 'Value_Accuracy_groundtruth_cds': 0.2722, 'Joint_Accuracy_groundtruth_cds': 0.1803, 'Recall_at_1_groundtruth_cds': 0.1172, 'Recall_at_5_groundtruth_cds': 0.3249, 'Recall_at_10_groundtruth_cds': 0.4315, 'Turn_Accuracy_groundtruth_cds': 0.1055, 'Cascading_Score_groundtruth_cds': 0.0372},
{'Intent_Accuracy_groundtruth_cds': 0.8749, 'Nextstep_Accuracy_groundtruth_cds': 0.8061, 'Action_Accuracy_groundtruth_cds': 0.7396, 'Value_Accuracy_groundtruth_cds': 0.3842, 'Joint_Accuracy_groundtruth_cds': 0.2626, 'Recall_at_1_groundtruth_cds': 0.1448, 'Recall_at_5_groundtruth_cds': 0.3872, 'Recall_at_10_groundtruth_cds': 0.5035, 'Turn_Accuracy_groundtruth_cds': 0.1788, 'Cascading_Score_groundtruth_cds': 0.0721},
{'Intent_Accuracy_groundtruth_cds': 0.9045, 'Nextstep_Accuracy_groundtruth_cds': 0.8149, 'Action_Accuracy_groundtruth_cds': 0.7531, 'Value_Accuracy_groundtruth_cds': 0.403, 'Joint_Accuracy_groundtruth_cds': 0.2763, 'Recall_at_1_groundtruth_cds': 0.1733, 'Recall_at_5_groundtruth_cds': 0.441, 'Recall_at_10_groundtruth_cds': 0.5608, 'Turn_Accuracy_groundtruth_cds': 0.2067, 'Cascading_Score_groundtruth_cds': 0.0807},
{'Intent_Accuracy_groundtruth_cds': 0.9259, 'Nextstep_Accuracy_groundtruth_cds': 0.8178, 'Action_Accuracy_groundtruth_cds': 0.7602, 'Value_Accuracy_groundtruth_cds': 0.4419, 'Joint_Accuracy_groundtruth_cds': 0.304, 'Recall_at_1_groundtruth_cds': 0.183, 'Recall_at_5_groundtruth_cds': 0.4603, 'Recall_at_10_groundtruth_cds': 0.5829, 'Turn_Accuracy_groundtruth_cds': 0.2227, 'Cascading_Score_groundtruth_cds': 0.0846},
{'Intent_Accuracy_groundtruth_cds': 0.9324, 'Nextstep_Accuracy_groundtruth_cds': 0.8143, 'Action_Accuracy_groundtruth_cds': 0.7539, 'Value_Accuracy_groundtruth_cds': 0.4316, 'Joint_Accuracy_groundtruth_cds': 0.2956, 'Recall_at_1_groundtruth_cds': 0.1892, 'Recall_at_5_groundtruth_cds': 0.4649, 'Recall_at_10_groundtruth_cds': 0.5844, 'Turn_Accuracy_groundtruth_cds': 0.2262, 'Cascading_Score_groundtruth_cds': 0.0851}
        ]

groundtruth_roberta = [ # abcde2
{'Intent_Accuracy_groundtruth_cds': 0.3804, 'Nextstep_Accuracy_groundtruth_cds': 0.7978, 'Action_Accuracy_groundtruth_cds': 0.479, 'Value_Accuracy_groundtruth_cds': 0.0626, 'Joint_Accuracy_groundtruth_cds': 0.0391, 'Recall_at_1_groundtruth_cds': 0.0897, 'Recall_at_5_groundtruth_cds': 0.3081, 'Recall_at_10_groundtruth_cds': 0.427, 'Turn_Accuracy_groundtruth_cds': 0.0446, 'Cascading_Score_groundtruth_cds': 0.0282},
{'Intent_Accuracy_groundtruth_cds': 0.6187, 'Nextstep_Accuracy_groundtruth_cds': 0.8061, 'Action_Accuracy_groundtruth_cds': 0.5817, 'Value_Accuracy_groundtruth_cds': 0.0609, 'Joint_Accuracy_groundtruth_cds': 0.0401, 'Recall_at_1_groundtruth_cds': 0.1066, 'Recall_at_5_groundtruth_cds': 0.3479, 'Recall_at_10_groundtruth_cds': 0.4771, 'Turn_Accuracy_groundtruth_cds': 0.0776, 'Cascading_Score_groundtruth_cds': 0.0473},
{'Intent_Accuracy_groundtruth_cds': 0.7306, 'Nextstep_Accuracy_groundtruth_cds': 0.8143, 'Action_Accuracy_groundtruth_cds': 0.6771, 'Value_Accuracy_groundtruth_cds': 0.1149, 'Joint_Accuracy_groundtruth_cds': 0.0789, 'Recall_at_1_groundtruth_cds': 0.128, 'Recall_at_5_groundtruth_cds': 0.3884, 'Recall_at_10_groundtruth_cds': 0.5115, 'Turn_Accuracy_groundtruth_cds': 0.1046, 'Cascading_Score_groundtruth_cds': 0.0571},
{'Intent_Accuracy_groundtruth_cds': 0.7618, 'Nextstep_Accuracy_groundtruth_cds': 0.8163, 'Action_Accuracy_groundtruth_cds': 0.698, 'Value_Accuracy_groundtruth_cds': 0.1637, 'Joint_Accuracy_groundtruth_cds': 0.1108, 'Recall_at_1_groundtruth_cds': 0.1533, 'Recall_at_5_groundtruth_cds': 0.4309, 'Recall_at_10_groundtruth_cds': 0.559, 'Turn_Accuracy_groundtruth_cds': 0.1311, 'Cascading_Score_groundtruth_cds': 0.0663},
{'Intent_Accuracy_groundtruth_cds': 0.7849, 'Nextstep_Accuracy_groundtruth_cds': 0.8249, 'Action_Accuracy_groundtruth_cds': 0.7109, 'Value_Accuracy_groundtruth_cds': 0.1974, 'Joint_Accuracy_groundtruth_cds': 0.1341, 'Recall_at_1_groundtruth_cds': 0.1692, 'Recall_at_5_groundtruth_cds': 0.4502, 'Recall_at_10_groundtruth_cds': 0.5785, 'Turn_Accuracy_groundtruth_cds': 0.1486, 'Cascading_Score_groundtruth_cds': 0.0698},
{'Intent_Accuracy_groundtruth_cds': 0.7929, 'Nextstep_Accuracy_groundtruth_cds': 0.8108, 'Action_Accuracy_groundtruth_cds': 0.7205, 'Value_Accuracy_groundtruth_cds': 0.2034, 'Joint_Accuracy_groundtruth_cds': 0.1398, 'Recall_at_1_groundtruth_cds': 0.136, 'Recall_at_5_groundtruth_cds': 0.3805, 'Recall_at_10_groundtruth_cds': 0.4967, 'Turn_Accuracy_groundtruth_cds': 0.1313, 'Cascading_Score_groundtruth_cds': 0.062},
{'Intent_Accuracy_groundtruth_cds': 0.8537, 'Nextstep_Accuracy_groundtruth_cds': 0.8134, 'Action_Accuracy_groundtruth_cds': 0.7245, 'Value_Accuracy_groundtruth_cds': 0.2543, 'Joint_Accuracy_groundtruth_cds': 0.1714, 'Recall_at_1_groundtruth_cds': 0.1485, 'Recall_at_5_groundtruth_cds': 0.408, 'Recall_at_10_groundtruth_cds': 0.5258, 'Turn_Accuracy_groundtruth_cds': 0.1558, 'Cascading_Score_groundtruth_cds': 0.0698},
{'Intent_Accuracy_groundtruth_cds': 0.8943, 'Nextstep_Accuracy_groundtruth_cds': 0.8259, 'Action_Accuracy_groundtruth_cds': 0.7457, 'Value_Accuracy_groundtruth_cds': 0.3069, 'Joint_Accuracy_groundtruth_cds': 0.2113, 'Recall_at_1_groundtruth_cds': 0.1704, 'Recall_at_5_groundtruth_cds': 0.44, 'Recall_at_10_groundtruth_cds': 0.5629, 'Turn_Accuracy_groundtruth_cds': 0.1874, 'Cascading_Score_groundtruth_cds': 0.078},
{'Intent_Accuracy_groundtruth_cds': 0.9048, 'Nextstep_Accuracy_groundtruth_cds': 0.8339, 'Action_Accuracy_groundtruth_cds': 0.7616, 'Value_Accuracy_groundtruth_cds': 0.3371, 'Joint_Accuracy_groundtruth_cds': 0.2311, 'Recall_at_1_groundtruth_cds': 0.1792, 'Recall_at_5_groundtruth_cds': 0.4589, 'Recall_at_10_groundtruth_cds': 0.5824, 'Turn_Accuracy_groundtruth_cds': 0.1981, 'Cascading_Score_groundtruth_cds': 0.0799},
{'Intent_Accuracy_groundtruth_cds': 0.9154, 'Nextstep_Accuracy_groundtruth_cds': 0.839, 'Action_Accuracy_groundtruth_cds': 0.7679, 'Value_Accuracy_groundtruth_cds': 0.3451, 'Joint_Accuracy_groundtruth_cds': 0.2361, 'Recall_at_1_groundtruth_cds': 0.1927, 'Recall_at_5_groundtruth_cds': 0.4784, 'Recall_at_10_groundtruth_cds': 0.598, 'Turn_Accuracy_groundtruth_cds': 0.2098, 'Cascading_Score_groundtruth_cds': 0.0841}
 ]

bert_exp = [ # abcde3
{'Intent_Accuracy_bert_exp': 0.5017, 'Nextstep_Accuracy_bert_exp': 0.7933, 'Action_Accuracy_bert_exp': 0.6493, 'Value_Accuracy_bert_exp': 0.2462, 'Joint_Accuracy_bert_exp': 0.1691, 'Recall_at_1_bert_exp': 0.0875, 'Recall_at_5_bert_exp': 0.2895, 'Recall_at_10_bert_exp': 0.405, 'Turn_Accuracy_bert_exp': 0.0741, 'Cascading_Score_bert_exp': 0.0391},
{'Intent_Accuracy_bert_exp': 0.6631, 'Nextstep_Accuracy_bert_exp': 0.7994, 'Action_Accuracy_bert_exp': 0.6942, 'Value_Accuracy_bert_exp': 0.3036, 'Joint_Accuracy_bert_exp': 0.2084, 'Recall_at_1_bert_exp': 0.124, 'Recall_at_5_bert_exp': 0.3715, 'Recall_at_10_bert_exp': 0.4921, 'Turn_Accuracy_bert_exp': 0.1192, 'Cascading_Score_bert_exp': 0.0565},
{'Intent_Accuracy_bert_exp': 0.8176, 'Nextstep_Accuracy_bert_exp': 0.8149, 'Action_Accuracy_bert_exp': 0.738, 'Value_Accuracy_bert_exp': 0.3353, 'Joint_Accuracy_bert_exp': 0.229, 'Recall_at_1_bert_exp': 0.1526, 'Recall_at_5_bert_exp': 0.4064, 'Recall_at_10_bert_exp': 0.5244, 'Turn_Accuracy_bert_exp': 0.1701, 'Cascading_Score_bert_exp': 0.0729},
{'Intent_Accuracy_bert_exp': 0.8551, 'Nextstep_Accuracy_bert_exp': 0.8218, 'Action_Accuracy_bert_exp': 0.7537, 'Value_Accuracy_bert_exp': 0.3639, 'Joint_Accuracy_bert_exp': 0.2514, 'Recall_at_1_bert_exp': 0.1771, 'Recall_at_5_bert_exp': 0.4428, 'Recall_at_10_bert_exp': 0.5547, 'Turn_Accuracy_bert_exp': 0.2007, 'Cascading_Score_bert_exp': 0.0824},
{'Intent_Accuracy_bert_exp': 0.884, 'Nextstep_Accuracy_bert_exp': 0.8259, 'Action_Accuracy_bert_exp': 0.7579, 'Value_Accuracy_bert_exp': 0.3762, 'Joint_Accuracy_bert_exp': 0.2592, 'Recall_at_1_bert_exp': 0.188, 'Recall_at_5_bert_exp': 0.4545, 'Recall_at_10_bert_exp': 0.566, 'Turn_Accuracy_bert_exp': 0.2138, 'Cascading_Score_bert_exp': 0.0872},
{'Intent_Accuracy_bert_exp': 0.9071, 'Nextstep_Accuracy_bert_exp': 0.8175, 'Action_Accuracy_bert_exp': 0.7478, 'Value_Accuracy_bert_exp': 0.3793, 'Joint_Accuracy_bert_exp': 0.262, 'Recall_at_1_bert_exp': 0.1508, 'Recall_at_5_bert_exp': 0.401, 'Recall_at_10_bert_exp': 0.5208, 'Turn_Accuracy_bert_exp': 0.197, 'Cascading_Score_bert_exp': 0.0835},
{'Intent_Accuracy_bert_exp': 0.9226, 'Nextstep_Accuracy_bert_exp': 0.8255, 'Action_Accuracy_bert_exp': 0.7749, 'Value_Accuracy_bert_exp': 0.423, 'Joint_Accuracy_bert_exp': 0.2932, 'Recall_at_1_bert_exp': 0.1708, 'Recall_at_5_bert_exp': 0.4352, 'Recall_at_10_bert_exp': 0.5474, 'Turn_Accuracy_bert_exp': 0.2179, 'Cascading_Score_bert_exp': 0.0886},
{'Intent_Accuracy_bert_exp': 0.9364, 'Nextstep_Accuracy_bert_exp': 0.8361, 'Action_Accuracy_bert_exp': 0.7885, 'Value_Accuracy_bert_exp': 0.469, 'Joint_Accuracy_bert_exp': 0.324, 'Recall_at_1_bert_exp': 0.1994, 'Recall_at_5_bert_exp': 0.4632, 'Recall_at_10_bert_exp': 0.5764, 'Turn_Accuracy_bert_exp': 0.2485, 'Cascading_Score_bert_exp': 0.0976},
{'Intent_Accuracy_bert_exp': 0.9411, 'Nextstep_Accuracy_bert_exp': 0.8436, 'Action_Accuracy_bert_exp': 0.7916, 'Value_Accuracy_bert_exp': 0.4859, 'Joint_Accuracy_bert_exp': 0.3319, 'Recall_at_1_bert_exp': 0.2157, 'Recall_at_5_bert_exp': 0.4896, 'Recall_at_10_bert_exp': 0.604, 'Turn_Accuracy_bert_exp': 0.2592, 'Cascading_Score_bert_exp': 0.1022},
{'Intent_Accuracy_bert_exp': 0.9418, 'Nextstep_Accuracy_bert_exp': 0.8457, 'Action_Accuracy_bert_exp': 0.7971, 'Value_Accuracy_bert_exp': 0.5044, 'Joint_Accuracy_bert_exp': 0.3474, 'Recall_at_1_bert_exp': 0.2281, 'Recall_at_5_bert_exp': 0.5019, 'Recall_at_10_bert_exp': 0.6124, 'Turn_Accuracy_bert_exp': 0.271, 'Cascading_Score_bert_exp': 0.1064}
    ]

bert_exp_lem = [ # abcde1
{'Intent_Accuracy_bert_exp_lem': 0.4879, 'Nextstep_Accuracy_bert_exp_lem': 0.7455, 'Action_Accuracy_bert_exp_lem': 0.6571, 'Value_Accuracy_bert_exp_lem': 0.5356, 'Joint_Accuracy_bert_exp_lem': 0.1899, 'Recall_at_1_bert_exp_lem': 0.0966, 'Recall_at_5_bert_exp_lem': 0.3227, 'Recall_at_10_bert_exp_lem': 0.448, 'Turn_Accuracy_bert_exp_lem': 0.0732, 'Cascading_Score_bert_exp_lem': 0.0368},
{'Intent_Accuracy_bert_exp_lem': 0.7211, 'Nextstep_Accuracy_bert_exp_lem': 0.8009, 'Action_Accuracy_bert_exp_lem': 0.6985, 'Value_Accuracy_bert_exp_lem': 0.588, 'Joint_Accuracy_bert_exp_lem': 0.2084, 'Recall_at_1_bert_exp_lem': 0.1228, 'Recall_at_5_bert_exp_lem': 0.3724, 'Recall_at_10_bert_exp_lem': 0.4955, 'Turn_Accuracy_bert_exp_lem': 0.1287, 'Cascading_Score_bert_exp_lem': 0.0589},
{'Intent_Accuracy_bert_exp_lem': 0.8044, 'Nextstep_Accuracy_bert_exp_lem': 0.8118, 'Action_Accuracy_bert_exp_lem': 0.7423, 'Value_Accuracy_bert_exp_lem': 0.6051, 'Joint_Accuracy_bert_exp_lem': 0.215, 'Recall_at_1_bert_exp_lem': 0.1493, 'Recall_at_5_bert_exp_lem': 0.4072, 'Recall_at_10_bert_exp_lem': 0.5238, 'Turn_Accuracy_bert_exp_lem': 0.1599, 'Cascading_Score_bert_exp_lem': 0.0685},
{'Intent_Accuracy_bert_exp_lem': 0.88, 'Nextstep_Accuracy_bert_exp_lem': 0.8182, 'Action_Accuracy_bert_exp_lem': 0.7506, 'Value_Accuracy_bert_exp_lem': 0.631, 'Joint_Accuracy_bert_exp_lem': 0.226, 'Recall_at_1_bert_exp_lem': 0.1734, 'Recall_at_5_bert_exp_lem': 0.4297, 'Recall_at_10_bert_exp_lem': 0.5396, 'Turn_Accuracy_bert_exp_lem': 0.1938, 'Cascading_Score_bert_exp_lem': 0.0793},
{'Intent_Accuracy_bert_exp_lem': 0.8951, 'Nextstep_Accuracy_bert_exp_lem': 0.8268, 'Action_Accuracy_bert_exp_lem': 0.7616, 'Value_Accuracy_bert_exp_lem': 0.642, 'Joint_Accuracy_bert_exp_lem': 0.2298, 'Recall_at_1_bert_exp_lem': 0.1831, 'Recall_at_5_bert_exp_lem': 0.448, 'Recall_at_10_bert_exp_lem': 0.5585, 'Turn_Accuracy_bert_exp_lem': 0.2019, 'Cascading_Score_bert_exp_lem': 0.0807},
{'Intent_Accuracy_bert_exp_lem': 0.8898, 'Nextstep_Accuracy_bert_exp_lem': 0.8211, 'Action_Accuracy_bert_exp_lem': 0.7569, 'Value_Accuracy_bert_exp_lem': 0.647, 'Joint_Accuracy_bert_exp_lem': 0.2317, 'Recall_at_1_bert_exp_lem': 0.1515, 'Recall_at_5_bert_exp_lem': 0.4036, 'Recall_at_10_bert_exp_lem': 0.5195, 'Turn_Accuracy_bert_exp_lem': 0.1809, 'Cascading_Score_bert_exp_lem': 0.075},
{'Intent_Accuracy_bert_exp_lem': 0.9125, 'Nextstep_Accuracy_bert_exp_lem': 0.8217, 'Action_Accuracy_bert_exp_lem': 0.7622, 'Value_Accuracy_bert_exp_lem': 0.6641, 'Joint_Accuracy_bert_exp_lem': 0.2353, 'Recall_at_1_bert_exp_lem': 0.1697, 'Recall_at_5_bert_exp_lem': 0.431, 'Recall_at_10_bert_exp_lem': 0.5423, 'Turn_Accuracy_bert_exp_lem': 0.19, 'Cascading_Score_bert_exp_lem': 0.0807},
{'Intent_Accuracy_bert_exp_lem': 0.9172, 'Nextstep_Accuracy_bert_exp_lem': 0.8328, 'Action_Accuracy_bert_exp_lem': 0.7781, 'Value_Accuracy_bert_exp_lem': 0.6806, 'Joint_Accuracy_bert_exp_lem': 0.2441, 'Recall_at_1_bert_exp_lem': 0.1919, 'Recall_at_5_bert_exp_lem': 0.4516, 'Recall_at_10_bert_exp_lem': 0.5656, 'Turn_Accuracy_bert_exp_lem': 0.2093, 'Cascading_Score_bert_exp_lem': 0.0853},
{'Intent_Accuracy_bert_exp_lem': 0.9231, 'Nextstep_Accuracy_bert_exp_lem': 0.8415, 'Action_Accuracy_bert_exp_lem': 0.7834, 'Value_Accuracy_bert_exp_lem': 0.7049, 'Joint_Accuracy_bert_exp_lem': 0.2518, 'Recall_at_1_bert_exp_lem': 0.2118, 'Recall_at_5_bert_exp_lem': 0.4796, 'Recall_at_10_bert_exp_lem': 0.5896, 'Turn_Accuracy_bert_exp_lem': 0.2249, 'Cascading_Score_bert_exp_lem': 0.0903},
{'Intent_Accuracy_bert_exp_lem': 0.9267, 'Nextstep_Accuracy_bert_exp_lem': 0.8433, 'Action_Accuracy_bert_exp_lem': 0.7885, 'Value_Accuracy_bert_exp_lem': 0.7093, 'Joint_Accuracy_bert_exp_lem': 0.2543, 'Recall_at_1_bert_exp_lem': 0.2199, 'Recall_at_5_bert_exp_lem': 0.4936, 'Recall_at_10_bert_exp_lem': 0.605, 'Turn_Accuracy_bert_exp_lem': 0.2315, 'Cascading_Score_bert_exp_lem': 0.0931}
    ]

bert_exp_cor = [ # abcde3
{'Intent_Accuracy_bert_exp_cor': 0.5164, 'Nextstep_Accuracy_bert_exp_cor': 0.7953, 'Action_Accuracy_bert_exp_cor': 0.6385, 'Value_Accuracy_bert_exp_cor': 0.1831, 'Joint_Accuracy_bert_exp_cor': 0.1267, 'Recall_at_1_bert_exp_cor': 0.0959, 'Recall_at_5_bert_exp_cor': 0.3224, 'Recall_at_10_bert_exp_cor': 0.4448, 'Turn_Accuracy_bert_exp_cor': 0.0679, 'Cascading_Score_bert_exp_cor': 0.0372},
{'Intent_Accuracy_bert_exp_cor': 0.7808, 'Nextstep_Accuracy_bert_exp_cor': 0.7983, 'Action_Accuracy_bert_exp_cor': 0.6932, 'Value_Accuracy_bert_exp_cor': 0.3131, 'Joint_Accuracy_bert_exp_cor': 0.2127, 'Recall_at_1_bert_exp_cor': 0.1227, 'Recall_at_5_bert_exp_cor': 0.362, 'Recall_at_10_bert_exp_cor': 0.4831, 'Turn_Accuracy_bert_exp_cor': 0.1413, 'Cascading_Score_bert_exp_cor': 0.066},
{'Intent_Accuracy_bert_exp_cor': 0.8496, 'Nextstep_Accuracy_bert_exp_cor': 0.8069, 'Action_Accuracy_bert_exp_cor': 0.7113, 'Value_Accuracy_bert_exp_cor': 0.4039, 'Joint_Accuracy_bert_exp_cor': 0.2765, 'Recall_at_1_bert_exp_cor': 0.1443, 'Recall_at_5_bert_exp_cor': 0.4047, 'Recall_at_10_bert_exp_cor': 0.5235, 'Turn_Accuracy_bert_exp_cor': 0.1835, 'Cascading_Score_bert_exp_cor': 0.0766},
{'Intent_Accuracy_bert_exp_cor': 0.8762, 'Nextstep_Accuracy_bert_exp_cor': 0.8127, 'Action_Accuracy_bert_exp_cor': 0.7313, 'Value_Accuracy_bert_exp_cor': 0.413, 'Joint_Accuracy_bert_exp_cor': 0.2814, 'Recall_at_1_bert_exp_cor': 0.1694, 'Recall_at_5_bert_exp_cor': 0.4403, 'Recall_at_10_bert_exp_cor': 0.5554, 'Turn_Accuracy_bert_exp_cor': 0.2063, 'Cascading_Score_bert_exp_cor': 0.0826},
{'Intent_Accuracy_bert_exp_cor': 0.8839, 'Nextstep_Accuracy_bert_exp_cor': 0.8179, 'Action_Accuracy_bert_exp_cor': 0.7376, 'Value_Accuracy_bert_exp_cor': 0.4264, 'Joint_Accuracy_bert_exp_cor': 0.2916, 'Recall_at_1_bert_exp_cor': 0.1807, 'Recall_at_5_bert_exp_cor': 0.4597, 'Recall_at_10_bert_exp_cor': 0.5727, 'Turn_Accuracy_bert_exp_cor': 0.2157, 'Cascading_Score_bert_exp_cor': 0.0861},
{'Intent_Accuracy_bert_exp_cor': 0.8992, 'Nextstep_Accuracy_bert_exp_cor': 0.8132, 'Action_Accuracy_bert_exp_cor': 0.7311, 'Value_Accuracy_bert_exp_cor': 0.4256, 'Joint_Accuracy_bert_exp_cor': 0.2887, 'Recall_at_1_bert_exp_cor': 0.1434, 'Recall_at_5_bert_exp_cor': 0.3895, 'Recall_at_10_bert_exp_cor': 0.5043, 'Turn_Accuracy_bert_exp_cor': 0.1981, 'Cascading_Score_bert_exp_cor': 0.0805},
{'Intent_Accuracy_bert_exp_cor': 0.9203, 'Nextstep_Accuracy_bert_exp_cor': 0.8232, 'Action_Accuracy_bert_exp_cor': 0.7598, 'Value_Accuracy_bert_exp_cor': 0.4644, 'Joint_Accuracy_bert_exp_cor': 0.3164, 'Recall_at_1_bert_exp_cor': 0.1743, 'Recall_at_5_bert_exp_cor': 0.446, 'Recall_at_10_bert_exp_cor': 0.5625, 'Turn_Accuracy_bert_exp_cor': 0.2277, 'Cascading_Score_bert_exp_cor': 0.0894},
{'Intent_Accuracy_bert_exp_cor': 0.9243, 'Nextstep_Accuracy_bert_exp_cor': 0.8267, 'Action_Accuracy_bert_exp_cor': 0.7671, 'Value_Accuracy_bert_exp_cor': 0.4899, 'Joint_Accuracy_bert_exp_cor': 0.3337, 'Recall_at_1_bert_exp_cor': 0.1973, 'Recall_at_5_bert_exp_cor': 0.4647, 'Recall_at_10_bert_exp_cor': 0.5761, 'Turn_Accuracy_bert_exp_cor': 0.2474, 'Cascading_Score_bert_exp_cor': 0.0985},
{'Intent_Accuracy_bert_exp_cor': 0.9308, 'Nextstep_Accuracy_bert_exp_cor': 0.8309, 'Action_Accuracy_bert_exp_cor': 0.7722, 'Value_Accuracy_bert_exp_cor': 0.4967, 'Joint_Accuracy_bert_exp_cor': 0.3388, 'Recall_at_1_bert_exp_cor': 0.2184, 'Recall_at_5_bert_exp_cor': 0.4958, 'Recall_at_10_bert_exp_cor': 0.6067, 'Turn_Accuracy_bert_exp_cor': 0.2622, 'Cascading_Score_bert_exp_cor': 0.1015},
{'Intent_Accuracy_bert_exp_cor': 0.9339, 'Nextstep_Accuracy_bert_exp_cor': 0.8325, 'Action_Accuracy_bert_exp_cor': 0.7777, 'Value_Accuracy_bert_exp_cor': 0.509, 'Joint_Accuracy_bert_exp_cor': 0.3472, 'Recall_at_1_bert_exp_cor': 0.2274, 'Recall_at_5_bert_exp_cor': 0.504, 'Recall_at_10_bert_exp_cor': 0.614, 'Turn_Accuracy_bert_exp_cor': 0.2699, 'Cascading_Score_bert_exp_cor': 0.1055}
    ]

bert_exp_sen = [ # abcde2
{'Intent_Accuracy_bert_exp_sen': 0.4826, 'Nextstep_Accuracy_bert_exp_sen': 0.7782, 'Action_Accuracy_bert_exp_sen': 0.6147, 'Value_Accuracy_bert_exp_sen': 0.3526, 'Joint_Accuracy_bert_exp_sen': 0.1856, 'Recall_at_1_bert_exp_sen': 0.0932, 'Recall_at_5_bert_exp_sen': 0.3103, 'Recall_at_10_bert_exp_sen': 0.4381, 'Turn_Accuracy_bert_exp_sen': 0.0625, 'Cascading_Score_bert_exp_sen': 0.0306},
{'Intent_Accuracy_bert_exp_sen': 0.7009, 'Nextstep_Accuracy_bert_exp_sen': 0.7949, 'Action_Accuracy_bert_exp_sen': 0.7003, 'Value_Accuracy_bert_exp_sen': 0.3688, 'Joint_Accuracy_bert_exp_sen': 0.1999, 'Recall_at_1_bert_exp_sen': 0.124, 'Recall_at_5_bert_exp_sen': 0.3652, 'Recall_at_10_bert_exp_sen': 0.488, 'Turn_Accuracy_bert_exp_sen': 0.117, 'Cascading_Score_bert_exp_sen': 0.0463},
{'Intent_Accuracy_bert_exp_sen': 0.8707, 'Nextstep_Accuracy_bert_exp_sen': 0.8132, 'Action_Accuracy_bert_exp_sen': 0.7308, 'Value_Accuracy_bert_exp_sen': 0.3951, 'Joint_Accuracy_bert_exp_sen': 0.216, 'Recall_at_1_bert_exp_sen': 0.1451, 'Recall_at_5_bert_exp_sen': 0.3986, 'Recall_at_10_bert_exp_sen': 0.5203, 'Turn_Accuracy_bert_exp_sen': 0.1686, 'Cascading_Score_bert_exp_sen': 0.0688},
{'Intent_Accuracy_bert_exp_sen': 0.9101, 'Nextstep_Accuracy_bert_exp_sen': 0.8212, 'Action_Accuracy_bert_exp_sen': 0.7392, 'Value_Accuracy_bert_exp_sen': 0.4387, 'Joint_Accuracy_bert_exp_sen': 0.239, 'Recall_at_1_bert_exp_sen': 0.1623, 'Recall_at_5_bert_exp_sen': 0.4297, 'Recall_at_10_bert_exp_sen': 0.546, 'Turn_Accuracy_bert_exp_sen': 0.1964, 'Cascading_Score_bert_exp_sen': 0.0791},
{'Intent_Accuracy_bert_exp_sen': 0.9116, 'Nextstep_Accuracy_bert_exp_sen': 0.8256, 'Action_Accuracy_bert_exp_sen': 0.7419, 'Value_Accuracy_bert_exp_sen': 0.4535, 'Joint_Accuracy_bert_exp_sen': 0.2486, 'Recall_at_1_bert_exp_sen': 0.1749, 'Recall_at_5_bert_exp_sen': 0.4428, 'Recall_at_10_bert_exp_sen': 0.5589, 'Turn_Accuracy_bert_exp_sen': 0.2067, 'Cascading_Score_bert_exp_sen': 0.0832},
{'Intent_Accuracy_bert_exp_sen': 0.9085, 'Nextstep_Accuracy_bert_exp_sen': 0.8033, 'Action_Accuracy_bert_exp_sen': 0.7463, 'Value_Accuracy_bert_exp_sen': 0.465, 'Joint_Accuracy_bert_exp_sen': 0.2535, 'Recall_at_1_bert_exp_sen': 0.1521, 'Recall_at_5_bert_exp_sen': 0.3958, 'Recall_at_10_bert_exp_sen': 0.5231, 'Turn_Accuracy_bert_exp_sen': 0.1958, 'Cascading_Score_bert_exp_sen': 0.0814},
{'Intent_Accuracy_bert_exp_sen': 0.9204, 'Nextstep_Accuracy_bert_exp_sen': 0.8114, 'Action_Accuracy_bert_exp_sen': 0.7606, 'Value_Accuracy_bert_exp_sen': 0.5209, 'Joint_Accuracy_bert_exp_sen': 0.285, 'Recall_at_1_bert_exp_sen': 0.1733, 'Recall_at_5_bert_exp_sen': 0.4282, 'Recall_at_10_bert_exp_sen': 0.545, 'Turn_Accuracy_bert_exp_sen': 0.2092, 'Cascading_Score_bert_exp_sen': 0.0812},
{'Intent_Accuracy_bert_exp_sen': 0.9279, 'Nextstep_Accuracy_bert_exp_sen': 0.8224, 'Action_Accuracy_bert_exp_sen': 0.7639, 'Value_Accuracy_bert_exp_sen': 0.5292, 'Joint_Accuracy_bert_exp_sen': 0.2891, 'Recall_at_1_bert_exp_sen': 0.1932, 'Recall_at_5_bert_exp_sen': 0.4536, 'Recall_at_10_bert_exp_sen': 0.5706, 'Turn_Accuracy_bert_exp_sen': 0.2285, 'Cascading_Score_bert_exp_sen': 0.0907},
{'Intent_Accuracy_bert_exp_sen': 0.9333, 'Nextstep_Accuracy_bert_exp_sen': 0.8298, 'Action_Accuracy_bert_exp_sen': 0.7726, 'Value_Accuracy_bert_exp_sen': 0.5386, 'Joint_Accuracy_bert_exp_sen': 0.2958, 'Recall_at_1_bert_exp_sen': 0.2027, 'Recall_at_5_bert_exp_sen': 0.4767, 'Recall_at_10_bert_exp_sen': 0.591, 'Turn_Accuracy_bert_exp_sen': 0.2387, 'Cascading_Score_bert_exp_sen': 0.0948},
{'Intent_Accuracy_bert_exp_sen': 0.9359, 'Nextstep_Accuracy_bert_exp_sen': 0.834, 'Action_Accuracy_bert_exp_sen': 0.7779, 'Value_Accuracy_bert_exp_sen': 0.5508, 'Joint_Accuracy_bert_exp_sen': 0.3028, 'Recall_at_1_bert_exp_sen': 0.2166, 'Recall_at_5_bert_exp_sen': 0.489, 'Recall_at_10_bert_exp_sen': 0.6041, 'Turn_Accuracy_bert_exp_sen': 0.2492, 'Cascading_Score_bert_exp_sen': 0.0982}
     ]

bert_exp_cor_lem = [ # abcde1
{'Intent_Accuracy_bert_exp_cor_lem': 0.4072, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.7968, 'Action_Accuracy_bert_exp_cor_lem': 0.6306, 'Value_Accuracy_bert_exp_cor_lem': 0.5538, 'Joint_Accuracy_bert_exp_cor_lem': 0.1985, 'Recall_at_1_bert_exp_cor_lem': 0.0955, 'Recall_at_5_bert_exp_cor_lem': 0.3116, 'Recall_at_10_bert_exp_cor_lem': 0.4374, 'Turn_Accuracy_bert_exp_cor_lem': 0.0618, 'Cascading_Score_bert_exp_cor_lem': 0.0293},
{'Intent_Accuracy_bert_exp_cor_lem': 0.6612, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.8056, 'Action_Accuracy_bert_exp_cor_lem': 0.6783, 'Value_Accuracy_bert_exp_cor_lem': 0.5736, 'Joint_Accuracy_bert_exp_cor_lem': 0.2027, 'Recall_at_1_bert_exp_cor_lem': 0.1276, 'Recall_at_5_bert_exp_cor_lem': 0.3728, 'Recall_at_10_bert_exp_cor_lem': 0.4936, 'Turn_Accuracy_bert_exp_cor_lem': 0.1155, 'Cascading_Score_bert_exp_cor_lem': 0.0525},
{'Intent_Accuracy_bert_exp_cor_lem': 0.8034, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.8104, 'Action_Accuracy_bert_exp_cor_lem': 0.7172, 'Value_Accuracy_bert_exp_cor_lem': 0.6183, 'Joint_Accuracy_bert_exp_cor_lem': 0.2188, 'Recall_at_1_bert_exp_cor_lem': 0.1425, 'Recall_at_5_bert_exp_cor_lem': 0.4006, 'Recall_at_10_bert_exp_cor_lem': 0.5229, 'Turn_Accuracy_bert_exp_cor_lem': 0.1552, 'Cascading_Score_bert_exp_cor_lem': 0.0656},
{'Intent_Accuracy_bert_exp_cor_lem': 0.8671, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.8178, 'Action_Accuracy_bert_exp_cor_lem': 0.7325, 'Value_Accuracy_bert_exp_cor_lem': 0.6327, 'Joint_Accuracy_bert_exp_cor_lem': 0.2245, 'Recall_at_1_bert_exp_cor_lem': 0.1635, 'Recall_at_5_bert_exp_cor_lem': 0.4303, 'Recall_at_10_bert_exp_cor_lem': 0.5537, 'Turn_Accuracy_bert_exp_cor_lem': 0.1822, 'Cascading_Score_bert_exp_cor_lem': 0.0738},
{'Intent_Accuracy_bert_exp_cor_lem': 0.8682, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.8214, 'Action_Accuracy_bert_exp_cor_lem': 0.737, 'Value_Accuracy_bert_exp_cor_lem': 0.6376, 'Joint_Accuracy_bert_exp_cor_lem': 0.2262, 'Recall_at_1_bert_exp_cor_lem': 0.1758, 'Recall_at_5_bert_exp_cor_lem': 0.4465, 'Recall_at_10_bert_exp_cor_lem': 0.564, 'Turn_Accuracy_bert_exp_cor_lem': 0.1884, 'Cascading_Score_bert_exp_cor_lem': 0.0759},
{'Intent_Accuracy_bert_exp_cor_lem': 0.873, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.8133, 'Action_Accuracy_bert_exp_cor_lem': 0.7494, 'Value_Accuracy_bert_exp_cor_lem': 0.647, 'Joint_Accuracy_bert_exp_cor_lem': 0.2282, 'Recall_at_1_bert_exp_cor_lem': 0.1492, 'Recall_at_5_bert_exp_cor_lem': 0.397, 'Recall_at_10_bert_exp_cor_lem': 0.5166, 'Turn_Accuracy_bert_exp_cor_lem': 0.1744, 'Cascading_Score_bert_exp_cor_lem': 0.0726},
{'Intent_Accuracy_bert_exp_cor_lem': 0.9103, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.8164, 'Action_Accuracy_bert_exp_cor_lem': 0.7714, 'Value_Accuracy_bert_exp_cor_lem': 0.674, 'Joint_Accuracy_bert_exp_cor_lem': 0.2404, 'Recall_at_1_bert_exp_cor_lem': 0.1701, 'Recall_at_5_bert_exp_cor_lem': 0.4211, 'Recall_at_10_bert_exp_cor_lem': 0.5293, 'Turn_Accuracy_bert_exp_cor_lem': 0.1978, 'Cascading_Score_bert_exp_cor_lem': 0.0798},
{'Intent_Accuracy_bert_exp_cor_lem': 0.9222, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.8254, 'Action_Accuracy_bert_exp_cor_lem': 0.7804, 'Value_Accuracy_bert_exp_cor_lem': 0.6768, 'Joint_Accuracy_bert_exp_cor_lem': 0.2419, 'Recall_at_1_bert_exp_cor_lem': 0.1874, 'Recall_at_5_bert_exp_cor_lem': 0.4451, 'Recall_at_10_bert_exp_cor_lem': 0.5532, 'Turn_Accuracy_bert_exp_cor_lem': 0.2094, 'Cascading_Score_bert_exp_cor_lem': 0.0838},
{'Intent_Accuracy_bert_exp_cor_lem': 0.926, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.8286, 'Action_Accuracy_bert_exp_cor_lem': 0.7785, 'Value_Accuracy_bert_exp_cor_lem': 0.6795, 'Joint_Accuracy_bert_exp_cor_lem': 0.2419, 'Recall_at_1_bert_exp_cor_lem': 0.2049, 'Recall_at_5_bert_exp_cor_lem': 0.4746, 'Recall_at_10_bert_exp_cor_lem': 0.5876, 'Turn_Accuracy_bert_exp_cor_lem': 0.2209, 'Cascading_Score_bert_exp_cor_lem': 0.0875},
{'Intent_Accuracy_bert_exp_cor_lem': 0.9273, 'Nextstep_Accuracy_bert_exp_cor_lem': 0.8354, 'Action_Accuracy_bert_exp_cor_lem': 0.7808, 'Value_Accuracy_bert_exp_cor_lem': 0.6784, 'Joint_Accuracy_bert_exp_cor_lem': 0.2425, 'Recall_at_1_bert_exp_cor_lem': 0.2183, 'Recall_at_5_bert_exp_cor_lem': 0.487, 'Recall_at_10_bert_exp_cor_lem': 0.5975, 'Turn_Accuracy_bert_exp_cor_lem': 0.2291, 'Cascading_Score_bert_exp_cor_lem': 0.0907}
     ]

bert_exp_cor_sen = [ # abcde1
{'Intent_Accuracy_bert_exp_cor_sen': 0.4993, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.7846, 'Action_Accuracy_bert_exp_cor_sen': 0.639, 'Value_Accuracy_bert_exp_cor_sen': 0.3556, 'Joint_Accuracy_bert_exp_cor_sen': 0.1925, 'Recall_at_1_bert_exp_cor_sen': 0.1011, 'Recall_at_5_bert_exp_cor_sen': 0.3226, 'Recall_at_10_bert_exp_cor_sen': 0.4414, 'Turn_Accuracy_bert_exp_cor_sen': 0.0742, 'Cascading_Score_bert_exp_cor_sen': 0.0354},
{'Intent_Accuracy_bert_exp_cor_sen': 0.7516, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.7917, 'Action_Accuracy_bert_exp_cor_sen': 0.698, 'Value_Accuracy_bert_exp_cor_sen': 0.3787, 'Joint_Accuracy_bert_exp_cor_sen': 0.2056, 'Recall_at_1_bert_exp_cor_sen': 0.1266, 'Recall_at_5_bert_exp_cor_sen': 0.3714, 'Recall_at_10_bert_exp_cor_sen': 0.4888, 'Turn_Accuracy_bert_exp_cor_sen': 0.127, 'Cascading_Score_bert_exp_cor_sen': 0.054},
{'Intent_Accuracy_bert_exp_cor_sen': 0.808, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.8065, 'Action_Accuracy_bert_exp_cor_sen': 0.7178, 'Value_Accuracy_bert_exp_cor_sen': 0.4299, 'Joint_Accuracy_bert_exp_cor_sen': 0.2355, 'Recall_at_1_bert_exp_cor_sen': 0.1541, 'Recall_at_5_bert_exp_cor_sen': 0.4105, 'Recall_at_10_bert_exp_cor_sen': 0.5272, 'Turn_Accuracy_bert_exp_cor_sen': 0.1647, 'Cascading_Score_bert_exp_cor_sen': 0.0681},
{'Intent_Accuracy_bert_exp_cor_sen': 0.868, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.8145, 'Action_Accuracy_bert_exp_cor_sen': 0.7278, 'Value_Accuracy_bert_exp_cor_sen': 0.4526, 'Joint_Accuracy_bert_exp_cor_sen': 0.2463, 'Recall_at_1_bert_exp_cor_sen': 0.1708, 'Recall_at_5_bert_exp_cor_sen': 0.4332, 'Recall_at_10_bert_exp_cor_sen': 0.5514, 'Turn_Accuracy_bert_exp_cor_sen': 0.1897, 'Cascading_Score_bert_exp_cor_sen': 0.0765},
{'Intent_Accuracy_bert_exp_cor_sen': 0.8859, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.8165, 'Action_Accuracy_bert_exp_cor_sen': 0.7306, 'Value_Accuracy_bert_exp_cor_sen': 0.4583, 'Joint_Accuracy_bert_exp_cor_sen': 0.2496, 'Recall_at_1_bert_exp_cor_sen': 0.1781, 'Recall_at_5_bert_exp_cor_sen': 0.4428, 'Recall_at_10_bert_exp_cor_sen': 0.5561, 'Turn_Accuracy_bert_exp_cor_sen': 0.1997, 'Cascading_Score_bert_exp_cor_sen': 0.0797},
{'Intent_Accuracy_bert_exp_cor_sen': 0.8797, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.802, 'Action_Accuracy_bert_exp_cor_sen': 0.7229, 'Value_Accuracy_bert_exp_cor_sen': 0.4688, 'Joint_Accuracy_bert_exp_cor_sen': 0.2577, 'Recall_at_1_bert_exp_cor_sen': 0.1474, 'Recall_at_5_bert_exp_cor_sen': 0.3895, 'Recall_at_10_bert_exp_cor_sen': 0.5109, 'Turn_Accuracy_bert_exp_cor_sen': 0.1859, 'Cascading_Score_bert_exp_cor_sen': 0.0772},
{'Intent_Accuracy_bert_exp_cor_sen': 0.928, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.8151, 'Action_Accuracy_bert_exp_cor_sen': 0.75, 'Value_Accuracy_bert_exp_cor_sen': 0.5002, 'Joint_Accuracy_bert_exp_cor_sen': 0.2742, 'Recall_at_1_bert_exp_cor_sen': 0.1672, 'Recall_at_5_bert_exp_cor_sen': 0.4208, 'Recall_at_10_bert_exp_cor_sen': 0.5389, 'Turn_Accuracy_bert_exp_cor_sen': 0.2118, 'Cascading_Score_bert_exp_cor_sen': 0.0848},
{'Intent_Accuracy_bert_exp_cor_sen': 0.9312, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.8179, 'Action_Accuracy_bert_exp_cor_sen': 0.7561, 'Value_Accuracy_bert_exp_cor_sen': 0.537, 'Joint_Accuracy_bert_exp_cor_sen': 0.2924, 'Recall_at_1_bert_exp_cor_sen': 0.1798, 'Recall_at_5_bert_exp_cor_sen': 0.4404, 'Recall_at_10_bert_exp_cor_sen': 0.5516, 'Turn_Accuracy_bert_exp_cor_sen': 0.2243, 'Cascading_Score_bert_exp_cor_sen': 0.0887},
{'Intent_Accuracy_bert_exp_cor_sen': 0.9358, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.8283, 'Action_Accuracy_bert_exp_cor_sen': 0.7685, 'Value_Accuracy_bert_exp_cor_sen': 0.5536, 'Joint_Accuracy_bert_exp_cor_sen': 0.3036, 'Recall_at_1_bert_exp_cor_sen': 0.1939, 'Recall_at_5_bert_exp_cor_sen': 0.4644, 'Recall_at_10_bert_exp_cor_sen': 0.5784, 'Turn_Accuracy_bert_exp_cor_sen': 0.2374, 'Cascading_Score_bert_exp_cor_sen': 0.0935},
{'Intent_Accuracy_bert_exp_cor_sen': 0.9365, 'Nextstep_Accuracy_bert_exp_cor_sen': 0.8297, 'Action_Accuracy_bert_exp_cor_sen': 0.7751, 'Value_Accuracy_bert_exp_cor_sen': 0.5611, 'Joint_Accuracy_bert_exp_cor_sen': 0.3077, 'Recall_at_1_bert_exp_cor_sen': 0.2023, 'Recall_at_5_bert_exp_cor_sen': 0.4682, 'Recall_at_10_bert_exp_cor_sen': 0.583, 'Turn_Accuracy_bert_exp_cor_sen': 0.2434, 'Cascading_Score_bert_exp_cor_sen': 0.096}
     ]

bert_exp_sen_lem = [ # abcde3
{'Intent_Accuracy_bert_exp_sen_lem': 0.4698, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.7376, 'Action_Accuracy_bert_exp_sen_lem': 0.6066, 'Value_Accuracy_bert_exp_sen_lem': 0.5085, 'Joint_Accuracy_bert_exp_sen_lem': 0.1752, 'Recall_at_1_bert_exp_sen_lem': 0.0922, 'Recall_at_5_bert_exp_sen_lem': 0.3109, 'Recall_at_10_bert_exp_sen_lem': 0.4271, 'Turn_Accuracy_bert_exp_sen_lem': 0.042, 'Cascading_Score_bert_exp_sen_lem': 0.007},
{'Intent_Accuracy_bert_exp_sen_lem': 0.6612, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.785, 'Action_Accuracy_bert_exp_sen_lem': 0.6877, 'Value_Accuracy_bert_exp_sen_lem': 0.5527, 'Joint_Accuracy_bert_exp_sen_lem': 0.1952, 'Recall_at_1_bert_exp_sen_lem': 0.1215, 'Recall_at_5_bert_exp_sen_lem': 0.3568, 'Recall_at_10_bert_exp_sen_lem': 0.4666, 'Turn_Accuracy_bert_exp_sen_lem': 0.1043, 'Cascading_Score_bert_exp_sen_lem': 0.0435},
{'Intent_Accuracy_bert_exp_sen_lem': 0.7733, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.8037, 'Action_Accuracy_bert_exp_sen_lem': 0.7017, 'Value_Accuracy_bert_exp_sen_lem': 0.5714, 'Joint_Accuracy_bert_exp_sen_lem': 0.2021, 'Recall_at_1_bert_exp_sen_lem': 0.1406, 'Recall_at_5_bert_exp_sen_lem': 0.3904, 'Recall_at_10_bert_exp_sen_lem': 0.5086, 'Turn_Accuracy_bert_exp_sen_lem': 0.1407, 'Cascading_Score_bert_exp_sen_lem': 0.0584},
{'Intent_Accuracy_bert_exp_sen_lem': 0.8305, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.8104, 'Action_Accuracy_bert_exp_sen_lem': 0.7215, 'Value_Accuracy_bert_exp_sen_lem': 0.604, 'Joint_Accuracy_bert_exp_sen_lem': 0.2143, 'Recall_at_1_bert_exp_sen_lem': 0.159, 'Recall_at_5_bert_exp_sen_lem': 0.4227, 'Recall_at_10_bert_exp_sen_lem': 0.5375, 'Turn_Accuracy_bert_exp_sen_lem': 0.1647, 'Cascading_Score_bert_exp_sen_lem': 0.0683},
{'Intent_Accuracy_bert_exp_sen_lem': 0.8405, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.8181, 'Action_Accuracy_bert_exp_sen_lem': 0.7211, 'Value_Accuracy_bert_exp_sen_lem': 0.6034, 'Joint_Accuracy_bert_exp_sen_lem': 0.2141, 'Recall_at_1_bert_exp_sen_lem': 0.1671, 'Recall_at_5_bert_exp_sen_lem': 0.4328, 'Recall_at_10_bert_exp_sen_lem': 0.5466, 'Turn_Accuracy_bert_exp_sen_lem': 0.173, 'Cascading_Score_bert_exp_sen_lem': 0.0719},
{'Intent_Accuracy_bert_exp_sen_lem': 0.8646, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.8002, 'Action_Accuracy_bert_exp_sen_lem': 0.7243, 'Value_Accuracy_bert_exp_sen_lem': 0.5985, 'Joint_Accuracy_bert_exp_sen_lem': 0.2107, 'Recall_at_1_bert_exp_sen_lem': 0.1406, 'Recall_at_5_bert_exp_sen_lem': 0.3848, 'Recall_at_10_bert_exp_sen_lem': 0.5072, 'Turn_Accuracy_bert_exp_sen_lem': 0.1588, 'Cascading_Score_bert_exp_sen_lem': 0.0653},
{'Intent_Accuracy_bert_exp_sen_lem': 0.907, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.8148, 'Action_Accuracy_bert_exp_sen_lem': 0.7357, 'Value_Accuracy_bert_exp_sen_lem': 0.6117, 'Joint_Accuracy_bert_exp_sen_lem': 0.2182, 'Recall_at_1_bert_exp_sen_lem': 0.1634, 'Recall_at_5_bert_exp_sen_lem': 0.4197, 'Recall_at_10_bert_exp_sen_lem': 0.5307, 'Turn_Accuracy_bert_exp_sen_lem': 0.188, 'Cascading_Score_bert_exp_sen_lem': 0.0776},
{'Intent_Accuracy_bert_exp_sen_lem': 0.9216, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.8208, 'Action_Accuracy_bert_exp_sen_lem': 0.7559, 'Value_Accuracy_bert_exp_sen_lem': 0.6431, 'Joint_Accuracy_bert_exp_sen_lem': 0.2278, 'Recall_at_1_bert_exp_sen_lem': 0.1835, 'Recall_at_5_bert_exp_sen_lem': 0.4446, 'Recall_at_10_bert_exp_sen_lem': 0.5594, 'Turn_Accuracy_bert_exp_sen_lem': 0.2033, 'Cascading_Score_bert_exp_sen_lem': 0.0824},
{'Intent_Accuracy_bert_exp_sen_lem': 0.9315, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.8294, 'Action_Accuracy_bert_exp_sen_lem': 0.7531, 'Value_Accuracy_bert_exp_sen_lem': 0.6575, 'Joint_Accuracy_bert_exp_sen_lem': 0.2327, 'Recall_at_1_bert_exp_sen_lem': 0.195, 'Recall_at_5_bert_exp_sen_lem': 0.4691, 'Recall_at_10_bert_exp_sen_lem': 0.5873, 'Turn_Accuracy_bert_exp_sen_lem': 0.2142, 'Cascading_Score_bert_exp_sen_lem': 0.0866},
{'Intent_Accuracy_bert_exp_sen_lem': 0.9316, 'Nextstep_Accuracy_bert_exp_sen_lem': 0.8299, 'Action_Accuracy_bert_exp_sen_lem': 0.7565, 'Value_Accuracy_bert_exp_sen_lem': 0.6597, 'Joint_Accuracy_bert_exp_sen_lem': 0.2335, 'Recall_at_1_bert_exp_sen_lem': 0.2049, 'Recall_at_5_bert_exp_sen_lem': 0.4797, 'Recall_at_10_bert_exp_sen_lem': 0.5971, 'Turn_Accuracy_bert_exp_sen_lem': 0.2204, 'Cascading_Score_bert_exp_sen_lem': 0.0888}
     ]

bert_exp_cor_sen_lem = [ # abcde2
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.4957, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.7384, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.6167, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.5747, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.1864, 'Recall_at_1_bert_exp_cor_sen_lem': 0.0928, 'Recall_at_5_bert_exp_cor_sen_lem': 0.3215, 'Recall_at_10_bert_exp_cor_sen_lem': 0.4464, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.0528, 'Cascading_Score_bert_exp_cor_sen_lem': 0.018},
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.7113, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.7961, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.6756, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.5609, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.1999, 'Recall_at_1_bert_exp_cor_sen_lem': 0.1266, 'Recall_at_5_bert_exp_cor_sen_lem': 0.3739, 'Recall_at_10_bert_exp_cor_sen_lem': 0.4955, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.1175, 'Cascading_Score_bert_exp_cor_sen_lem': 0.0495},
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.8088, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.8021, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.7129, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.6222, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.2229, 'Recall_at_1_bert_exp_cor_sen_lem': 0.147, 'Recall_at_5_bert_exp_cor_sen_lem': 0.4023, 'Recall_at_10_bert_exp_cor_sen_lem': 0.5242, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.1536, 'Cascading_Score_bert_exp_cor_sen_lem': 0.0606},
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.8631, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.8109, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.7386, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.652, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.2315, 'Recall_at_1_bert_exp_cor_sen_lem': 0.1611, 'Recall_at_5_bert_exp_cor_sen_lem': 0.4264, 'Recall_at_10_bert_exp_cor_sen_lem': 0.55, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.1733, 'Cascading_Score_bert_exp_cor_sen_lem': 0.0684},
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.8931, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.8167, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.7449, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.6498, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.2313, 'Recall_at_1_bert_exp_cor_sen_lem': 0.1707, 'Recall_at_5_bert_exp_cor_sen_lem': 0.4342, 'Recall_at_10_bert_exp_cor_sen_lem': 0.5541, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.1886, 'Cascading_Score_bert_exp_cor_sen_lem': 0.0734},
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.9034, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.7989, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.748, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.6613, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.2357, 'Recall_at_1_bert_exp_cor_sen_lem': 0.147, 'Recall_at_5_bert_exp_cor_sen_lem': 0.3918, 'Recall_at_10_bert_exp_cor_sen_lem': 0.507, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.1763, 'Cascading_Score_bert_exp_cor_sen_lem': 0.0695},
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.922, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.8145, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.7541, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.6553, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.2349, 'Recall_at_1_bert_exp_cor_sen_lem': 0.1654, 'Recall_at_5_bert_exp_cor_sen_lem': 0.4122, 'Recall_at_10_bert_exp_cor_sen_lem': 0.5318, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.1897, 'Cascading_Score_bert_exp_cor_sen_lem': 0.0734},
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.9235, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.8145, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.759, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.6597, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.2361, 'Recall_at_1_bert_exp_cor_sen_lem': 0.1779, 'Recall_at_5_bert_exp_cor_sen_lem': 0.4419, 'Recall_at_10_bert_exp_cor_sen_lem': 0.5598, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.1991, 'Cascading_Score_bert_exp_cor_sen_lem': 0.0774},
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.9236, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.8243, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.7614, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.6657, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.2376, 'Recall_at_1_bert_exp_cor_sen_lem': 0.1898, 'Recall_at_5_bert_exp_cor_sen_lem': 0.4633, 'Recall_at_10_bert_exp_cor_sen_lem': 0.5763, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.2081, 'Cascading_Score_bert_exp_cor_sen_lem': 0.0786},
{'Intent_Accuracy_bert_exp_cor_sen_lem': 0.9231, 'Nextstep_Accuracy_bert_exp_cor_sen_lem': 0.8267, 'Action_Accuracy_bert_exp_cor_sen_lem': 0.7669, 'Value_Accuracy_bert_exp_cor_sen_lem': 0.6663, 'Joint_Accuracy_bert_exp_cor_sen_lem': 0.2388, 'Recall_at_1_bert_exp_cor_sen_lem': 0.1959, 'Recall_at_5_bert_exp_cor_sen_lem': 0.4673, 'Recall_at_10_bert_exp_cor_sen_lem': 0.587, 'Turn_Accuracy_bert_exp_cor_sen_lem': 0.2107, 'Cascading_Score_bert_exp_cor_sen_lem': 0.0803}
    ]

bert_cor_sen = [ # abcde1
{'Intent_Accuracy_bert_cor_sen': 0.5154, 'Nextstep_Accuracy_bert_cor_sen': 0.7808, 'Action_Accuracy_bert_cor_sen': 0.633, 'Value_Accuracy_bert_cor_sen': 0.3255, 'Joint_Accuracy_bert_cor_sen': 0.1775, 'Recall_at_1_bert_cor_sen': 0.0951, 'Recall_at_5_bert_cor_sen': 0.3156, 'Recall_at_10_bert_cor_sen': 0.4378, 'Turn_Accuracy_bert_cor_sen': 0.0703, 'Cascading_Score_bert_cor_sen': 0.0324},
{'Intent_Accuracy_bert_cor_sen': 0.7381, 'Nextstep_Accuracy_bert_cor_sen': 0.7967, 'Action_Accuracy_bert_cor_sen': 0.6811, 'Value_Accuracy_bert_cor_sen': 0.3736, 'Joint_Accuracy_bert_cor_sen': 0.1993, 'Recall_at_1_bert_cor_sen': 0.1234, 'Recall_at_5_bert_cor_sen': 0.3692, 'Recall_at_10_bert_cor_sen': 0.4873, 'Turn_Accuracy_bert_cor_sen': 0.1262, 'Cascading_Score_bert_cor_sen': 0.0554},
{'Intent_Accuracy_bert_cor_sen': 0.8965, 'Nextstep_Accuracy_bert_cor_sen': 0.8109, 'Action_Accuracy_bert_cor_sen': 0.7349, 'Value_Accuracy_bert_cor_sen': 0.4119, 'Joint_Accuracy_bert_cor_sen': 0.2231, 'Recall_at_1_bert_cor_sen': 0.1479, 'Recall_at_5_bert_cor_sen': 0.4021, 'Recall_at_10_bert_cor_sen': 0.5202, 'Turn_Accuracy_bert_cor_sen': 0.1721, 'Cascading_Score_bert_cor_sen': 0.0708},
{'Intent_Accuracy_bert_cor_sen': 0.906, 'Nextstep_Accuracy_bert_cor_sen': 0.8234, 'Action_Accuracy_bert_cor_sen': 0.7494, 'Value_Accuracy_bert_cor_sen': 0.4333, 'Joint_Accuracy_bert_cor_sen': 0.2372, 'Recall_at_1_bert_cor_sen': 0.1659, 'Recall_at_5_bert_cor_sen': 0.4302, 'Recall_at_10_bert_cor_sen': 0.5485, 'Turn_Accuracy_bert_cor_sen': 0.1916, 'Cascading_Score_bert_cor_sen': 0.0776},
{'Intent_Accuracy_bert_cor_sen': 0.9089, 'Nextstep_Accuracy_bert_cor_sen': 0.8303, 'Action_Accuracy_bert_cor_sen': 0.7518, 'Value_Accuracy_bert_cor_sen': 0.4376, 'Joint_Accuracy_bert_cor_sen': 0.2388, 'Recall_at_1_bert_cor_sen': 0.1752, 'Recall_at_5_bert_cor_sen': 0.4449, 'Recall_at_10_bert_cor_sen': 0.5601, 'Turn_Accuracy_bert_cor_sen': 0.2015, 'Cascading_Score_bert_cor_sen': 0.0817},
{'Intent_Accuracy_bert_cor_sen': 0.9053, 'Nextstep_Accuracy_bert_cor_sen': 0.8098, 'Action_Accuracy_bert_cor_sen': 0.7437, 'Value_Accuracy_bert_cor_sen': 0.4398, 'Joint_Accuracy_bert_cor_sen': 0.2398, 'Recall_at_1_bert_cor_sen': 0.1523, 'Recall_at_5_bert_cor_sen': 0.3982, 'Recall_at_10_bert_cor_sen': 0.5179, 'Turn_Accuracy_bert_cor_sen': 0.1865, 'Cascading_Score_bert_cor_sen': 0.0777},
{'Intent_Accuracy_bert_cor_sen': 0.917, 'Nextstep_Accuracy_bert_cor_sen': 0.8209, 'Action_Accuracy_bert_cor_sen': 0.7704, 'Value_Accuracy_bert_cor_sen': 0.4745, 'Joint_Accuracy_bert_cor_sen': 0.2592, 'Recall_at_1_bert_cor_sen': 0.173, 'Recall_at_5_bert_cor_sen': 0.4243, 'Recall_at_10_bert_cor_sen': 0.5336, 'Turn_Accuracy_bert_cor_sen': 0.2027, 'Cascading_Score_bert_cor_sen': 0.0814},
{'Intent_Accuracy_bert_cor_sen': 0.9305, 'Nextstep_Accuracy_bert_cor_sen': 0.8292, 'Action_Accuracy_bert_cor_sen': 0.7736, 'Value_Accuracy_bert_cor_sen': 0.4911, 'Joint_Accuracy_bert_cor_sen': 0.2669, 'Recall_at_1_bert_cor_sen': 0.1879, 'Recall_at_5_bert_cor_sen': 0.4452, 'Recall_at_10_bert_cor_sen': 0.5591, 'Turn_Accuracy_bert_cor_sen': 0.2189, 'Cascading_Score_bert_cor_sen': 0.0865},
{'Intent_Accuracy_bert_cor_sen': 0.9357, 'Nextstep_Accuracy_bert_cor_sen': 0.8338, 'Action_Accuracy_bert_cor_sen': 0.7838, 'Value_Accuracy_bert_cor_sen': 0.5212, 'Joint_Accuracy_bert_cor_sen': 0.2846, 'Recall_at_1_bert_cor_sen': 0.1985, 'Recall_at_5_bert_cor_sen': 0.4666, 'Recall_at_10_bert_cor_sen': 0.5841, 'Turn_Accuracy_bert_cor_sen': 0.2352, 'Cascading_Score_bert_cor_sen': 0.0945},
{'Intent_Accuracy_bert_cor_sen': 0.9359, 'Nextstep_Accuracy_bert_cor_sen': 0.8361, 'Action_Accuracy_bert_cor_sen': 0.7834, 'Value_Accuracy_bert_cor_sen': 0.5342, 'Joint_Accuracy_bert_cor_sen': 0.291, 'Recall_at_1_bert_cor_sen': 0.2105, 'Recall_at_5_bert_cor_sen': 0.4805, 'Recall_at_10_bert_cor_sen': 0.5982, 'Turn_Accuracy_bert_cor_sen': 0.2441, 'Cascading_Score_bert_cor_sen': 0.0973}
    ] 

bert_cor_sen_lem = [ # abcde3
{'Intent_Accuracy_bert_cor_sen_lem': 0.4303, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.7562, 'Action_Accuracy_bert_cor_sen_lem': 0.631, 'Value_Accuracy_bert_cor_sen_lem': 0.5488, 'Joint_Accuracy_bert_cor_sen_lem': 0.1946, 'Recall_at_1_bert_cor_sen_lem': 0.0899, 'Recall_at_5_bert_cor_sen_lem': 0.3032, 'Recall_at_10_bert_cor_sen_lem': 0.4284, 'Turn_Accuracy_bert_cor_sen_lem': 0.0578, 'Cascading_Score_bert_cor_sen_lem': 0.0276},
{'Intent_Accuracy_bert_cor_sen_lem': 0.6937, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.7924, 'Action_Accuracy_bert_cor_sen_lem': 0.687, 'Value_Accuracy_bert_cor_sen_lem': 0.5703, 'Joint_Accuracy_bert_cor_sen_lem': 0.204, 'Recall_at_1_bert_cor_sen_lem': 0.1253, 'Recall_at_5_bert_cor_sen_lem': 0.3782, 'Recall_at_10_bert_cor_sen_lem': 0.4968, 'Turn_Accuracy_bert_cor_sen_lem': 0.1184, 'Cascading_Score_bert_cor_sen_lem': 0.0504},
{'Intent_Accuracy_bert_cor_sen_lem': 0.7423, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.811, 'Action_Accuracy_bert_cor_sen_lem': 0.7164, 'Value_Accuracy_bert_cor_sen_lem': 0.6045, 'Joint_Accuracy_bert_cor_sen_lem': 0.2158, 'Recall_at_1_bert_cor_sen_lem': 0.1521, 'Recall_at_5_bert_cor_sen_lem': 0.4078, 'Recall_at_10_bert_cor_sen_lem': 0.5234, 'Turn_Accuracy_bert_cor_sen_lem': 0.1437, 'Cascading_Score_bert_cor_sen_lem': 0.0608},
{'Intent_Accuracy_bert_cor_sen_lem': 0.7949, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.8178, 'Action_Accuracy_bert_cor_sen_lem': 0.7306, 'Value_Accuracy_bert_cor_sen_lem': 0.62, 'Joint_Accuracy_bert_cor_sen_lem': 0.2196, 'Recall_at_1_bert_cor_sen_lem': 0.167, 'Recall_at_5_bert_cor_sen_lem': 0.4323, 'Recall_at_10_bert_cor_sen_lem': 0.5508, 'Turn_Accuracy_bert_cor_sen_lem': 0.166, 'Cascading_Score_bert_cor_sen_lem': 0.0714},
{'Intent_Accuracy_bert_cor_sen_lem': 0.8294, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.8219, 'Action_Accuracy_bert_cor_sen_lem': 0.7421, 'Value_Accuracy_bert_cor_sen_lem': 0.6216, 'Joint_Accuracy_bert_cor_sen_lem': 0.2215, 'Recall_at_1_bert_cor_sen_lem': 0.1809, 'Recall_at_5_bert_cor_sen_lem': 0.4444, 'Recall_at_10_bert_cor_sen_lem': 0.5596, 'Turn_Accuracy_bert_cor_sen_lem': 0.1811, 'Cascading_Score_bert_cor_sen_lem': 0.0763},
{'Intent_Accuracy_bert_cor_sen_lem': 0.8794, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.8132, 'Action_Accuracy_bert_cor_sen_lem': 0.7435, 'Value_Accuracy_bert_cor_sen_lem': 0.6167, 'Joint_Accuracy_bert_cor_sen_lem': 0.22, 'Recall_at_1_bert_cor_sen_lem': 0.1529, 'Recall_at_5_bert_cor_sen_lem': 0.4064, 'Recall_at_10_bert_cor_sen_lem': 0.5277, 'Turn_Accuracy_bert_cor_sen_lem': 0.1703, 'Cascading_Score_bert_cor_sen_lem': 0.0706},
{'Intent_Accuracy_bert_cor_sen_lem': 0.9205, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.8118, 'Action_Accuracy_bert_cor_sen_lem': 0.7573, 'Value_Accuracy_bert_cor_sen_lem': 0.6277, 'Joint_Accuracy_bert_cor_sen_lem': 0.2237, 'Recall_at_1_bert_cor_sen_lem': 0.1718, 'Recall_at_5_bert_cor_sen_lem': 0.4317, 'Recall_at_10_bert_cor_sen_lem': 0.5391, 'Turn_Accuracy_bert_cor_sen_lem': 0.1946, 'Cascading_Score_bert_cor_sen_lem': 0.0788},
{'Intent_Accuracy_bert_cor_sen_lem': 0.9243, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.8212, 'Action_Accuracy_bert_cor_sen_lem': 0.7645, 'Value_Accuracy_bert_cor_sen_lem': 0.631, 'Joint_Accuracy_bert_cor_sen_lem': 0.2247, 'Recall_at_1_bert_cor_sen_lem': 0.1934, 'Recall_at_5_bert_cor_sen_lem': 0.4509, 'Recall_at_10_bert_cor_sen_lem': 0.5584, 'Turn_Accuracy_bert_cor_sen_lem': 0.2084, 'Cascading_Score_bert_cor_sen_lem': 0.0863},
{'Intent_Accuracy_bert_cor_sen_lem': 0.9244, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.8279, 'Action_Accuracy_bert_cor_sen_lem': 0.7679, 'Value_Accuracy_bert_cor_sen_lem': 0.6387, 'Joint_Accuracy_bert_cor_sen_lem': 0.228, 'Recall_at_1_bert_cor_sen_lem': 0.2054, 'Recall_at_5_bert_cor_sen_lem': 0.4719, 'Recall_at_10_bert_cor_sen_lem': 0.584, 'Turn_Accuracy_bert_cor_sen_lem': 0.2141, 'Cascading_Score_bert_cor_sen_lem': 0.0865},
{'Intent_Accuracy_bert_cor_sen_lem': 0.9248, 'Nextstep_Accuracy_bert_cor_sen_lem': 0.8277, 'Action_Accuracy_bert_cor_sen_lem': 0.7665, 'Value_Accuracy_bert_cor_sen_lem': 0.6426, 'Joint_Accuracy_bert_cor_sen_lem': 0.2292, 'Recall_at_1_bert_cor_sen_lem': 0.2081, 'Recall_at_5_bert_cor_sen_lem': 0.4805, 'Recall_at_10_bert_cor_sen_lem': 0.5908, 'Turn_Accuracy_bert_cor_sen_lem': 0.2185, 'Cascading_Score_bert_cor_sen_lem': 0.0898}
    ] 

bert_cor = [ # abcde2
{'Intent_Accuracy_bert_cor': 0.5317, 'Nextstep_Accuracy_bert_cor': 0.7979, 'Action_Accuracy_bert_cor': 0.6508, 'Value_Accuracy_bert_cor': 0.2374, 'Joint_Accuracy_bert_cor': 0.1626, 'Recall_at_1_bert_cor': 0.0849, 'Recall_at_5_bert_cor': 0.3057, 'Recall_at_10_bert_cor': 0.4307, 'Turn_Accuracy_bert_cor': 0.0721, 'Cascading_Score_bert_cor': 0.039},
{'Intent_Accuracy_bert_cor': 0.6893, 'Nextstep_Accuracy_bert_cor': 0.7962, 'Action_Accuracy_bert_cor': 0.7084, 'Value_Accuracy_bert_cor': 0.3145, 'Joint_Accuracy_bert_cor': 0.2168, 'Recall_at_1_bert_cor': 0.1211, 'Recall_at_5_bert_cor': 0.3632, 'Recall_at_10_bert_cor': 0.4829, 'Turn_Accuracy_bert_cor': 0.1255, 'Cascading_Score_bert_cor': 0.0578},
{'Intent_Accuracy_bert_cor': 0.8318, 'Nextstep_Accuracy_bert_cor': 0.8098, 'Action_Accuracy_bert_cor': 0.7286, 'Value_Accuracy_bert_cor': 0.349, 'Joint_Accuracy_bert_cor': 0.2404, 'Recall_at_1_bert_cor': 0.1467, 'Recall_at_5_bert_cor': 0.4051, 'Recall_at_10_bert_cor': 0.5258, 'Turn_Accuracy_bert_cor': 0.1708, 'Cascading_Score_bert_cor': 0.0721},
{'Intent_Accuracy_bert_cor': 0.8543, 'Nextstep_Accuracy_bert_cor': 0.8154, 'Action_Accuracy_bert_cor': 0.7331, 'Value_Accuracy_bert_cor': 0.3902, 'Joint_Accuracy_bert_cor': 0.2689, 'Recall_at_1_bert_cor': 0.1698, 'Recall_at_5_bert_cor': 0.4327, 'Recall_at_10_bert_cor': 0.5514, 'Turn_Accuracy_bert_cor': 0.1978, 'Cascading_Score_bert_cor': 0.0803},
{'Intent_Accuracy_bert_cor': 0.8726, 'Nextstep_Accuracy_bert_cor': 0.8193, 'Action_Accuracy_bert_cor': 0.7419, 'Value_Accuracy_bert_cor': 0.4053, 'Joint_Accuracy_bert_cor': 0.2812, 'Recall_at_1_bert_cor': 0.1807, 'Recall_at_5_bert_cor': 0.4486, 'Recall_at_10_bert_cor': 0.5643, 'Turn_Accuracy_bert_cor': 0.2118, 'Cascading_Score_bert_cor': 0.0846},
{'Intent_Accuracy_bert_cor': 0.8985, 'Nextstep_Accuracy_bert_cor': 0.8071, 'Action_Accuracy_bert_cor': 0.7414, 'Value_Accuracy_bert_cor': 0.4159, 'Joint_Accuracy_bert_cor': 0.2865, 'Recall_at_1_bert_cor': 0.149, 'Recall_at_5_bert_cor': 0.4009, 'Recall_at_10_bert_cor': 0.5184, 'Turn_Accuracy_bert_cor': 0.1995, 'Cascading_Score_bert_cor': 0.0782},
{'Intent_Accuracy_bert_cor': 0.9227, 'Nextstep_Accuracy_bert_cor': 0.7865, 'Action_Accuracy_bert_cor': 0.7586, 'Value_Accuracy_bert_cor': 0.4376, 'Joint_Accuracy_bert_cor': 0.3038, 'Recall_at_1_bert_cor': 0.1695, 'Recall_at_5_bert_cor': 0.4214, 'Recall_at_10_bert_cor': 0.5337, 'Turn_Accuracy_bert_cor': 0.2191, 'Cascading_Score_bert_cor': 0.0867},
{'Intent_Accuracy_bert_cor': 0.9352, 'Nextstep_Accuracy_bert_cor': 0.825, 'Action_Accuracy_bert_cor': 0.7677, 'Value_Accuracy_bert_cor': 0.4633, 'Joint_Accuracy_bert_cor': 0.3219, 'Recall_at_1_bert_cor': 0.1928, 'Recall_at_5_bert_cor': 0.4606, 'Recall_at_10_bert_cor': 0.5791, 'Turn_Accuracy_bert_cor': 0.2416, 'Cascading_Score_bert_cor': 0.0932},
{'Intent_Accuracy_bert_cor': 0.9354, 'Nextstep_Accuracy_bert_cor': 0.8277, 'Action_Accuracy_bert_cor': 0.7779, 'Value_Accuracy_bert_cor': 0.4722, 'Joint_Accuracy_bert_cor': 0.328, 'Recall_at_1_bert_cor': 0.2097, 'Recall_at_5_bert_cor': 0.4803, 'Recall_at_10_bert_cor': 0.5952, 'Turn_Accuracy_bert_cor': 0.2504, 'Cascading_Score_bert_cor': 0.0969},
{'Intent_Accuracy_bert_cor': 0.936, 'Nextstep_Accuracy_bert_cor': 0.8323, 'Action_Accuracy_bert_cor': 0.7775, 'Value_Accuracy_bert_cor': 0.4761, 'Joint_Accuracy_bert_cor': 0.3297, 'Recall_at_1_bert_cor': 0.219, 'Recall_at_5_bert_cor': 0.4964, 'Recall_at_10_bert_cor': 0.6103, 'Turn_Accuracy_bert_cor': 0.2559, 'Cascading_Score_bert_cor': 0.0987}
    ] 


bert_sen = [ # abcde1
{'Intent_Accuracy_bert_sen': 0.4707, 'Nextstep_Accuracy_bert_sen': 0.7768, 'Action_Accuracy_bert_sen': 0.5807, 'Value_Accuracy_bert_sen': 0.3181, 'Joint_Accuracy_bert_sen': 0.1654, 'Recall_at_1_bert_sen': 0.0946, 'Recall_at_5_bert_sen': 0.3165, 'Recall_at_10_bert_sen': 0.4353, 'Turn_Accuracy_bert_sen': 0.0659, 'Cascading_Score_bert_sen': 0.0325},
{'Intent_Accuracy_bert_sen': 0.696, 'Nextstep_Accuracy_bert_sen': 0.7911, 'Action_Accuracy_bert_sen': 0.6758, 'Value_Accuracy_bert_sen': 0.3938, 'Joint_Accuracy_bert_sen': 0.2125, 'Recall_at_1_bert_sen': 0.1264, 'Recall_at_5_bert_sen': 0.3771, 'Recall_at_10_bert_sen': 0.4976, 'Turn_Accuracy_bert_sen': 0.1257, 'Cascading_Score_bert_sen': 0.058},
{'Intent_Accuracy_bert_sen': 0.861, 'Nextstep_Accuracy_bert_sen': 0.8059, 'Action_Accuracy_bert_sen': 0.7143, 'Value_Accuracy_bert_sen': 0.4238, 'Joint_Accuracy_bert_sen': 0.2284, 'Recall_at_1_bert_sen': 0.1516, 'Recall_at_5_bert_sen': 0.4142, 'Recall_at_10_bert_sen': 0.5376, 'Turn_Accuracy_bert_sen': 0.1751, 'Cascading_Score_bert_sen': 0.0719},
{'Intent_Accuracy_bert_sen': 0.897, 'Nextstep_Accuracy_bert_sen': 0.8172, 'Action_Accuracy_bert_sen': 0.7192, 'Value_Accuracy_bert_sen': 0.4488, 'Joint_Accuracy_bert_sen': 0.2431, 'Recall_at_1_bert_sen': 0.1699, 'Recall_at_5_bert_sen': 0.4359, 'Recall_at_10_bert_sen': 0.5514, 'Turn_Accuracy_bert_sen': 0.197, 'Cascading_Score_bert_sen': 0.0792},
{'Intent_Accuracy_bert_sen': 0.9013, 'Nextstep_Accuracy_bert_sen': 0.8207, 'Action_Accuracy_bert_sen': 0.728, 'Value_Accuracy_bert_sen': 0.4593, 'Joint_Accuracy_bert_sen': 0.2506, 'Recall_at_1_bert_sen': 0.1758, 'Recall_at_5_bert_sen': 0.4508, 'Recall_at_10_bert_sen': 0.5616, 'Turn_Accuracy_bert_sen': 0.2035, 'Cascading_Score_bert_sen': 0.0822},
{'Intent_Accuracy_bert_sen': 0.8961, 'Nextstep_Accuracy_bert_sen': 0.7971, 'Action_Accuracy_bert_sen': 0.7194, 'Value_Accuracy_bert_sen': 0.4745, 'Joint_Accuracy_bert_sen': 0.2565, 'Recall_at_1_bert_sen': 0.1481, 'Recall_at_5_bert_sen': 0.3926, 'Recall_at_10_bert_sen': 0.5111, 'Turn_Accuracy_bert_sen': 0.1886, 'Cascading_Score_bert_sen': 0.0786},
{'Intent_Accuracy_bert_sen': 0.9141, 'Nextstep_Accuracy_bert_sen': 0.8147, 'Action_Accuracy_bert_sen': 0.7378, 'Value_Accuracy_bert_sen': 0.5009, 'Joint_Accuracy_bert_sen': 0.2732, 'Recall_at_1_bert_sen': 0.1699, 'Recall_at_5_bert_sen': 0.4261, 'Recall_at_10_bert_sen': 0.5449, 'Turn_Accuracy_bert_sen': 0.2086, 'Cascading_Score_bert_sen': 0.0853},
{'Intent_Accuracy_bert_sen': 0.9314, 'Nextstep_Accuracy_bert_sen': 0.8239, 'Action_Accuracy_bert_sen': 0.7437, 'Value_Accuracy_bert_sen': 0.5407, 'Joint_Accuracy_bert_sen': 0.2922, 'Recall_at_1_bert_sen': 0.1874, 'Recall_at_5_bert_sen': 0.4516, 'Recall_at_10_bert_sen': 0.563, 'Turn_Accuracy_bert_sen': 0.2264, 'Cascading_Score_bert_sen': 0.0879},
{'Intent_Accuracy_bert_sen': 0.9351, 'Nextstep_Accuracy_bert_sen': 0.8285, 'Action_Accuracy_bert_sen': 0.7579, 'Value_Accuracy_bert_sen': 0.5476, 'Joint_Accuracy_bert_sen': 0.2979, 'Recall_at_1_bert_sen': 0.2009, 'Recall_at_5_bert_sen': 0.4717, 'Recall_at_10_bert_sen': 0.5864, 'Turn_Accuracy_bert_sen': 0.2378, 'Cascading_Score_bert_sen': 0.0937},
{'Intent_Accuracy_bert_sen': 0.938, 'Nextstep_Accuracy_bert_sen': 0.8319, 'Action_Accuracy_bert_sen': 0.7634, 'Value_Accuracy_bert_sen': 0.5613, 'Joint_Accuracy_bert_sen': 0.3042, 'Recall_at_1_bert_sen': 0.2068, 'Recall_at_5_bert_sen': 0.4823, 'Recall_at_10_bert_sen': 0.5956, 'Turn_Accuracy_bert_sen': 0.2437, 'Cascading_Score_bert_sen': 0.0966}
    ] 


bert_sen_lem = [ # abcde3
{'Intent_Accuracy_bert_sen_lem': 0.509, 'Nextstep_Accuracy_bert_sen_lem': 0.7668, 'Action_Accuracy_bert_sen_lem': 0.5947, 'Value_Accuracy_bert_sen_lem': 0.5356, 'Joint_Accuracy_bert_sen_lem': 0.1838, 'Recall_at_1_bert_sen_lem': 0.0881, 'Recall_at_5_bert_sen_lem': 0.3026, 'Recall_at_10_bert_sen_lem': 0.4267, 'Turn_Accuracy_bert_sen_lem': 0.0645, 'Cascading_Score_bert_sen_lem': 0.0302},
{'Intent_Accuracy_bert_sen_lem': 0.7407, 'Nextstep_Accuracy_bert_sen_lem': 0.7896, 'Action_Accuracy_bert_sen_lem': 0.6927, 'Value_Accuracy_bert_sen_lem': 0.6007, 'Joint_Accuracy_bert_sen_lem': 0.2113, 'Recall_at_1_bert_sen_lem': 0.1198, 'Recall_at_5_bert_sen_lem': 0.3695, 'Recall_at_10_bert_sen_lem': 0.4851, 'Turn_Accuracy_bert_sen_lem': 0.1226, 'Cascading_Score_bert_sen_lem': 0.0516},
{'Intent_Accuracy_bert_sen_lem': 0.8755, 'Nextstep_Accuracy_bert_sen_lem': 0.8054, 'Action_Accuracy_bert_sen_lem': 0.7158, 'Value_Accuracy_bert_sen_lem': 0.6089, 'Joint_Accuracy_bert_sen_lem': 0.216, 'Recall_at_1_bert_sen_lem': 0.1469, 'Recall_at_5_bert_sen_lem': 0.4064, 'Recall_at_10_bert_sen_lem': 0.5274, 'Turn_Accuracy_bert_sen_lem': 0.1668, 'Cascading_Score_bert_sen_lem': 0.0672},
{'Intent_Accuracy_bert_sen_lem': 0.9114, 'Nextstep_Accuracy_bert_sen_lem': 0.8133, 'Action_Accuracy_bert_sen_lem': 0.7331, 'Value_Accuracy_bert_sen_lem': 0.6343, 'Joint_Accuracy_bert_sen_lem': 0.2251, 'Recall_at_1_bert_sen_lem': 0.1632, 'Recall_at_5_bert_sen_lem': 0.4251, 'Recall_at_10_bert_sen_lem': 0.5463, 'Turn_Accuracy_bert_sen_lem': 0.1842, 'Cascading_Score_bert_sen_lem': 0.0718},
{'Intent_Accuracy_bert_sen_lem': 0.9178, 'Nextstep_Accuracy_bert_sen_lem': 0.814, 'Action_Accuracy_bert_sen_lem': 0.74, 'Value_Accuracy_bert_sen_lem': 0.6437, 'Joint_Accuracy_bert_sen_lem': 0.2286, 'Recall_at_1_bert_sen_lem': 0.1761, 'Recall_at_5_bert_sen_lem': 0.4382, 'Recall_at_10_bert_sen_lem': 0.5559, 'Turn_Accuracy_bert_sen_lem': 0.1926, 'Cascading_Score_bert_sen_lem': 0.074},
{'Intent_Accuracy_bert_sen_lem': 0.9114, 'Nextstep_Accuracy_bert_sen_lem': 0.8078, 'Action_Accuracy_bert_sen_lem': 0.7325, 'Value_Accuracy_bert_sen_lem': 0.6288, 'Joint_Accuracy_bert_sen_lem': 0.2241, 'Recall_at_1_bert_sen_lem': 0.1449, 'Recall_at_5_bert_sen_lem': 0.3941, 'Recall_at_10_bert_sen_lem': 0.517, 'Turn_Accuracy_bert_sen_lem': 0.1756, 'Cascading_Score_bert_sen_lem': 0.0718},
{'Intent_Accuracy_bert_sen_lem': 0.9075, 'Nextstep_Accuracy_bert_sen_lem': 0.8132, 'Action_Accuracy_bert_sen_lem': 0.7457, 'Value_Accuracy_bert_sen_lem': 0.6486, 'Joint_Accuracy_bert_sen_lem': 0.2317, 'Recall_at_1_bert_sen_lem': 0.1671, 'Recall_at_5_bert_sen_lem': 0.425, 'Recall_at_10_bert_sen_lem': 0.5357, 'Turn_Accuracy_bert_sen_lem': 0.1917, 'Cascading_Score_bert_sen_lem': 0.076},
{'Intent_Accuracy_bert_sen_lem': 0.9305, 'Nextstep_Accuracy_bert_sen_lem': 0.815, 'Action_Accuracy_bert_sen_lem': 0.7474, 'Value_Accuracy_bert_sen_lem': 0.6448, 'Joint_Accuracy_bert_sen_lem': 0.2308, 'Recall_at_1_bert_sen_lem': 0.1848, 'Recall_at_5_bert_sen_lem': 0.4481, 'Recall_at_10_bert_sen_lem': 0.5548, 'Turn_Accuracy_bert_sen_lem': 0.2057, 'Cascading_Score_bert_sen_lem': 0.0827},
{'Intent_Accuracy_bert_sen_lem': 0.9314, 'Nextstep_Accuracy_bert_sen_lem': 0.8228, 'Action_Accuracy_bert_sen_lem': 0.7573, 'Value_Accuracy_bert_sen_lem': 0.6558, 'Joint_Accuracy_bert_sen_lem': 0.2339, 'Recall_at_1_bert_sen_lem': 0.2009, 'Recall_at_5_bert_sen_lem': 0.4684, 'Recall_at_10_bert_sen_lem': 0.5807, 'Turn_Accuracy_bert_sen_lem': 0.2172, 'Cascading_Score_bert_sen_lem': 0.086},
{'Intent_Accuracy_bert_sen_lem': 0.9311, 'Nextstep_Accuracy_bert_sen_lem': 0.8264, 'Action_Accuracy_bert_sen_lem': 0.7639, 'Value_Accuracy_bert_sen_lem': 0.6657, 'Joint_Accuracy_bert_sen_lem': 0.238, 'Recall_at_1_bert_sen_lem': 0.2085, 'Recall_at_5_bert_sen_lem': 0.4739, 'Recall_at_10_bert_sen_lem': 0.5839, 'Turn_Accuracy_bert_sen_lem': 0.2226, 'Cascading_Score_bert_sen_lem': 0.0881}
    ] 


bert_cor_lem = [ # abcde2
{'Intent_Accuracy_bert_cor_lem': 0.4977, 'Nextstep_Accuracy_bert_cor_lem': 0.7856, 'Action_Accuracy_bert_cor_lem': 0.6296, 'Value_Accuracy_bert_cor_lem': 0.5604, 'Joint_Accuracy_bert_cor_lem': 0.195, 'Recall_at_1_bert_cor_lem': 0.0908, 'Recall_at_5_bert_cor_lem': 0.3135, 'Recall_at_10_bert_cor_lem': 0.4363, 'Turn_Accuracy_bert_cor_lem': 0.0716, 'Cascading_Score_bert_cor_lem': 0.0365},
{'Intent_Accuracy_bert_cor_lem': 0.7364, 'Nextstep_Accuracy_bert_cor_lem': 0.7904, 'Action_Accuracy_bert_cor_lem': 0.6921, 'Value_Accuracy_bert_cor_lem': 0.6122, 'Joint_Accuracy_bert_cor_lem': 0.219, 'Recall_at_1_bert_cor_lem': 0.1225, 'Recall_at_5_bert_cor_lem': 0.3628, 'Recall_at_10_bert_cor_lem': 0.4828, 'Turn_Accuracy_bert_cor_lem': 0.1273, 'Cascading_Score_bert_cor_lem': 0.0565},
{'Intent_Accuracy_bert_cor_lem': 0.8171, 'Nextstep_Accuracy_bert_cor_lem': 0.8119, 'Action_Accuracy_bert_cor_lem': 0.7103, 'Value_Accuracy_bert_cor_lem': 0.6277, 'Joint_Accuracy_bert_cor_lem': 0.2219, 'Recall_at_1_bert_cor_lem': 0.1461, 'Recall_at_5_bert_cor_lem': 0.3992, 'Recall_at_10_bert_cor_lem': 0.5145, 'Turn_Accuracy_bert_cor_lem': 0.1588, 'Cascading_Score_bert_cor_lem': 0.0665},
{'Intent_Accuracy_bert_cor_lem': 0.8452, 'Nextstep_Accuracy_bert_cor_lem': 0.816, 'Action_Accuracy_bert_cor_lem': 0.7327, 'Value_Accuracy_bert_cor_lem': 0.6641, 'Joint_Accuracy_bert_cor_lem': 0.2357, 'Recall_at_1_bert_cor_lem': 0.1661, 'Recall_at_5_bert_cor_lem': 0.4259, 'Recall_at_10_bert_cor_lem': 0.541, 'Turn_Accuracy_bert_cor_lem': 0.182, 'Cascading_Score_bert_cor_lem': 0.0747},
{'Intent_Accuracy_bert_cor_lem': 0.8573, 'Nextstep_Accuracy_bert_cor_lem': 0.8202, 'Action_Accuracy_bert_cor_lem': 0.7419, 'Value_Accuracy_bert_cor_lem': 0.6702, 'Joint_Accuracy_bert_cor_lem': 0.2372, 'Recall_at_1_bert_cor_lem': 0.1792, 'Recall_at_5_bert_cor_lem': 0.4469, 'Recall_at_10_bert_cor_lem': 0.5598, 'Turn_Accuracy_bert_cor_lem': 0.1931, 'Cascading_Score_bert_cor_lem': 0.0779},
{'Intent_Accuracy_bert_cor_lem': 0.8929, 'Nextstep_Accuracy_bert_cor_lem': 0.8101, 'Action_Accuracy_bert_cor_lem': 0.7311, 'Value_Accuracy_bert_cor_lem': 0.674, 'Joint_Accuracy_bert_cor_lem': 0.237, 'Recall_at_1_bert_cor_lem': 0.1393, 'Recall_at_5_bert_cor_lem': 0.3831, 'Recall_at_10_bert_cor_lem': 0.5059, 'Turn_Accuracy_bert_cor_lem': 0.1728, 'Cascading_Score_bert_cor_lem': 0.073},
{'Intent_Accuracy_bert_cor_lem': 0.9273, 'Nextstep_Accuracy_bert_cor_lem': 0.816, 'Action_Accuracy_bert_cor_lem': 0.7567, 'Value_Accuracy_bert_cor_lem': 0.6757, 'Joint_Accuracy_bert_cor_lem': 0.2404, 'Recall_at_1_bert_cor_lem': 0.1716, 'Recall_at_5_bert_cor_lem': 0.4126, 'Recall_at_10_bert_cor_lem': 0.5347, 'Turn_Accuracy_bert_cor_lem': 0.1934, 'Cascading_Score_bert_cor_lem': 0.0811},
{'Intent_Accuracy_bert_cor_lem': 0.9355, 'Nextstep_Accuracy_bert_cor_lem': 0.8277, 'Action_Accuracy_bert_cor_lem': 0.7679, 'Value_Accuracy_bert_cor_lem': 0.6862, 'Joint_Accuracy_bert_cor_lem': 0.2447, 'Recall_at_1_bert_cor_lem': 0.1875, 'Recall_at_5_bert_cor_lem': 0.4496, 'Recall_at_10_bert_cor_lem': 0.5701, 'Turn_Accuracy_bert_cor_lem': 0.2116, 'Cascading_Score_bert_cor_lem': 0.0856},
{'Intent_Accuracy_bert_cor_lem': 0.9411, 'Nextstep_Accuracy_bert_cor_lem': 0.8317, 'Action_Accuracy_bert_cor_lem': 0.7704, 'Value_Accuracy_bert_cor_lem': 0.6928, 'Joint_Accuracy_bert_cor_lem': 0.2463, 'Recall_at_1_bert_cor_lem': 0.2121, 'Recall_at_5_bert_cor_lem': 0.4896, 'Recall_at_10_bert_cor_lem': 0.6044, 'Turn_Accuracy_bert_cor_lem': 0.2265, 'Cascading_Score_bert_cor_lem': 0.0903},
{'Intent_Accuracy_bert_cor_lem': 0.9426, 'Nextstep_Accuracy_bert_cor_lem': 0.8348, 'Action_Accuracy_bert_cor_lem': 0.7732, 'Value_Accuracy_bert_cor_lem': 0.6939, 'Joint_Accuracy_bert_cor_lem': 0.2465, 'Recall_at_1_bert_cor_lem': 0.2225, 'Recall_at_5_bert_cor_lem': 0.4996, 'Recall_at_10_bert_cor_lem': 0.6098, 'Turn_Accuracy_bert_cor_lem': 0.2344, 'Cascading_Score_bert_cor_lem': 0.0924}
    ] 


bert_lem = [ # abcde1
{'Intent_Accuracy_bert_lem': 0.4738, 'Nextstep_Accuracy_bert_lem': 0.7861, 'Action_Accuracy_bert_lem': 0.6198, 'Value_Accuracy_bert_lem': 0.5521, 'Joint_Accuracy_bert_lem': 0.1956, 'Recall_at_1_bert_lem': 0.1, 'Recall_at_5_bert_lem': 0.3225, 'Recall_at_10_bert_lem': 0.4449, 'Turn_Accuracy_bert_lem': 0.0725, 'Cascading_Score_bert_lem': 0.0359},
{'Intent_Accuracy_bert_lem': 0.6625, 'Nextstep_Accuracy_bert_lem': 0.8037, 'Action_Accuracy_bert_lem': 0.6781, 'Value_Accuracy_bert_lem': 0.6122, 'Joint_Accuracy_bert_lem': 0.2166, 'Recall_at_1_bert_lem': 0.1304, 'Recall_at_5_bert_lem': 0.3734, 'Recall_at_10_bert_lem': 0.4875, 'Turn_Accuracy_bert_lem': 0.1198, 'Cascading_Score_bert_lem': 0.0551},
{'Intent_Accuracy_bert_lem': 0.8042, 'Nextstep_Accuracy_bert_lem': 0.8132, 'Action_Accuracy_bert_lem': 0.7366, 'Value_Accuracy_bert_lem': 0.6365, 'Joint_Accuracy_bert_lem': 0.2266, 'Recall_at_1_bert_lem': 0.1554, 'Recall_at_5_bert_lem': 0.4104, 'Recall_at_10_bert_lem': 0.53, 'Turn_Accuracy_bert_lem': 0.1636, 'Cascading_Score_bert_lem': 0.0713},
{'Intent_Accuracy_bert_lem': 0.8629, 'Nextstep_Accuracy_bert_lem': 0.8198, 'Action_Accuracy_bert_lem': 0.7465, 'Value_Accuracy_bert_lem': 0.6696, 'Joint_Accuracy_bert_lem': 0.238, 'Recall_at_1_bert_lem': 0.1722, 'Recall_at_5_bert_lem': 0.4351, 'Recall_at_10_bert_lem': 0.5532, 'Turn_Accuracy_bert_lem': 0.1888, 'Cascading_Score_bert_lem': 0.078},
{'Intent_Accuracy_bert_lem': 0.8733, 'Nextstep_Accuracy_bert_lem': 0.8241, 'Action_Accuracy_bert_lem': 0.7565, 'Value_Accuracy_bert_lem': 0.6751, 'Joint_Accuracy_bert_lem': 0.24, 'Recall_at_1_bert_lem': 0.1877, 'Recall_at_5_bert_lem': 0.4546, 'Recall_at_10_bert_lem': 0.5667, 'Turn_Accuracy_bert_lem': 0.2006, 'Cascading_Score_bert_lem': 0.0826},
{'Intent_Accuracy_bert_lem': 0.8643, 'Nextstep_Accuracy_bert_lem': 0.8023, 'Action_Accuracy_bert_lem': 0.7347, 'Value_Accuracy_bert_lem': 0.6448, 'Joint_Accuracy_bert_lem': 0.2294, 'Recall_at_1_bert_lem': 0.1553, 'Recall_at_5_bert_lem': 0.4046, 'Recall_at_10_bert_lem': 0.5178, 'Turn_Accuracy_bert_lem': 0.1764, 'Cascading_Score_bert_lem': 0.0745},
{'Intent_Accuracy_bert_lem': 0.9021, 'Nextstep_Accuracy_bert_lem': 0.8281, 'Action_Accuracy_bert_lem': 0.7649, 'Value_Accuracy_bert_lem': 0.6823, 'Joint_Accuracy_bert_lem': 0.2435, 'Recall_at_1_bert_lem': 0.1751, 'Recall_at_5_bert_lem': 0.4267, 'Recall_at_10_bert_lem': 0.5409, 'Turn_Accuracy_bert_lem': 0.1989, 'Cascading_Score_bert_lem': 0.0843},
{'Intent_Accuracy_bert_lem': 0.9249, 'Nextstep_Accuracy_bert_lem': 0.8282, 'Action_Accuracy_bert_lem': 0.774, 'Value_Accuracy_bert_lem': 0.6851, 'Joint_Accuracy_bert_lem': 0.2453, 'Recall_at_1_bert_lem': 0.1965, 'Recall_at_5_bert_lem': 0.4631, 'Recall_at_10_bert_lem': 0.5778, 'Turn_Accuracy_bert_lem': 0.2174, 'Cascading_Score_bert_lem': 0.0894},
{'Intent_Accuracy_bert_lem': 0.9354, 'Nextstep_Accuracy_bert_lem': 0.8352, 'Action_Accuracy_bert_lem': 0.7783, 'Value_Accuracy_bert_lem': 0.69, 'Joint_Accuracy_bert_lem': 0.2461, 'Recall_at_1_bert_lem': 0.2174, 'Recall_at_5_bert_lem': 0.4895, 'Recall_at_10_bert_lem': 0.6033, 'Turn_Accuracy_bert_lem': 0.2325, 'Cascading_Score_bert_lem': 0.0953},
{'Intent_Accuracy_bert_lem': 0.9359, 'Nextstep_Accuracy_bert_lem': 0.8339, 'Action_Accuracy_bert_lem': 0.7783, 'Value_Accuracy_bert_lem': 0.6862, 'Joint_Accuracy_bert_lem': 0.2453, 'Recall_at_1_bert_lem': 0.2311, 'Recall_at_5_bert_lem': 0.5017, 'Recall_at_10_bert_lem': 0.6127, 'Turn_Accuracy_bert_lem': 0.2409, 'Cascading_Score_bert_lem': 0.0988}
    ] 



def get_metric(metric, data):
    y = [next(v for k,v in d.items() if metric in k) for d in data]
    x = list(np.arange(0., len(y)))
    return x, y
    

def plot_step_accuracy(experiment):
    m = metrics[0:5]
    for i in range(0,len(m)):
        x, y = get_metric(m[i], list(experiment.values())[0])
        plt.plot(x, y)
    plt.title('stepwise accuracy | ' + list(experiment)[0])
    plt.xlabel('epochs')
    plt.ylabel('performance')
    plt.legend(m, loc='upper left', bbox_to_anchor=(1.0, 0.75))
    plt.show()

def plot_cascading_accuracy(experiment):
    m = metrics[5:10]
    for i in range(0,len(m)):
        x, y = get_metric(m[i], list(experiment.values())[0])
        plt.plot(x, y)
    plt.title('cascading accuracy | ' + list(experiment)[0])
    plt.xlabel('epochs')
    plt.ylabel('performance')
    plt.legend(m, loc='upper left', bbox_to_anchor=(1.0, 0.75))
    plt.show()

def plot_acc(experiment):
    m1 = metrics[0:8]
    m2 = metrics[8:10]
    fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 9))
    fig.suptitle('groundtruth vs. ' + str(list(experiment)[0]))
    fig.tight_layout(pad=3.0)
    
    for i in range(0,len(m1)):
        x, y = get_metric(m1[i], list(experiment.values())[0])
        ax1.plot(x, y)
    
    for i in range(0,len(m2)):
        x, y = get_metric(m2[i], list(experiment.values())[0])
        ax2.plot(x, y)

    
    ax1.set_title('stepwise accuracy')
    ax1.set(xlabel='epochs', ylabel='performance')
    ax1.legend(m1, loc='upper left', bbox_to_anchor=(1.0, 0.75))
    ax2.set_title('cascading accuracy')
    ax2.set(xlabel='epochs', ylabel='performance')
    ax2.legend(m2, loc='upper left', bbox_to_anchor=(1.0, 0.75))
    plt.show()


def plot_accuracy(experiment1, experiment2):
    m1 = metrics[0:8]
    m2 = metrics[8:10]
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 9))
    #fig.suptitle('groundtruth vs. ' + str(list(experiment2)[0]))
    fig.tight_layout(pad=3.0)
    
    for i in range(0,len(m1)):
        hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])][0]

        x, y = get_metric(m1[i], list(experiment1.values())[0])
        ax1.plot(x, y, 'o', color=hexadecimal)
        x, y = get_metric(m1[i], list(experiment2.values())[0])
        ax1.plot(x, y, color=hexadecimal)
    
    for i in range(0,len(m2)):
        hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])][0]

        x, y = get_metric(m2[i], list(experiment1.values())[0])
        ax2.plot(x, y, 'o', color=hexadecimal)
        x, y = get_metric(m2[i], list(experiment2.values())[0])
        ax2.plot(x, y, color=hexadecimal)
    
    ax1.set_title('stepwise accuracy')
    ax1.set(xlabel='epochs', ylabel='performance')
    l1 = []
    for i in range(0, len(m1)):
        l1.append(m1[i])
        l1.append(m1[i])
    ax1.legend(l1, loc='upper left', bbox_to_anchor=(1.0, 1))
    ax2.set_title('cascading accuracy')
    ax2.set(xlabel='epochs', ylabel='performance')
    l2 = []
    for i in range(0, len(m2)):
        l2.append(m2[i])
        l2.append(m2[i])
    ax2.legend(l2, loc='upper left', bbox_to_anchor=(1.0, 0.75))
    plt.show()


def colorhex(red, green, blue):
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}"

def plot_metric(experiments, metric, pre=False):
    labels = []
    for i in range(0,len(experiments)):
        x, y = get_metric(metric, list(experiments[i].values())[0])
        label = list(experiments[i])[0]
        exp_label = label.replace('bert ', '').split(' ')
        
        EXP = (255,31,180) # red
        COR = (255,195,21) # yellow
        SEN = (31,255,173) # green
        LEM = (28,163,236) # blue
        # EXP = (238,64,53) # red
        # COR = (255,230,0) # yellow
        # SEN = (0,255,131) # green
        # LEM = (28,163,236) # blue
        
        r = 0
        g = 0
        b = 0
        if "EXP" in label:
            r += EXP[0]
            g += EXP[1]
            b += EXP[2]
        if "COR" in label:
            r += COR[0]
            g += COR[1]
            b += COR[2]
        if "SEN" in label:
            r += SEN[0]
            g += SEN[1]
            b += SEN[2]
        if "LEM" in label:
            r += LEM[0]
            g += LEM[1]
            b += LEM[2]
            
        r = r / len(exp_label)
        g = g / len(exp_label)
        b = b / len(exp_label)
        color = colorhex(r,g,b)
        
        if not pre:
            if i == 0: # if 'groundtruth' is in label # if i == 0
                plt.plot(x, y, 'o')
            else:
                plt.plot(x, y, color=color)
        else:
            plt.plot(x, y)
        labels.append(label)
    plt.rcParams['figure.figsize'] = [4, 4]
    plt.title(metric)
    plt.xlabel('epochs')
    plt.ylabel('performance')
    plt.legend(labels, loc='upper left', bbox_to_anchor=(1.0, 1.05))
    plt.show()




experiments = [
    {'bert groundtruth' : groundtruth_bert},
    {'roberta groundtruth' : groundtruth_roberta},
    {'albert groundtruth' : groundtruth_albert},
    {'bert EXP' : bert_exp},
    {'bert COR' : bert_cor},
    {'bert SEN' : bert_sen},
    {'bert LEM' : bert_lem},
    {'bert EXP COR' : bert_exp_cor},
    {'bert EXP SEN' : bert_exp_sen},
    {'bert EXP LEM' : bert_exp_lem},
    {'bert COR SEN' : bert_cor_sen},
    {'bert COR LEM' : bert_cor_lem},
    {'bert SEN LEM' : bert_sen_lem},
    {'bert EXP COR SEN' : bert_exp_cor_sen},
    {'bert EXP COR LEM' : bert_exp_cor_lem},
    {'bert EXP SEN LEM' : bert_exp_sen_lem},
    {'bert COR SEN LEM' : bert_cor_sen_lem},
    {'bert EXP COR SEN LEM' : bert_exp_cor_sen_lem},
    ]



def print_results_table():
    rows = []
    firstrow = ['experiment']
    for item in metrics:
        firstrow.append(item)
    
    rows.append(firstrow)
    
    for item in experiments:
        experiment = list(item.values())[0]
        row = list(experiment[len(experiment)-1].values())  
        row.insert(0, list(item)[0])
        rows.append(row)
    
    for i in range(1,len(metrics)+1):
        col = []
        for j in range(1,len(rows)):
            col.append(rows[j][i])
        col = list(np.array(col).astype(float))
        max_val = np.amax(col)
        max_indices = np.where(col == max_val)
        for k in max_indices:
            index = k[0] +1
            rows[index][i] = '***' + str(max_val)
    print(tabulate(rows, headers='firstrow', tablefmt='fancy_grid'))



ex = experiments.copy()
plot_metric([ex[0], ex[1], ex[2]], metrics[0], pre=True)
ex.remove(ex[1])
ex.remove(ex[1])
for i in range(0, 10):
    plot_metric(ex, metrics[i])
    
    
    
    
# plot_step_accuracy(experiments[0])
# plot_cascading_accuracy(experiments[0])
#plot_accuracy(experiments[0], experiments[3])
#plot_metric(experiments, metrics[0])
print_results_table()







