# Comparing 1, 3, 7, 8 since all single repo
# Then 2, 4, 9, 10 since all merge all

# # Data for Experiment 1
# data_exp1 = {
#     'Repository': [
#         'SciLean', 'FLT', 'PFR', 'PrimeNumberTheoremAnd', 'Compfiles', 'Debate', 
#         'Mathematics in Lean Source', 'Lean4Lean', 'Matrix Cookbook', 'Math Workshop', 
#         'LeanEuclid', 'Foundation', 'Con-nf', 'Saturn'
#     ],
#     'Validation R@10 Exp1': [71.27, 67.74, 59.55, 63.92, 59.62, 68.29, 67.69, 56.63, 69.7, 66.37, 78.24, 70.22, 59.59, 75.93],
#     'Average Test R@10 Exp1': [73.86, 69.18, 67.36, 66.29, 64.42, 66.26, 66.23, 65.33, 65.42, 65.69, 68.06, 68.26, 69, 69.13],
#     'task1 Test R@10: SciLean': [
#         73.85920584286112,
#         73.32868035876952,
#         72.97772414971524,
#         72.74959562478136,
#         70.63821600225762,
#         72.18909062705497,
#         71.68465119988151,
#         71.8550266804353,
#         72.47968578392056,
#         72.18145668925757,
#         72.24165404514586,
#         72.63323167669378,
#         72.21137620840445,
#         72.12031405939281,
#     ],
#     'task2 Test R@10: FLT': [
#         65.03756037379306,
#         65.4437900601135,
#         65.28681708194893,
#         64.28556082911982,
#         66.08491251665177,
#         66.83173447303471,
#         66.63359608842127,
#         67.56111646191407,
#         67.00525453696974,
#         67.6021387153508,
#         68.08308913175348,
#         70.12400489797922,
#         69.70057805078555,
#     ],
#     'task3 Test R@10: PFR': [
#         63.65575097301064,
#         63.83202408689207,
#         61.996316570687036,
#         63.73794117626312,
#         62.99436339195744,
#         63.558241815199715,
#         64.06524363935924,
#         64.44548116405595,
#         65.13062758577081,
#         65.25185096089368,
#         65.00655738804377,
#         64.14948786520812,
#     ],
#     'task4 Test R@10: PrimeNumberTheoremAnd': [
#         63.28736352066478,
#         63.878383496419964,
#         64.38922210110593,
#         64.42142632528835,
#         64.92999140161042,
#         64.9833680612955,
#         64.15851606432513,
#         64.61950014485748,
#         65.03301446024096,
#         64.76086730141155,
#         64.66836648850575,
#     ],
#     'task5 Test R@10: Compfiles': [
#         61.31705700621867,
#         61.984780260073194,
#         62.13446696670244,
#         61.8048483218664,
#         62.4541698020182,
#         61.373927241559436,
#         61.633013487753864,
#         61.964453731984804,
#         61.765413169572156,
#         61.1067724527919,
#     ],
#     'task6 Test R@10: Debate': [
#         69.17481903899771,
#         70.53592874311944,
#         70.1920320484163,
#         70.42574055395158,
#         71.42967644176399,
#         72.53351750701256,
#         72.37694021333849,
#         72.19307837625084,
#         71.89162921977173,
#     ],
#     'task7 Test R@10: Mathematics in Lean Source': [
#         64.9855925838272,
#         65.85078226948954,
#         67.40270833949648,
#         67.37677992091432,
#         67.75467208696138,
#         67.98786350323937,
#         68.36090423433816,
#         67.44245661956367,
#     ],
#     'task8 Test R@10: Lean4Lean': [
#         57.84334415584416,
#         50.66531385281385,
#         53.019480519480524,
#         56.99675324675325,
#         57.28084415584416,
#         60.62364718614719,
#         60.831980519480524,
#     ],
#     'task9 Test R@10: Matrix Cookbook': [
#         68.76854354153882,
#         68.50513676574997,
#         68.70271194563648,
#         68.93010378859435,
#         69.0352744303216,
#         68.44923943980548,
#     ],
#     'task10 Test R@10: Math Workshop': [
#         67.35894379864388,
#         68.54867887415723,
#         68.16078135860015,
#         67.45318190100068,
#         67.44333207732106,
#     ],
#     'task11 Test R@10: LeanEuclid': [
#         82.87842511306198,
#         84.63288108539506,
#         85.38707102952914,
#         85.19686086725194,
#     ],
#     'task12 Test R@10: Foundation': [
#         66.73914804016844,
#         67.25107304826693,
#         68.00645853579528,
#     ],
#     'task13 Test R@10: Con-nf': [
#         72.80222608248087,
#         70.88659571143648,
#     ],
#     'task14 Test R@10: Saturn': [
#         75.92063492063491,
#     ],
# }

# # Data for Experiment 3
# data_exp3 = {
#     'Repository': [
#         'Compfiles', 'Mathematics in Lean Source', 'PrimeNumberTheoremAnd', 'Math Workshop',
#         'FLT', 'PFR', 'SciLean', 'Debate', 'Matrix Cookbook', 'Con-nf',
#         'Foundation', 'Saturn', 'LeanEuclid', 'Lean4Lean'
#     ],
#     'Validation R@10 Exp3': [61.55, 67.99, 62.11, 64.44, 69.52, 63.13, 69.13, 70.35, 69.23, 58.69, 67.33, 77.12, 75.33, 74.85],
#     'Average Test R@10 Exp3': [60.65, 63.41, 62.55, 63.13, 64.13, 64.18, 65.39, 65.93, 66.85, 66.79, 66.67, 66.85, 68.39, 69.18],
#     'task1 Test R@10: Compfiles': [
#         60.64706363237122, 61.32526532789667, 61.6708414949816, 61.08772811083543, 60.261503729020575, 60.30689658891723, 60.60123583745497, 
#         60.153335213407054, 59.7709911492489, 60.00540189142794, 59.91546810620901,  59.88792078948343, 59.64851162308727, 59.420134063412014],
#     'task2 Test R@10: Mathematics in Lean Source': [
#         65.48591635672048, 65.75192142681436, 66.4970212474497, 66.49549432082041, 68.10014741519532, 67.93505866948891, 67.88206930569434, 
#         67.97385677238191, 67.61899831570183, 67.69791601581457, 67.581090432529, 67.75405538565178, 67.55591710838813],
#     'task3 Test R@10: PrimeNumberTheoremAnd': [
#         60.24074464461104, 61.601678946721826, 61.53315591144687, 62.06460041347961, 61.59168435062894, 62.286496517728665, 63.17427181726865, 
#         62.95367306725815, 62.84812634406436, 62.796309821659605, 62.573145480848204, 62.1576880065672],
#     'task4 Test R@10: Math Workshop': [
#         63.32418885891475, 65.32513361607977, 64.5102791379126, 67.1694072669325, 66.1893594713575, 67.5335852605448, 68.069787398159,
#         68.17676215211705, 67.79894096460394, 68.11922176655432, 68.37256090708857],
#     'task5 Test R@10: FLT': [
#         67.05795410771319, 67.50657961559557, 68.07826197730941, 68.94304048302321, 69.25704440208798, 69.06493687974346, 69.13220145373448, 
#         69.16048162503891, 69.47002065174642, 69.48301090016267],
#     'task6 Test R@10: PFR': [
#         62.58480312148852, 64.14710648979965, 64.12721536773596, 63.79502213074667, 63.97614939811086, 63.862150608153875, 63.66221697616451,
#         64.11137631469667, 63.98214095462508],
#     'task7 Test R@10: SciLean': [
#         68.18567421512381, 68.94310521480176, 70.20595528030704, 70.11859835977681, 70.07787376617783, 70.21904098724055, 69.74478926670498,
#         69.6667197138306],
#     'task8 Test R@10: Debate': [68.91867342976552, 68.8826460829178, 69.18231380479203, 69.31982755616717 , 69.58427923681404, 69.13118545804842, 69.48585868047269],
#     'task9 Test R@10: Matrix Cookbook': [71.04190539789398, 71.1094077126274, 71.08151370983799,  71.44692514637913, 71.19988426372417, 71.55403847771102],
#     'task10 Test R@10: Con-nf': [65.80145354021758, 66.0667469234885, 64.94783306581058, 66.0404405207776, 65.51765650080257 ],
#     'task11 Test R@10: Foundation': [65.21872588006309,  64.71692632303098 , 65.3519623141716, 65.13856379408705 ],
#     'task12 Test R@10: Saturn': [70.44871794871794, 73.2396449704142, 71.61242603550296 ],
#     'task13 Test R@10: LeanEuclid': [82.70537124802527, 83.11611374407583],
#     'task14 Test R@10: Lean4Lean': [81.45833333333331],
# }

# # Data for Experiment 7
# data_exp7 = {
#     'Repository': [
#         'SciLean', 'FLT', 'PFR', 'PrimeNumberTheoremAnd', 'Compfiles', 'Debate', 
#         'Mathematics in Lean Source', 'Lean4Lean', 'Matrix Cookbook', 'Math Workshop', 
#         'LeanEuclid', 'Foundation', 'Con-nf', 'Saturn'
#     ],
#     'Validation R@10 Exp7': [73.29, 64.81, 65.7, 62.29, 59.84, 66.27, 65.93, 69.54, 70.61, 69.37, 78.43, 65.66, 65.88, 76.62],
#     'Average Test R@10 Exp7': [71.2, 67.98, 65.76, 63.8, 62.69, 64.03, 64.96, 66.34, 67.34, 66.83, 68.53, 68.88, 68.93, 68.69],
#     'task1 Test R@10: SciLean': [
#         71.20389812006579,
#         70.88572734281318,
#         69.45071377207105,
#         68.04213122576395,
#         67.54972978525873,
#         69.00726445137622,
#         68.1475817004759,
#         69.16021492368797,
#         68.88467893457913,
#         68.66443399876533,
#         70.03684790111936,
#         70.79591178393574,
#         70.79269071285039,
#         70.4829950488633,
#     ],
#     'task2 Test R@10: FLT': [
#         65.07176102225607,
#         65.40441571629691,
#         65.46359663438871,
#         66.71763275228622,
#         66.32677938123483,
#         66.51353212244301,
#         67.46049110405545,
#         68.08911632921534,
#         67.86661907949036,
#         67.94782170524743,
#         68.34642420285985,
#         68.75055747332976,
#         67.64915090410139,
#     ],
#     'task3 Test R@10: PFR': [
#         62.42702304718558,
#         62.63653303342254,
#         61.08633206103329,
#         63.23399216111696,
#         63.389472189854345,
#         63.61926782962023,
#         63.87887090330584,
#         63.63956773198856,
#         63.98858679821081,
#         64.57938360670029,
#         64.55085193940373,
#         64.16258899533722,
#     ],
#     'task4 Test R@10: PrimeNumberTheoremAnd': [
#         59.05709262625734,
#         60.03949195908337,
#         60.71556174953353,
#         62.44838672264446,
#         62.986986620603055,
#         62.76881018292028,
#         62.97411652294068,
#         63.57632397421652,
#         63.45998124042688,
#         63.43669432056458,
#         63.36692156577178,
#     ],
#     'task5 Test R@10: Compfiles': [
#         58.03951954155118,
#         58.38592095720916,
#         58.366492642567124,
#         58.755716361631585,
#         58.53107733971926,
#         58.45073710566204,
#         58.660424661063374,
#         58.730219829023675,
#         58.29625734796436,
#         58.01394642544443,
#     ],
#     'task6 Test R@10: Debate': [
#         66.49117346834545,
#         67.58958046653301,
#         67.88538784536891,
#         69.36154187240723,
#         69.21244248624944,
#         70.29165080942265,
#         70.17823480171256,
#         70.45734791001368,
#         70.35635100396924,
#     ],
#     'task7 Test R@10: Mathematics in Lean Source': [
#         68.23328801265332,
#         68.61442088273124,
#         69.81774137323912,
#         70.04925969698739,
#         70.67932720667662,
#         70.89359973013498,
#         70.56087893914433,
#         70.63516112622426,
#     ],
#     'task8 Test R@10: Lean4Lean': [
#         72.24691961135538,
#         76.13472290072193,
#         72.3144981816208,
#         74.04874341855289,
#         75.21711990446725,
#         73.9551104597514,
#         73.78141453617759,
#     ],
#     'task9 Test R@10: Matrix Cookbook': [
#         68.60023836792804,
#         69.58270850427712,
#         69.10722555259355,
#         69.5082364668895,
#         70.28638374297653,
#         70.60975490571967,
#     ],
#     'task10 Test R@10: Math Workshop': [
#         65.55791237186585,
#         66.37793196098843,
#         66.27235794843769,
#         66.75759235227673,
#         66.29062797667449,
#     ],
#     'task11 Test R@10: LeanEuclid': [
#         79.13774749840323,
#         78.7604381047004,
#         80.15802048588935,
#         78.5824285004613,
#     ],
#     'task12 Test R@10: Foundation': [
#         69.86559891464196,
#         70.67254914144866,
#         70.08670954245116,
#     ],
#     'task13 Test R@10: Con-nf': [
#         67.35413639054848,
#         66.54201876670793,
#     ],
#     'task14 Test R@10: Saturn': [
#         71.1620644312952,
#     ],
# }

# # Data for Experiment 8
# data_exp8 = {
#     'Repository': [
#         'Compfiles', 'Mathematics in Lean Source', 'PrimeNumberTheoremAnd', 'Math Workshop',
#         'FLT', 'PFR', 'SciLean', 'Debate', 'Matrix Cookbook', 'Con-nf',
#         'Foundation', 'Saturn', 'LeanEuclid', 'Lean4Lean'
#     ],
#     'Validation R@10 Exp8': [62.79, 68.67, 61.33, 66.67, 65.92, 64.63, 68.99, 68.7, 72.74, 60.18, 64.22, 85.74, 81.55, 76.62],
#     'Average Test R@10 Exp8': [58.75, 63.04, 63.23, 63.54, 65.08, 64.35, 65.39, 66.12, 67.13, 67.23, 67.15, 68.99, 68.8, 69.46],
#     'task1 Test R@10: Compfiles': [
#         58.753674683076795, 59.266322324139544, 59.11501222901359, 58.65061069597731, 59.08351152440277, 58.254012190948366, 58.31714930746805, 
#         58.16705172694482, 58.2052292761354, 57.81466274231215, 57.931586927587, 58.19557776263805, 57.83569946245733, 58.135263114918466
#     ],
#     'task2 Test R@10: Mathematics in Lean Source': [
#         66.82241247037165, 68.22394378997019, 68.68426125028768, 68.46477136753248, 68.52621401540969, 69.0316174088683, 69.85560219203676,
#         71.13058457926404, 71.08407817561478, 70.98203735928827, 71.3442495324248, 70.89091325285804, 70.812148413589
#     ],
#     'task3 Test R@10: PrimeNumberTheoremAnd': [
#         62.37342972083976, 63.14389800117126, 63.57248915109329, 62.72338659851425, 63.358305722154476, 64.10691753468002, 63.84201634681258,
#         64.27687840450473, 64.27556330718753, 64.32555466833091, 64.09342765687724, 64.9165197040928
#     ],
#     'task4 Test R@10: Math Workshop': [
#         63.692126983903755, 64.88112322603578, 63.9572265456425, 64.24406215475595, 65.50233141990883, 65.34918568101253, 65.3123566360483,
#         65.3153365514187, 65.78415399457909, 64.77471389023222, 65.22434587068398
#     ],
#     'task5 Test R@10: FLT': [
#         69.4153115431529, 69.43774305709876, 70.20327098838466, 70.4426156383421, 71.33675063495751, 71.06544237274925, 70.90625206477227,
#         71.0986096866722, 71.13206954581167, 70.97284724070164
#     ],
#     'task6 Test R@10: PFR': [
#         63.22184229942324, 63.58756427174265, 63.68275102010593, 63.94294965538092, 63.77168689248713, 63.62856794668035, 63.94054979500917,
#         63.14677779628911, 62.79802453366899
#     ],
#     'task7 Test R@10: SciLean': [
#         68.99506046579262, 68.18453403141949, 68.60493009000173, 69.21270209449912, 68.94233134861383, 69.55466069440035, 69.0684458531042,
#         69.1317642961663
#     ],
#     'task8 Test R@10: Debate': [
#         68.90809008062388, 70.17778949140212, 69.6991246532138, 69.75589568301386, 69.9558657259099, 69.73137122548178, 69.76256041434732 
#     ],
#     'task9 Test R@10: Matrix Cookbook': [
#         71.60183467326324, 71.1446252160538, 70.97734804877662, 71.58040636612066, 71.22861530004387, 71.39100846243703
#     ],
#     'task10 Test R@10: Con-nf': [
#         68.94028103044498, 68.89075917252147, 69.10787470725997, 67.92715651834504, 68.53508001561279
#     ],
#     'task11 Test R@10: Foundation': [
#         67.02251895209642, 67.00373960937341, 66.70270105481373, 67.33863788793366
#     ],
#     'task12 Test R@10: Saturn': [85.92970521541949, 84.79591836734693, 86.49659863945578],
#     'task13 Test R@10: LeanEuclid': [73.07777777777778, 73.46666666666667],
#     'task14 Test R@10: Lean4Lean': [73.47417840375586],
# }

# experiments = {
#     'Exp1': data_exp1,
#     'Exp3': data_exp3,
#     'Exp7': data_exp7,
#     'Exp8': data_exp8
# }

data_exp2 = {
    'Repository': [
        'SciLean', 'FLT', 'PFR', 'PrimeNumberTheoremAnd', 'Compfiles', 'Debate', 
        'Mathematics in Lean Source', 'Lean4Lean', 'Matrix Cookbook', 'Math Workshop', 
        'LeanEuclid', 'Foundation', 'Con-nf', 'Saturn'
    ],
    'Validation R@10 Exp2': [71.27, 59.29, 57.57, 53.16, 51.42, 52.21, 51.93, 52.44, 51.71, 52.08, 53.9, 52.94, 52.7, 51.81],
    'Average Test R@10 Exp2': [73.86, 65.43, 61.53, 61.29, 58.67, 58.03, 57.94, 57, 57.05, 56.27, 56.57, 56.31, 56.41, 56.32],
    'task1 Test R@10: SciLean': [
        73.85920584286112,
        70.4579133925345,
        68.316469521893,
        67.90886820790239,
        66.26775857459364,
        65.1863329835098,
        66.11454151758758,
        65.49267947670623,
        66.0915207460527,
        65.60843234691674,
        65.87934271440959,
        65.105183944337,
        65.68838438266373,
        66.84521562307594,
    ],
    'task2 Test R@10: FLT': [
        60.41022737534429,
        58.987313433920406,
        60.87678226114952,
        60.47232549716421,
        60.562633281826464,
        62.29210323087069,
        61.56478574490388,
        63.70063625691466,
        63.86741708869263,
        63.86150251891875,
        63.381070179393404,
        63.89301350369212,
        64.95940392339118,
    ],
    'task3 Test R@10: PFR': [
        57.30030047280151,
        59.60314627209562,
        59.46457185947182,
        59.83230331244399,
        61.59515295512789,
        61.56510787163324,
        63.48492167024828,
        63.22532833530327,
        63.98100550475312,
        63.543777206435635,
        63.83561769953895,
        64.34717610079744,
    ],
    'task4 Test R@10: PrimeNumberTheoremAnd': [
        56.777902241002934,
        54.823681559951574,
        55.28315697194416,
        55.27295565053002,
        54.85060753939472,
        55.2432804360722,
        54.75311062124564,
        54.27678025361091,
        54.02412199265747,
        54.47915924260318,
        54.0540391613046,
    ],
    'task5 Test R@10: Compfiles': [
        52.32093982804993,
        53.64640835094869,
        53.89929200176733,
        54.430799371670346,
        55.38139247708932,
        55.708592210789874,
        54.48909584810797,
        53.675278891116704,
        53.86301247371078,
        53.3014804609546,
    ],
    'task6 Test R@10: Debate': [
        53.640941545984965,
        52.992856469278635,
        52.57001676598133,
        52.3270635085884,
        51.748206406397614,
        52.98268665015553,
        53.932115533594384,
        54.08160982377089,
        54.43214844918658,
    ],
    'task7 Test R@10: Mathematics in Lean Source': [
        53.41737552513791,
        52.98718251289621,
        52.62259373666953,
        52.05657401186898,
        53.20380574025763,
        54.00296153515336,
        54.28289011583058,
        54.72063043294029,
    ],
    'task8 Test R@10: Lean4Lean': [
        52.57595336920967,
        52.3039034557376,
        52.03838690505675,
        53.0743508110164,
        53.910932518608156,
        54.42548951684667,
        54.75246126175249,
    ],
    'task9 Test R@10: Matrix Cookbook': [
        52.31907021679297,
        51.89741035540292,
        53.04600967290217,
        53.93221347411854,
        54.21754791276364,
        54.49382811769489,
    ],
    'task10 Test R@10: Math Workshop': [
        51.754142076086914,
        53.093740827890315,
        53.84223557275687,
        54.266993672535214,
        54.61534823991611,
    ],
    'task11 Test R@10: LeanEuclid': [
        54.38031392149536,
        53.591737974422905,
        53.77143829106941,
        53.39120103554753,
    ],
    'task12 Test R@10: Foundation': [
        52.81997397554872,
        53.3280994935056,
        52.87695204243097,
    ],
    'task13 Test R@10: Con-nf': [
        53.24711069813978,
        52.74557168830668,
    ],
    'task14 Test R@10: Saturn': [
        52.9463950036271,
    ],
}

# Data for Experiment 4
data_exp4 = {
    'Repository': [
        'Compfiles', 'Mathematics in Lean Source', 'PrimeNumberTheoremAnd', 'Math Workshop', 
        'FLT', 'PFR', 'SciLean', 'Debate', 'Matrix Cookbook', 'Con-nf', 
        'Foundation', 'Saturn', 'LeanEuclid', 'Lean4Lean'
    ],
    'Validation R@10 Exp4': [61.55, 57.03, 61.23, 59.21, 57.58, 58.22, 53.68, 57.61, 57.92, 54.45, 54.03, 53.25, 53.31, 53.22],
    'Average Test R@10 Exp4': [60.65, 58.24, 57.76, 57.62, 58.51, 58.42, 56.87, 57.05, 58.86, 57.6, 57.56, 56.8, 57.05, 56.96],
    'task1 Test R@10: Compfiles': [
        60.64706363237122,
        60.87058222125128,
        61.20314291705401,
        62.164709408238814,
        62.54625538110046,
        62.605559047049844,
        60.543708430470886,
        60.09202255909037,
        62.74683828569773,
        61.31248900877101,
        61.03120716533379,
        59.28773209128845,
        59.58850886677559,
        59.5471029783468,
    ],
    'task2 Test R@10: Mathematics in Lean Source': [
        55.6103669864236,
        56.94559559462778,
        57.71218888302529,
        58.778048773867475,
        58.55343790728406,
        56.817972704052536,
        57.23680070578614,
        60.405953467167485,
        58.237986812545536,
        57.39587429354147,
        57.14343579464831,
        56.733972106873495,
        57.23261373542922,
    ],
    'task3 Test R@10: PrimeNumberTheoremAnd': [
        55.1297784659227,
        55.49679205065039,
        56.49806013564142,
        56.61584872651873,
        55.781158894609206,
        55.41176036293268,
        58.701211714197974,
        56.62226849996534,
        57.0383467693606,
        56.34606341473738,
        56.2568215744792,
        56.70971367731465,
    ],
    'task4 Test R@10: Math Workshop': [
        55.09533711521228,
        56.418568532501645,
        56.663914194816115,
        55.64087359735254,
        55.17550466330876,
        58.40449656628799,
        56.19372355931239,
        56.405965289173764,
        55.868960454350436,
        55.99890932532983,
        56.57033631547549,
    ],
    'task5 Test R@10: FLT': [
        58.311188371562594,
        58.52064003242252,
        57.26226506559793,
        57.99944465839791,
        59.10945966438392,
        57.07522983264077,
        57.83354632978984,
        57.530887164973635,
        58.22372278714744,
        58.157630531696924,
    ],
    'task6 Test R@10: PFR': [
        57.577593801266694,
        56.863748657417176,
        57.310141786174654,
        59.81506547975275,
        58.363866954850195,
        58.12992455986847,
        58.118525635532805,
        58.61886630372002,
        59.00381545247168,
    ],
    'task7 Test R@10: SciLean': [
        55.20408975337252,
        56.968858799439104,
        56.076343902903346,
        56.98257609472564,
        56.880530237952534,
        56.231141953267574,
        56.33286242121278,
        56.05914533919315,
    ],
    'task8 Test R@10: Debate': [
        56.182715242832174,
        55.50730835841573,
        57.63332095246526,
        58.75252628521179,
        58.02517751155748,
        59.268801819568765,
        58.873660863897314,
    ],
    'task9 Test R@10: Matrix Cookbook': [
        58.9416477035174,
        56.97004063964541,
        57.74373819714964,
        57.39110877582434,
        58.195911232932595,
        58.042438976765474,
    ],
    'task10 Test R@10: Con-nf': [
        56.638492913745445,
        56.53239951219749,
        56.10038045801934,
        56.28020031665864,
        55.959590265934956,
    ],
    'task11 Test R@10: Foundation': [
        55.447689129740915,
        54.900201432618914,
        55.517675266442446,
        55.27021788074264,
    ],
    'task12 Test R@10: Saturn': [
        54.66225870618888,
        55.14346699755861,
        55.11580400855646,
    ],
    'task13 Test R@10: LeanEuclid': [
        55.47509274104077,
        55.28899200709597,
    ],
    'task14 Test R@10: Lean4Lean': [
        55.59152214607567,
    ],
}

# Data for Experiment 9
data_exp9 = {
    'Repository': [
        'SciLean', 'FLT', 'PFR', 'PrimeNumberTheoremAnd', 'Compfiles', 'Debate', 
        'Mathematics in Lean Source', 'Lean4Lean', 'Matrix Cookbook', 'Math Workshop', 
        'LeanEuclid', 'Foundation', 'Con-nf', 'Saturn'
    ],
    'Validation R@10 Exp9': [73.29, 66.44, 58.38, 54.37, 52.62, 51.78, 51.73, 52.28, 51.92, 51.73, 52.11, 50.7, 50.36, 51.11],
    'Average Test R@10 Exp9': [71.2, 69, 63.9, 61.27, 59.8, 57.86, 56.88, 56.6, 55.68, 54.93, 54.51, 54.03, 54.12, 53.72],
    'task1 Test R@10: SciLean': [
        71.20389812006579,
        70.9881839957107,
        66.93344855526337,
        66.02797635931941,
        65.1338413328574,
        63.59588466707295,
        63.72442619036681,
        63.5094477763303,
        63.91606396294816,
        62.476341679730794,
        62.655586252424186,
        63.571478870444466,
        63.64695500975041,
        64.02769788355525,
    ],
    'task2 Test R@10: FLT': [
        67.00248685289951,
        65.81362088927426,
        64.31070190698802,
        63.888435135856035,
        63.67064961631673,
        62.65907483926741,
        64.34004946986377,
        62.353850776306075,
        63.49044591737164,
        62.47514301761207,
        63.37187142293745,
        64.0303169816581,
        63.706809772834525,
    ],
    'task3 Test R@10: PFR': [
        58.95588073993555,
        59.201829372281225,
        61.20067760732025,
        60.57612342105728,
        61.50243984201597,
        61.92053493141635,
        62.868670313314944,
        62.712178397186214,
        62.98636166636735,
        63.85741355159591,
        63.64593201241535,
        64.54276911308345,
    ],
    'task4 Test R@10: PrimeNumberTheoremAnd': [
        55.55015185012409,
        55.40546180329634,
        55.09071724438188,
        54.683870250916335,
        55.197297843744266,
        54.2147986352595,
        54.06254027297892,
        54.03540926653086,
        53.611451168364,
        54.204199465165594,
        54.18700423211805,
    ],
    'task5 Test R@10: Compfiles': [
        53.359487049657325,
        52.36877496119525,
        52.07323078569177,
        52.22631957426211,
        51.81236774316182,
        51.38188897054315,
        51.37288679503196,
        50.91867841182277,
        51.378419331232905,
        50.90558937584463,
    ],
    'task6 Test R@10: Debate': [
        51.858687903990074,
        51.746787066331656,
        51.862990939786414,
        51.42923876809795,
        51.007559757203246,
        51.08534562016248,
        50.60405379775169,
        51.02485378810201,
        50.6480719008607,
    ],
    'task7 Test R@10: Mathematics in Lean Source': [
        51.76797074771562,
        51.94619762728132,
        51.630424292528154,
        51.12436710125164,
        51.0692896263256,
        50.59221375874334,
        51.13808729974287,
        50.66561539926503,
    ],
    'task8 Test R@10: Lean4Lean': [
        51.81514639956859,
        51.531305876503474,
        50.997500266415244,
        50.967696879252145,
        50.406733202029145,
        50.922880480433655,
        50.57832760122596,
    ],
    'task9 Test R@10: Matrix Cookbook': [
        51.39693047924862,
        51.057099627757474,
        51.01984350354044,
        50.42076682437718,
        50.92909787990237,
        50.47972424871923,
    ],
    'task10 Test R@10: Math Workshop': [
        50.97052909666344,
        50.89210591679806,
        50.34839835547995,
        50.761300902337034,
        50.49806052304165,
    ],
    'task11 Test R@10: LeanEuclid': [
        51.06167858385455,
        50.46114191520914,
        50.98073102920689,
        50.56836652154547,
    ],
    'task12 Test R@10: Foundation': [
        50.1432090122897,
        50.751128351257655,
        50.52011100997661,
    ],
    'task13 Test R@10: Con-nf': [
        50.17359788589463,
        50.30489746095695,
    ],
    'task14 Test R@10: Saturn': [
        50.49302781422868,
    ],
}

experiments = {
    'Exp2': data_exp2,
    'Exp4': data_exp4,
    'Exp9': data_exp9,
    # 'Exp10': data_exp10
}


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def calculate_AA(data, exp_name):
    test_accuracies = data[f'Average Test R@10 {exp_name}']
    return test_accuracies[-1]  # The last value represents AA_k

def calculate_AIA(data, exp_name):
    test_accuracies = data[f'Average Test R@10 {exp_name}']
    AA_values = [np.mean(test_accuracies[:i+1]) for i in range(len(test_accuracies))]
    return np.mean(AA_values)

def calculate_metrics(data, exp_name):
    metrics = {}
    
    # 2. Area Under the Learning Curve (AULC)
    aulc = np.trapz(data[f'Average Test R@10 {exp_name}']) / len(data['Repository'])
    metrics['AULC'] = aulc
    
    # 5. Backward Transfer
    backward_transfer = []
    for i in range(1, len(data['Repository'])):
        task_key = f'task{i} Test R@10: {data["Repository"][i-1]}'
        if task_key in data:
            backward_transfer.append(data[task_key][-1] - data[task_key][0])
    metrics['Avg_Backward_Transfer'] = np.mean(backward_transfer)
    
    # Extract the task-specific accuracies
    task_accuracies = {}
    for i in range(1, len(data['Repository']) + 1):
        key = f'task{i} Test R@10: {data["Repository"][i-1]}'
        if key in data:
            task_accuracies[i] = np.array(data[key])

    # Calculate min-ACC
    min_acc_values = []
    for k in range(2, len(task_accuracies) + 1):
        min_acc_sum = 0
        count = 0
        for i in range(1, k):
            if i in task_accuracies and len(task_accuracies[i][k-1:]) > 0:
                min_acc_sum += np.min(task_accuracies[i][k-1:])
                count += 1
        if count > 0:
            min_acc_values.append(min_acc_sum / count)
    
    metrics['min-ACC'] = np.mean(min_acc_values) if min_acc_values else 0

    # Worst-case Accuracy (WC-ACC)
    avg_test = np.array(data[f'Average Test R@10 {exp_name}'])
    tasks = len(avg_test)
    wc_acc_values = []
    for k in range(1, tasks + 1):
        if k == 1:
            wc_acc_values.append(avg_test[0])
        elif k-2 < len(min_acc_values):
            wc_acc = (1/k) * avg_test[k-1] + (1 - 1/k) * min_acc_values[k-2]
            wc_acc_values.append(wc_acc)
    
    metrics['WC-ACC'] = np.mean(wc_acc_values)

    # 3. Windowed Forgetting (WF)
    def calculate_WF(w):
        WF = 0
        for i in range(len(avg_test)):
            if i >= w:
                WF = max(WF, np.max(avg_test[i-w:i]) - avg_test[i])
        return WF
    
    metrics['WF5'] = calculate_WF(5)

    # 7. Windowed Plasticity (WP)
    window_size = 5
    wp_values = [max(0, avg_test[i] - avg_test[max(0, i-window_size)]) for i in range(len(avg_test))]
    metrics['WP5'] = np.mean(wp_values)

    return metrics

def calculate_additional_metrics(data, exp_name):
    metrics = {}
    
    # 3. Forgetting Measure (FM)
    fm_values = []
    for i in range(2, len(data['Repository']) + 1):
        task_key = f'task{i-1} Test R@10: {data["Repository"][i-2]}'
        if task_key in data:
            task_performances = [data[f'task{j} Test R@10: {data["Repository"][j-1]}'][i-j-1] for j in range(1, i) if f'task{j} Test R@10: {data["Repository"][j-1]}' in data]
            fm_values.append(np.max(task_performances) - data[task_key][-1])
    metrics['FM'] = np.mean(fm_values)
    
    # 4. Incremental Plasticity (IP)
    ip_values = np.diff(data[f'Validation R@10 {exp_name}'])
    metrics['IP'] = np.mean(ip_values)
    
    # 10. Time-Weighted Cumulative Performance (TWCP)
    weights = np.arange(len(data['Repository']), 0, -1)
    metrics['TWCP'] = np.sum(weights * np.array(data[f'Average Test R@10 {exp_name}'])) / np.sum(weights)
    
    # 12. Catastrophic Forgetting Resilience (CFR)
    metrics['CFR'] = np.min(data[f'Average Test R@10 {exp_name}']) / np.max(data[f'Average Test R@10 {exp_name}'])
    
    return metrics

def calculate_all_metrics(experiments):
    all_metrics = {}
    for exp_name, data in experiments.items():
        metrics = calculate_metrics(data, exp_name)
        additional_metrics = calculate_additional_metrics(data, exp_name)
        all_metrics[exp_name] = {**metrics, **additional_metrics}
    return all_metrics

lower_is_better_metrics = {'WF5', 'FM'}

def compare_metrics(all_metrics):
    comparison = {}
    for metric in all_metrics['Exp2'].keys():
        values = {exp: metrics[metric] for exp, metrics in all_metrics.items()}
        reverse = metric not in lower_is_better_metrics
        sorted_exps = sorted(values, key=values.get, reverse=reverse)
        sorted_values = [values[exp] for exp in sorted_exps]
        
        # Calculate percentage improvement
        if len(sorted_values) > 1:
            if sorted_values[1] == 0:
                if sorted_values[0] == 0:
                    percent_improvement = 0  # Both values are zero, no improvement
                else:
                    percent_improvement = float('inf')  # Non-zero divided by zero, infinite improvement
            else:
                if metric in lower_is_better_metrics:
                    percent_improvement = ((sorted_values[1] - sorted_values[0]) / sorted_values[1]) * 100
                else:
                    percent_improvement = ((sorted_values[0] - sorted_values[1]) / sorted_values[1]) * 100
        else:
            percent_improvement = 0
        
        comparison[metric] = {
            'Ranking': list(zip(sorted_exps, sorted_values)),
            'Improvement': percent_improvement
        }
    return comparison

def format_comparison(comparison):
    formatted_output = ""
    for metric, result in comparison.items():
        formatted_output += f"{metric}:\n"
        for i, (exp, value) in enumerate(result['Ranking']):
            formatted_output += f"  {i+1}. {exp}: {value:.4f}\n"
        if result['Improvement'] == float('inf'):
            formatted_output += "  Best improvement: inf%\n\n"
        elif metric in lower_is_better_metrics:
            formatted_output += f"  Best improvement: -{result['Improvement']:.2f}%\n\n"
        else:
            formatted_output += f"  Best improvement: +{result['Improvement']:.2f}%\n\n"
    return formatted_output

# Calculate metrics for all experiments
all_metrics = calculate_all_metrics(experiments)

# Compare metrics across experiments
comparison = compare_metrics(all_metrics)

# Print the results
print("Metrics Comparison:")
print(format_comparison(comparison))

