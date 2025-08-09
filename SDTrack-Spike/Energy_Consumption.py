MAC = 0
AC = 0

MAC += 86704128     # 第一层
# 头的最后三个卷积层
MAC += 8192
MAC += 8192
MAC += 4096

# convblock1_1
firing_rate = [0.3537, 0.1438, 0.1759, 0.1910, 0.1035]
AC += 37748736 * firing_rate[0]
AC += 28901376 * firing_rate[1]
AC += 37748736 * firing_rate[2]
AC += 679477248 * firing_rate[3]
AC += 679477248 * firing_rate[4]

# downsample1_2
firing_rate = [0.2112]
AC += 84934656 * firing_rate[0]

# convblock1_2
firing_rate = [0.3255, 0.1884, 0.1535, 0.2745, 0.0783]
AC += 37748736 * firing_rate[0]
AC += 28901376 * firing_rate[1]
AC += 37748736 * firing_rate[2]
AC += 679477248 * firing_rate[3]
AC += 679477248 * firing_rate[4]

# downsample2
firing_rate = [0.2528]
AC += 84934656 * firing_rate[0]

# convblock2_1
firing_rate = [0.3704, 0.2049, 0.1431, 0.3291, 0.0722]
AC += 37748736 * firing_rate[0]
AC += 14450688 * firing_rate[1]
AC += 37748736 * firing_rate[2]
AC += 679477248 * firing_rate[3]
AC += 679477248 * firing_rate[4]

# convblock2_2
firing_rate = [0.2882, 0.1511, 0.1817, 0.2628, 0.0676]
AC += 37748746 * firing_rate[0]
AC += 14450688 * firing_rate[1]
AC += 37748746 * firing_rate[2]
AC += 679477248 * firing_rate[3]
AC += 679477248 * firing_rate[4]

# downsample3
firing_rate = [0.2431]
AC += 84934656 * firing_rate[0]

# Block3_0
# firing_rate = []
# AC += 42467328 * firing_rate[0]
# AC += 1492992 * firing_rate[1]
# AC += 42467328 * firing_rate[2]
firing_rate = [0.2912, 0.2912, 0.2912, 0.1492, 0.2897, 0.0745]        # 前三个用的都是第head_spike的firingrate
AC += 10616832 * firing_rate[0]
AC += 10616832 * firing_rate[1]
AC += 42467328 * firing_rate[2]
AC += 42467328 * firing_rate[3]                 # proj_conv 
AC += 42467328 * firing_rate[4]
AC += 42467328 * firing_rate[5]


# Block3_1
# firing_rate = []
# AC += 42467328 * firing_rate[0]
# AC += 1492992 * firing_rate[1]
# AC += 42467328 * firing_rate[2]
firing_rate = [0.2728, 0.2728, 0.2728, 0.2344, 0.2573, 0.0692]
AC += 10616832 * firing_rate[0]
AC += 10616832 * firing_rate[1]
AC += 42467328 * firing_rate[2]
AC += 42467328 * firing_rate[3]
AC += 42467328 * firing_rate[4]
AC += 42467328 * firing_rate[5]

# Block3_2
# firing_rate = []
# AC += 42467328 * firing_rate[0]
# AC += 1492992 * firing_rate[1]
# AC += 42467328 * firing_rate[2]
firing_rate = [0.2543, 0.2543, 0.2543, 0.2737, 0.2467, 0.0594]
AC += 10616832 * firing_rate[0]
AC += 10616832 * firing_rate[1]
AC += 42467328 * firing_rate[2]
AC += 42467328 * firing_rate[3]
AC += 42467328 * firing_rate[4]
AC += 42467328 * firing_rate[5]

# Block3_3
# firing_rate = []
# AC += 42467328 * firing_rate[0]
# AC += 1492992 * firing_rate[1]
# AC += 42467328 * firing_rate[2]
firing_rate = [0.2471, 0.2471, 0.2471, 0.2266, 0.2331, 0.0520]
AC += 10616832 * firing_rate[0]
AC += 10616832 * firing_rate[1]
AC += 42467328 * firing_rate[2]
AC += 42467328 * firing_rate[3]
AC += 42467328 * firing_rate[4]
AC += 42467328 * firing_rate[5]

# Block3_4
# firing_rate = []
# AC += 42467328 * firing_rate[0]
# AC += 1492992 * firing_rate[1]
# AC += 42467328 * firing_rate[2]
firing_rate = [0.2295, 0.2295, 0.2295, 0.2718, 0.2187, 0.0479]
AC += 10616832 * firing_rate[0]
AC += 10616832 * firing_rate[1]
AC += 42467328 * firing_rate[2]
AC += 42467328 * firing_rate[3]
AC += 42467328 * firing_rate[4]
AC += 42467328 * firing_rate[5]

# Block3_5
# firing_rate = []
# AC += 42467328 * firing_rate[0]
# AC += 1492992 * firing_rate[1]
# AC += 42467328 * firing_rate[2]
firing_rate = [0.2127, 0.2127, 0.2127, 0.2776, 0.2088, 0.0559]
AC += 10616832 * firing_rate[0]
AC += 10616832 * firing_rate[1]
AC += 42467328 * firing_rate[2]
AC += 42467328 * firing_rate[3]
AC += 42467328 * firing_rate[4]
AC += 42467328 * firing_rate[5]

# downsample4
firing_rate = [0.2137]
AC += 268738560 * firing_rate[0]

# Block4
# Block4_0
# firing_rate = []
# AC += 83980800 * firing_rate[0]
# AC += 2099520 * firing_rate[1]
# AC += 83980800 * firing_rate[2]
firing_rate = [0.2350, 0.2350, 0.2350, 0.2977, 0.2423, 0.0466]
AC += 20995200 * firing_rate[0]
AC += 20995200 * firing_rate[1]
AC += 83980800 * firing_rate[2]
AC += 83980800 * firing_rate[3]
AC += 83980800 * firing_rate[4]
AC += 83980800 * firing_rate[5]

# Block4_1
# firing_rate = []
# AC += 83980800 * firing_rate[0]
# AC += 2099520 * firing_rate[1]
# AC += 83980800 * firing_rate[2]
firing_rate = [0.2202, 0.2202, 0.2202, 0.2032, 0.1997, 0.0388]
AC += 20995200 * firing_rate[0]
AC += 20995200 * firing_rate[1]
AC += 83980800 * firing_rate[2]
AC += 83980800 * firing_rate[3]
AC += 83980800 * firing_rate[4]
AC += 83980800 * firing_rate[5]

# box_head
# ctr
firing_rate = [0.2081, 0.1972, 0.2069, 0.2084]
AC += 106168320 * firing_rate[0]
AC += 37748736 * firing_rate[1]
AC += 9437184 * firing_rate[2]
AC += 2359296 * firing_rate[3]

# offset
firing_rate = [0.2081, 0.2003, 0.2115, 0.2294]
AC += 106168320 * firing_rate[0]
AC += 37748736 * firing_rate[1]
AC += 9437184 * firing_rate[2]
AC += 2359296 * firing_rate[3]

# size
firing_rate = [0.2081, 0.2101, 0.2064, 0.2063]
AC += 106168320 * firing_rate[0]
AC += 37748736 * firing_rate[1]
AC += 9437184 * firing_rate[2]
AC += 2359296 * firing_rate[3]

# Q * K
AC += 2470033
AC += 1133355
AC += 975112
AC += 1106898
AC += 785140
AC += 812550
AC += 710353
AC += 433240


# QK_Cache * V
AC += 44982358
AC += 15525178
AC += 11519087
AC += 16147058
AC += 7494488
AC += 9898700
AC += 10047538
AC += 3122735

print(MAC * 4.6 + AC * 0.9, end='')
print('pJ')