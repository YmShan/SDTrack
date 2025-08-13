MAC = 0
AC = 0

MAC += 173408256     # 第一层
# 头的最后三个卷积层
MAC += 8192
MAC += 16384
MAC += 16384

# convblock1_1
firing_rate = [0.3355, 0.1596, 0.1825, 0.2740, 0.1298]
AC += 150994944 * firing_rate[0]
AC += 115605504 * firing_rate[1]
AC += 150994944 * firing_rate[2]
AC += 2717908992 * firing_rate[3]
AC += 2717908992 * firing_rate[4]

# downsample1_2
firing_rate = [0.2280]
AC += 339738624

# convblock1_2
firing_rate = [0.3033, 0.1955, 0.1634, 0.2799, 0.0720]
AC += 150994944 * firing_rate[0]
AC += 57802752 * firing_rate[1]
AC += 150994944 * firing_rate[2]
AC += 2717908992 * firing_rate[3]
AC += 2717908992 * firing_rate[4]

# downsample2
firing_rate = [0.1706]
AC += 339738624 * firing_rate[0]

# convblock2_1
firing_rate = [0.3810, 0.2103, 0.1547, 0.3140, 0.0673]
AC += 150994944 * firing_rate[0]
AC += 28901376 * firing_rate[1]
AC += 150994944 * firing_rate[2]
AC += 2717908992 * firing_rate[3]
AC += 2717908992 * firing_rate[4]

# convblock2_2
firing_rate = [0.2799, 0.1507, 0.1881, 0.2363, 0.0608]
AC += 150994944 * firing_rate[0]
AC += 28901376 * firing_rate[1]
AC += 150994944 * firing_rate[2]
AC += 2717908992 * firing_rate[3]
AC += 2717908992 * firing_rate[4]

# downsample3
firing_rate = [0.2214]
AC += 339738624 * firing_rate[0]

# block3
# block3.0
firing_rate = [0.2244, 0.2244, 0.2244, 0.1629, 0.2374, 0.0786]
AC += 42467328 * firing_rate[0]
AC += 42467328 * firing_rate[1]
AC += 169869312 * firing_rate[2]
AC += 169869312 * firing_rate[3]
AC += 169869312 * firing_rate[4]
AC += 169869312 * firing_rate[5]

# block3.1
firing_rate = [0.2141, 0.2141, 0.2141, 0.2703, 0.1976, 0.0591]
AC += 42467328 * firing_rate[0]
AC += 42467328 * firing_rate[1]
AC += 169869312 * firing_rate[2]
AC += 169869312 * firing_rate[3]
AC += 169869312 * firing_rate[4]
AC += 169869312 * firing_rate[5]

# block3.2
firing_rate = [0.1865, 0.1865, 0.1865, 0.2678, 0.1813, 0.0416]
AC += 42467328 * firing_rate[0]
AC += 42467328 * firing_rate[1]
AC += 169869312 * firing_rate[2]
AC += 169869312 * firing_rate[3]
AC += 169869312 * firing_rate[4]
AC += 169869312 * firing_rate[5]

# block3.3
firing_rate = [0.1732, 0.1732, 0.1732, 0.2505, 0.1694, 0.0333]
AC += 42467328 * firing_rate[0]
AC += 42467328 * firing_rate[1]
AC += 169869312 * firing_rate[2]
AC += 169869312 * firing_rate[3]
AC += 169869312 * firing_rate[4]
AC += 169869312 * firing_rate[5]

# block3.4
firing_rate = [0.1633, 0.1633, 0.1633, 0.2572, 0.1610, 0.0285]
AC += 42467328 * firing_rate[0]
AC += 42467328 * firing_rate[1]
AC += 169869312 * firing_rate[2]
AC += 169869312 * firing_rate[3]
AC += 169869312 * firing_rate[4]
AC += 169869312 * firing_rate[5]

# block3.5
firing_rate = [0.1573, 0.1573, 0.1573, 0.2503, 0.1551, 0.0320]
AC += 42467328 * firing_rate[0]
AC += 42467328 * firing_rate[1]
AC += 169869312 * firing_rate[2]
AC += 169869312 * firing_rate[3]
AC += 169869312 * firing_rate[4]
AC += 169869312 * firing_rate[5]

# block3.6
firing_rate = [0.1521, 0.1521, 0.1521, 0.2858, 0.1490, 0.0385]
AC += 42467328 * firing_rate[0]
AC += 42467328 * firing_rate[1]
AC += 169869312 * firing_rate[2]
AC += 169869312 * firing_rate[3]
AC += 169869312 * firing_rate[4]
AC += 169869312 * firing_rate[5]

# block3.7
firing_rate = [0.1468, 0.1468, 0.1468, 0.2795, 0.1469, 0.0298]
AC += 42467328 * firing_rate[0]
AC += 42467328 * firing_rate[1]
AC += 169869312 * firing_rate[2]
AC += 169869312 * firing_rate[3]
AC += 169869312 * firing_rate[4]
AC += 169869312 * firing_rate[5]

# block3.8
firing_rate = [0.1451, 0.1451, 0.1451, 0.3075, 0.1433, 0.0393]
AC += 42467328 * firing_rate[0]
AC += 42467328 * firing_rate[1]
AC += 169869312 * firing_rate[2]
AC += 169869312 * firing_rate[3]
AC += 169869312 * firing_rate[4]
AC += 169869312 * firing_rate[5]

# downsample4
firing_rate = [0.1736]
AC += 573308928 * firing_rate[0]

# block4
# block4.0
firing_rate = [0.1639, 0.1639, 0.1639, 0.2940, 0.1880, 0.0482]
AC += 95551488 * firing_rate[0]
AC += 95551488 * firing_rate[1]
AC += 382205952 * firing_rate[2]
AC += 382205952 * firing_rate[3]
AC += 382205952 * firing_rate[4]
AC += 382205952 * firing_rate[5]

# block4.1
firing_rate = [0.1900, 0.1900, 0.1900, 0.1359, 0.1971, 0.0418]
AC += 95551488 * firing_rate[0]
AC += 95551488 * firing_rate[1]
AC += 382205952 * firing_rate[2]
AC += 382205952 * firing_rate[3]
AC += 382205952 * firing_rate[4]
AC += 382205952 * firing_rate[5]

# block4.2
firing_rate = [0.1757, 0.1757, 0.1757, 0.1025, 0.1886, 0.0302]
AC += 95551488 * firing_rate[0]
AC += 95551488 * firing_rate[1]
AC += 382205952 * firing_rate[2]
AC += 382205952 * firing_rate[3]
AC += 382205952 * firing_rate[4]
AC += 382205952 * firing_rate[5]

# box_head
# ctr
firing_rate = [0.1946, 0.1840, 0.1711, 0.2675]
AC += 452984832 * firing_rate[0]
AC += 150994944 * firing_rate[1]
AC += 37748736 * firing_rate[2]
AC += 9437184 * firing_rate[3]

# offset
firing_rate = [0.1946, 0.1632, 0.1827, 0.1048]
AC += 452984832 * firing_rate[0]
AC += 150994944 * firing_rate[1]
AC += 37748736 * firing_rate[2]
AC += 9437184 * firing_rate[3]

# size
firing_rate = [0.1946, 0.1778, 0.1873, 0.3208]
AC += 452984832 * firing_rate[0]
AC += 150994944 * firing_rate[1]
AC += 37748736 * firing_rate[2]
AC += 9437184 * firing_rate[3]

# Q * K
AC += 4438555
AC += 1880224
AC += 1847074
AC += 1840069
AC += 1355186
AC += 1462168
AC += 1379876
AC += 1261376
AC += 1235040
AC += 1235097
AC += 714203
AC += 465303

# QK cache * V
AC += 73186638
AC += 27676971
AC += 23729073
AC += 24081515
AC += 17516465
AC += 19720677
AC += 18954852
AC += 17244291
AC += 17386323
AC += 21363063
AC += 4521796
AC += 3101261



print(4 * (MAC * 4.6 + AC * 0.9), end='')
print('pJ')