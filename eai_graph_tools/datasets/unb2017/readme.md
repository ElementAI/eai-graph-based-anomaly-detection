# UNB2017

Download link: 

### Info: 

"""
+---------------------------+---------------------+---------------------+
|           Label           |      First Seen     |      Last Seen      |
+---------------------------+---------------------+---------------------+
|           BENIGN          | 2017-07-03 01:00:01 | 2017-07-07 12:59:00 |
|        SSH-Patator        | 2017-07-04 02:09:00 | 2017-07-04 03:11:00 |
|        FTP-Patator        | 2017-07-04 09:17:00 | 2017-07-04 10:30:00 |
|       DoS slowloris       | 2017-07-05 02:24:00 | 2017-07-05 10:11:00 |
|         Heartbleed        | 2017-07-05 03:12:00 | 2017-07-05 03:32:00 |
|      DoS Slowhttptest     | 2017-07-05 10:15:00 | 2017-07-05 10:37:00 |
|          DoS Hulk         | 2017-07-05 10:43:00 | 2017-07-05 11:07:00 |
|       DoS GoldenEye       | 2017-07-05 11:10:00 | 2017-07-05 11:19:00 |
|        Infiltration       | 2017-07-06 02:19:00 | 2017-07-06 03:45:00 |
|  Web Attack  Brute Force  | 2017-07-06 09:15:00 | 2017-07-06 10:00:00 |
|      Web Attack  XSS      | 2017-07-06 10:15:00 | 2017-07-06 10:35:00 |
| Web Attack  Sql Injection | 2017-07-06 10:40:00 | 2017-07-06 10:42:00 |
|          PortScan         | 2017-07-07 01:05:00 | 2017-07-07 03:23:00 |
|            DDoS           | 2017-07-07 03:56:00 | 2017-07-07 04:16:00 |
|            Bot            | 2017-07-07 09:34:00 | 2017-07-07 12:59:00 |
+---------------------------+---------------------+---------------------+
Those ranges are interleaved with BENIGN data (the unb2017.h5 dataframe contains 2830743 rows, making it unpractical to
fully depict here!).
"""

#### Fields
`unb_2017_all_dtypes`
Flow ID object
Source IP object
Source Port int64
Destination IP object
Destination Port int64
Protocol int64
Timestamp object
Flow Duration int64
Total Fwd Packets int64
Total Backward Packets int64
Total Length of Fwd Packets float64
Total Length of Bwd Packets float64
Fwd Packet Length Max int64
Fwd Packet Length Min int64
Fwd Packet Length Mean float64
Fwd Packet Length Std float64
Bwd Packet Length Max int64
Bwd Packet Length Min int64
Bwd Packet Length Mean float64
Bwd Packet Length Std float64
Flow Bytes/s object
Flow Packets/s object
Flow IAT Mean float64
Flow IAT Std float64
Flow IAT Max float64
Flow IAT Min float64
Fwd IAT Total float64
Fwd IAT Mean float64
Fwd IAT Std float64
Fwd IAT Max float64
Fwd IAT Min float64
Bwd IAT Total float64
Bwd IAT Mean float64
Bwd IAT Std float64
Bwd IAT Max float64
Bwd IAT Min float64
Fwd PSH Flags int64
Bwd PSH Flags int64
Fwd URG Flags int64
Bwd URG Flags int64
Fwd Header Length int64
Bwd Header Length int64
Fwd Packets/s float64
Bwd Packets/s float64
Min Packet Length int64
Max Packet Length int64
Packet Length Mean float64
Packet Length Std float64
Packet Length Variance float64
FIN Flag Count int64
SYN Flag Count int64
RST Flag Count int64
PSH Flag Count int64
ACK Flag Count int64
URG Flag Count int64
CWE Flag Count int64
ECE Flag Count int64
Down/Up Ratio int64
Average Packet Size float64
Avg Fwd Segment Size float64
Avg Bwd Segment Size float64
Fwd Header Length.1 int64
Fwd Avg Bytes/Bulk int64
Fwd Avg Packets/Bulk int64
Fwd Avg Bulk Rate int64
Bwd Avg Bytes/Bulk int64
Bwd Avg Packets/Bulk int64
Bwd Avg Bulk Rate int64
Subflow Fwd Packets int64
Subflow Fwd Bytes int64
Subflow Bwd Packets int64
Subflow Bwd Bytes int64
`Init_Win_bytes_forward` int64
`Init_Win_bytes_backward` int64
`act_data_pkt_fwd` int64
`min_seg_size_forward` int64
Active Mean float64
Active Std float64
Active Max float64
Active Min float64
Idle Mean float64
Idle Std float64
Idle Max float64
Idle Min float64
Label object
