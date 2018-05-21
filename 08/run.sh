#!/bin/bash
#javac -d ../bin *.java
#ulimit -t unlimited && nice -n 19 java -cp ../bin CuckooTabRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin CuckooMultShiftRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPModRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPMultShiftRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPTabRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPMultShiftSeqTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPTabSeqTest


ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 128 --we_dim 64 --cle_dim 32 --epochs 10 

