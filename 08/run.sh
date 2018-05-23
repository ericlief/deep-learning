#!/bin/bash
#javac -d ../bin *.java
#ulimit -t unlimited && nice -n 19 java -cp ../bin CuckooTabRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin CuckooMultShiftRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPModRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPMultShiftRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPTabRandTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPMultShiftSeqTest
#ulimit -t unlimited && nice -n 19 java -cp ../bin LPTabSeqTest


# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True
#limit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True 



#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn False --dropout .2
#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn False --dropout .4
ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True  --dropout .2
#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True  --dropout .4

# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True --droppout .1
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True  --droppout .2
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True  --droppout .3
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --dropout .4 --bn True






