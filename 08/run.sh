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
#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True  --dropout .2
#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True  --dropout .4

# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True --droppout .1
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True  --droppout .2
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --bn True  --droppout .3
# ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell GRU --rnn_cell_dim 200 --we_dim 64 --cle_dim 32 --epochs 10 --dropout .4 --bn True

# .15
ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 64 --learning_rate .001  --we_dim 64 --cle_dim 32 --epochs 10 --dropout 0  --bn False  --optimizer Adam

#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 64 --learning_rate .001 --learning_rate_final .0001  --we_dim 64 --cle_dim 32 --epochs 10 --dropout 0  --bn False  --optimizer Adamcase 

#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 64 --learning_rate .001 --learning_rate_final .0001  --we_dim 64 --cle_dim 32 --epochs 2 --dropout 0  --bn True --optimizer Adam

#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 64 --learning_rate .001 --learning_rate_final .0001  --we_dim 64 --cle_dim 32 --epochs 2 --dropout .4  --bn False --optimizer Adam 

#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 64 --learning_rate .001 --learning_rate_final .0001  --we_dim 64 --cle_dim 32 --epochs 2 --dropout .4  --bn True --optimizer Adam

#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 64 --learning_rate .001 --learning_rate_final .0001  --we_dim 64 --cle_dim 32 --epochs 2 --dropout .4  --bn True  --optimizer Adam --clip_gradient .75



#ulimit -t unlimited && nice -n 19 python3 tagger_sota.py --rnn_cell LSTM --rnn_cell_dim 12 --learning_rate .01 --learning_rate_final .0001  --we_dim 12 --cle_dim 12 --epochs 2 --dropout 0  --bn True --clip_gradient .75  --optimizer Adam



