#ulimit -t unlimited && nice -n 19 python3 uppercase.py --window 4 --activation relu --optimizer Adam --dropout .4 
#ulimit -t unlimited && nice -n 19 python3 uppercase.py --window 4 --activation relu --optimizer Adam --dropout .4 --layers 2
#ulimit -t unlimited && nice -n 19 python3 uppercase.py --window 6 --activation relu --optimizer Adam --dropout .4 --layers 2
ulimit -t unlimited && nice -n 19 python3 uppercase.py --window 8 --activation relu --optimizer Adam --dropout .4 --layers 4
