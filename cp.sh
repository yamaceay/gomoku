function logs() {
	echo "bin/logs/8_8_5_0$1.log" 
}

function best() {
	echo "bin/models/best_8_8_5_0$1.model" 
}

function buf() {
	echo "bin/buf_8_8_5_0$1.pkl" 
}

function curr() {
	echo "bin/models/curr_8_8_5_0$1.model" 
}

for file in logs buf best curr; do
	for i in 4; do
		rm -f $($file $i)
	done
done
