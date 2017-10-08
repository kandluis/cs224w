curl http://snap.stanford.edu/data/wiki-Vote.txt.gz -o wiki-Vote.txt.gz -s
curl http://snap.stanford.edu/class/cs224w-data/hw0/stackoverflow-Java.txt.gz -o stackoverflow-Java.txt.gz -s
gunzip wiki-Vote.txt.gz
gunzip stackoverflow-Java.txt.gz
rm wiki-Vote.txt.gz
rm stackoverflow-Java.txt.gz