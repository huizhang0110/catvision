
RES=""
for n in $(ibstat | grep InfiniBand -B 16 | grep "Rate: 200" -B 10 | grep -oP "\'[a-z0-9_]+\'" | sed 's/^.//;s/.$//' | sort -t '_' -k2 -n);
do
        RES+=$n:1/IB,
done

echo =$RES | sed 's/.$//'
